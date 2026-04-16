#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from os import makedirs
import time

import json
import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from scene import Scene
from gaussian_renderer import render
from utils.image_utils import save_exr
from utils.general_utils import safe_state
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from fused_ssim import fused_ssim
from lpipsPyTorch import LPIPS
from utils.image_utils import psnr

def render_set(model_path, name, iteration, views, gaussians, background,
               render_depth, executor, skip_lpips):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if render_depth:
        makedirs(depth_path, exist_ok=True)

    LPIPS_Model = LPIPS('vgg', '0.1').to('cuda')

    psnr_test = torch.zeros(1).cuda()
    ssim_test = torch.zeros(1).cuda()
    lpips_test = torch.zeros(1).cuda()
    result = {}

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, background, render_depth=render_depth)
        rendering, depth = render_pkg["render"], render_pkg["depth"]
        
        gt = view.original_image[0:3, :, :]

        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        if name == "test":
            psnr_test += psnr(rendering, gt).mean()
            ssim_test += fused_ssim(rendering.unsqueeze(0), gt.unsqueeze(0)).mean()
            if not skip_lpips:
                lpips_test += LPIPS_Model(rendering, gt).mean()

        executor.submit(torchvision.utils.save_image, rendering,
                        os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        executor.submit(torchvision.utils.save_image, gt,
                        os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if render_depth:
            executor.submit(save_exr, depth,
                            os.path.join(depth_path, '{0:05d}'.format(idx) + ".exr"))
    
    if name == "test":
        psnr_test /= len(views)
        ssim_test /= len(views)
        lpips_test /= len(views)

    result = {
        "psnr": psnr_test.item(),
        "ssim": ssim_test.item(),
        "lpips": lpips_test.item(),
    }

    # FPS test
    t_list = []
    for _ in range(5):
        for idx, view in enumerate(views):
            torch.cuda.synchronize()
            t_start = time.perf_counter_ns()
            render_pkg = render(view, gaussians, background, render_depth=False)
            torch.cuda.synchronize()
            t_end = time.perf_counter_ns()
            t_list.append(t_end - t_start)
    t = np.array(t_list[5:])
    fps = 1.0 / (t.mean() / 1e9)
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    return result


def render_sets(dataset : ModelParams, iteration : int, skip_train : bool,
                skip_test : bool, render_depth: bool, skip_lpips: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        result = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, background, render_depth,
                        executor, skip_lpips)
                torch.cuda.empty_cache()

            if not skip_test:
                result = render_set(dataset.model_path, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, background, render_depth,
                        executor, skip_lpips)
                torch.cuda.empty_cache()
        
        return result


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.set_grad_enabled(False)

    results = {}
    for iteration in args.iterations:
        result = render_sets(model.extract(args), iteration, args.skip_train,
                             args.skip_test, args.render_depth, args.skip_lpips)
        results[iteration] = result
    
    if not args.skip_test:
        with open(os.path.join(args.model_path, "results_high_pre.json"), 'w') as fp:
            json.dump(results, fp, indent=True)