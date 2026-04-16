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

import traceback
import os
import datetime

import torch.multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from fused_ssim import fused_ssim

from gaussian_renderer import render
from gaussian_2d_renderer import Gaussian2DModel, render_2d
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.loss_utils import ssim, ssim_2D, l1_loss
from utils.image_utils import psnr, save_images_multithread
from utils.general_utils import safe_state
from gaussian_2d_renderer import GaussianProjector

OPACITY_INIT = 0.02 # 0.02
BETA_VALUE = 0.0

COARSE_OPT_STEP = 200 # 400
FINETUNE_OPT_STEP = 1000 # 1000

lambda_dssim = 0.2

def loss_2d(img, gt_img, lambda_dssim):
    l1 = torch.abs((img - gt_img)).mean(dim=0, keepdim=True)
    loss_2D = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_2D(img, gt_img))
    return loss_2D

def max_2d(tensor):
    if tensor.dim() != 2:
        raise ValueError('Input tensor must be 2D')
    v, idx = torch.max(tensor, 0)
    M, col = torch.max(v, 0)
    col = int(col)
    row = int(idx[col])
    return M, row, col

def exclusive_sum(x):
    inclusive_cumsum = torch.cumsum(x, dim=0)
    return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), inclusive_cumsum[:-1]))

def prepare_logger(args):    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(
            args.model_path, "Log",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def optimization_process(steps, render_img, gt_img, gaussians: Gaussian2DModel,
                         tb_writer=None, prune_points=False):
    if gaussians.size() == 0:
        return

    for _ in range(steps):
        render_pkg = render_2d(render_img, gaussians)
        refined_img = render_pkg["render"]

        Ll1 = l1_loss(refined_img, gt_img)
        ssim_value = fused_ssim(refined_img.unsqueeze(0), gt_img.unsqueeze(0))

        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()

        gaussians.end_optimizer_step()

    if prune_points:
        valid_mask = (gaussians.get_opacity.flatten() > 0.005) & (render_pkg["pix_covered"] > 25)
        gaussians.prune_points(valid_mask)


def train_2d_kernels(args, gaussian_count, render_images, gt_images, depth_maps):
    print("Optimizing 2D kernels for " + args.model_path)

    tb_writer = prepare_logger(args)

    assert len(render_images) == len(gt_images)

    # [C, H, W]
    N = len(gt_images)
    H = gt_images[0].shape[1]
    W = gt_images[0].shape[2]

    # Create 2d gaussian models
    gaussian_models = [Gaussian2DModel(depth) for depth in depth_maps]
    for gm in gaussian_models:
        gm.training_setup(args, (H + W) / 2)

    # Distribute gaussians to each image
    temp_loss_list = torch.zeros(N, dtype=torch.float64)
    for img_idx in range(N):
        Ll1 = l1_loss(render_images[img_idx], gt_images[img_idx])
        ssim_value = ssim(render_images[img_idx], gt_images[img_idx])
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
        temp_loss_list[img_idx] = loss.clamp(min=0)
    temp_loss_list /= temp_loss_list.sum()
    kernel_count_per_img = torch.ceil(temp_loss_list * gaussian_count).int()

    # Add gaussians to the image with the highest loss
    for img_idx in tqdm(range(N)):
        gaussians = gaussian_models[img_idx]
        target_kernel_count = kernel_count_per_img[img_idx].item()

        # Initialize loss_2d and refined_img
        with torch.no_grad():
            refined_img = render_2d(render_images[img_idx], gaussians)["render"]
            loss_2D = loss_2d(refined_img, gt_images[img_idx], lambda_dssim).clamp(min=0)

        done = False
        while not done:
            with torch.no_grad():
                # Initialize new gaussian kernels
                new_coords = torch.multinomial(loss_2D.flatten() ** 2,
                                  min(round(target_kernel_count * 0.2), target_kernel_count - gaussians.size()))
                new_coords = torch.stack((new_coords // W, new_coords % W), dim=1)
                new_kernel_count = new_coords.shape[0]

                mean2D = new_coords.float().requires_grad_(True)
                scale = gaussians.inverse_scale_activation(
                    torch.full(size=(new_kernel_count, 2), fill_value= min(W, H) * 0.005, dtype=torch.float32
                            ).cuda()).requires_grad_(True)
                rotation = torch.zeros((new_kernel_count, 1)).cuda().requires_grad_(True)
                opacity = gaussians.inverse_opacity_activation(
                    torch.full(size=(new_kernel_count, 1), fill_value=OPACITY_INIT, dtype=torch.float32).cuda()
                    ).requires_grad_(True)
                color = (refined_img[:,new_coords[:,0], new_coords[:,1]] ).clamp_(1/255, 244/255).t().contiguous().requires_grad_(True)

                gaussians.add_gaussian_kernels(mean2D, color, opacity, scale, rotation)
                if gaussians.size() >= target_kernel_count:
                    done = True

            optimization_process(
                COARSE_OPT_STEP, render_images[img_idx], gt_images[img_idx],
                gaussians, tb_writer, prune_points=True)

            # Render and update
            refined_img = render_2d(render_images[img_idx], gaussians)["render"]
            loss_2D = loss_2d(refined_img, gt_images[img_idx], lambda_dssim).clamp(min=0)

            # log
            with torch.no_grad():
                if tb_writer is not None:
                    # Loss
                    Ll1 = l1_loss(refined_img, gt_images[img_idx])
                    ssim_value = fused_ssim(refined_img.unsqueeze(0), gt_images[img_idx].unsqueeze(0))
                    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)

                    tb_writer.add_scalar(f'img_{img_idx}/loss_l1', Ll1.item(), gaussians.size())
                    tb_writer.add_scalar(f'img_{img_idx}/ssim', ssim_value.item(), gaussians.size())
                    tb_writer.add_scalar(f'img_{img_idx}/loss_combined', loss.item(), gaussians.size())

        # Finetune kernels
        optimization_process(
            FINETUNE_OPT_STEP, render_images[img_idx], gt_images[img_idx],
            gaussians, prune_points=True)

    print("Optimization finished. Saving results.")

    os.makedirs(f'{args.model_path}/train/ours_{args.iterations}', exist_ok=True)
    torch.save([gs.capture() for gs in gaussian_models], 
               f'{args.model_path}/train/ours_{args.iterations}/gaussians_2d.pt')
    print(f"Save 2D kernels to {args.model_path}/train/ours_{args.iterations}/gaussians_2d.pt")

    return

def projection_task(args):
    depth_map, projector = args
    kernel_count = projector.gaussians_2d.size()
    if kernel_count == 0:
        return None

    try:
        params = projector.project(shared_cam_centers, shared_projs, shared_pix_coords,
                                   depth_map, projector.opa_inv_act, projector.scale_inv_act)
        return [projector.cam_idx, kernel_count, params]
    except Exception as e:
        print(f"Error in projection task: {e}. camera idx: {projector.cam_idx}")
        traceback.print_exc()
        return None


def init_worker(centers, proj_transforms, pix_coords):
    global shared_cam_centers
    global shared_projs
    global shared_pix_coords
    shared_cam_centers = centers
    shared_projs = proj_transforms
    shared_pix_coords = pix_coords

# Project 2D kernels to 3D space. Add to gaussian model
def project_2d_kernels(args, dataset: ModelParams, depth_maps):
    workers = min(os.cpu_count() - 1, args.workers)

    # Load 2d gaussian kernels
    with torch.no_grad():
        model_args = torch.load(f'{args.model_path}/train/ours_{args.iterations}/gaussians_2d.pt')
        gaussian_2d_models = [Gaussian2DModel().restore(arg).cpu_() for arg in model_args]
        del model_args

    # Load 3d gaussian model from point_cloud.ply
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iterations, shuffle=False)

    # Initialize parameters of additional 3D gaussians
    N = sum([m.size() for m in gaussian_2d_models])
    new_xyz = torch.zeros(size=(N, 3)).float().cuda()
    new_features_dc = torch.zeros(size=(N,1,3)).float().cuda()
    new_features_rest = torch.zeros(size=(N,gaussians._features_rest.shape[1],3)).float().cuda()
    new_opacity = torch.zeros(size=(N, 1)).float().cuda()
    new_opa_dir = torch.zeros(size=(N, 3)).float().cuda()
    new_theta = torch.full(size=(N, 1), fill_value=-1.0).float().cuda()
    new_beta = torch.full(size=(N, 1), fill_value=BETA_VALUE).float().cuda()
    new_scale = torch.zeros(size=(N, 3)).float().cuda()
    new_rotation = torch.zeros(size=(N, 4)).float().cuda()

    # Projectors
    projectors = []
    for cam_idx, camera in enumerate(scene.getTrainCameras()):
        projectors.append(GaussianProjector(
            gaussian_2d_models[cam_idx], camera, cam_idx, scene.cameras_extent,
            gaussians.inverse_opacity_activation, gaussians.scaling_inverse_activation))
    H = projectors[0].H
    W = projectors[0].W

    # Project 2D kernels to world space parallelly
    print("Projecting 2D kernels to world space...")
    progress_bar = tqdm(total=N)
 
    camera_centers = torch.stack([c.camera_center for c in scene.getTrainCameras()])
    full_proj_transforms = torch.stack([c.full_proj_transform for c in scene.getTrainCameras()])
    pix_coords = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing='ij')
    pix_coords = torch.cat([pix_coords[0].reshape(-1, 1), pix_coords[1].reshape(-1, 1)], dim=1)
    
    # Data in shared memory
    camera_centers = camera_centers.cpu().share_memory_()
    full_proj_transforms = full_proj_transforms.cpu().share_memory_()
    pix_coords = pix_coords.cpu().share_memory_()
    
    offsets = exclusive_sum(torch.tensor([m.size() for m in gaussian_2d_models]))
    descending_idx = np.argsort([m.size() for m in gaussian_2d_models])[::-1]
    with mp.Pool(processes=workers,
                 initializer=init_worker,
                 initargs=(camera_centers, full_proj_transforms, pix_coords)) as pool:
        results = pool.imap_unordered(projection_task,
            [(depth_maps[idx].cpu(), projectors[idx]) for idx in descending_idx])
        
        for result in results:
            if result is None:
                continue

            offset = offsets[result[0]]
            kernel_count = result[1]
            new_xyz[offset:offset+kernel_count, :] = result[2]["xyz"]
            new_features_dc[offset:offset+kernel_count,0, :] = result[2]["features_dc"]
            new_opacity[offset:offset+kernel_count, :] = result[2]["opacity"]
            new_opa_dir[offset:offset+kernel_count, :] = result[2]["opa_dir"]
            new_theta[offset:offset+kernel_count, :] = result[2]["theta"]
            new_scale[offset:offset+kernel_count, :] = result[2]["scale"]
            new_rotation[offset:offset+kernel_count, :] = result[2]["rotation"]
            progress_bar.update(kernel_count)

    progress_bar.close()

    # Add new kernels to the model
    gaussians._xyz = torch.cat((gaussians._xyz, new_xyz), dim=0)
    gaussians._features_dc = torch.cat((gaussians._features_dc, new_features_dc), dim=0)
    gaussians._features_rest = torch.cat((gaussians._features_rest, new_features_rest), dim=0)
    gaussians._opacity = torch.cat((gaussians._opacity, new_opacity), dim=0)
    gaussians._opa_dir = new_opa_dir
    gaussians._theta = new_theta
    gaussians._beta = new_beta
    gaussians._scaling = torch.cat((gaussians._scaling, new_scale), dim=0)
    gaussians._rotation = torch.cat((gaussians._rotation, new_rotation), dim=0)

    # Save modified model
    print(f"Saving refined model.")
    gaussians.save_ply(f'{args.model_path}/point_cloud/iteration_{args.iterations}/aug_point_cloud_init.ply')


def get_rendering_result(dataset: ModelParams, iteration: int):
    result = {
        "render": [],
        "gt": [],
        "depth": [],
    }

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for view in scene.getTrainCameras():
            render_pkg = render(view, gaussians, background, render_depth=True)
            result["render"].append(render_pkg["render"])
            result["gt"].append(view.original_image[0:3, :, :])
            result["depth"].append(render_pkg["depth"])
    
    return result, gaussians.get_xyz.shape[0]


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--skip_train_2d", action="store_true")
    parser.add_argument("--xy_2d_lr", type=float, default=0.001)       # 0.001
    parser.add_argument("--color_2d_lr", type=float, default=0.01)     # 0.01
    parser.add_argument("--opacity_2d_lr", type=float, default=0.02)   # 0.02
    parser.add_argument("--scale_2d_lr", type=float, default=0.001)    # 0.001
    parser.add_argument("--rotation_2d_lr", type=float, default=0.02)  # 0.02
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = get_combined_args(parser)
    print("Training and projecting 2D kernels for " + args.model_path)

    mp.set_start_method('spawn', force=True)

    # Initialize system state (RNG)
    safe_state(silent=args.quiet, seed=args.seed)

    render_results, gaussian_count = get_rendering_result(model.extract(args), args.iterations)
    depth_maps = render_results["depth"]
    torch.cuda.empty_cache()
    
    if not args.skip_train_2d:
        train_2d_kernels(args, gaussian_count * args.ratio, render_results["render"], render_results["gt"], render_results["depth"])
        del render_results
        torch.cuda.empty_cache()

    # Limit the thread number to elevate the efficiency of multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    project_2d_kernels(args, model.extract(args), depth_maps)
