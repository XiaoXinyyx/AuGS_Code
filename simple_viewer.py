"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import time
from typing import Tuple

import nerfview
import torch
import torch.nn.functional as F
import viser

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from scene.gaussian_model import GaussianModel


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.model is None:
        print("Please specify the path to the point cloud with --model")
        return

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    # TODO specify sh_degree in the command line
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(args.model)

    means = gaussians._xyz.detach()
    quats = F.normalize(gaussians._rotation, p=2, dim=-1).detach()
    scales = torch.exp(gaussians._scaling).detach()
    opacities = torch.sigmoid(gaussians._opacity).clone().flatten().detach()
    opacities_buffer = torch.sigmoid(gaussians._opacity).flatten().detach()
    sh0 = gaussians._features_dc.detach()
    shN = gaussians._features_rest.detach()
    colors = torch.cat([sh0, shN], dim=-2).detach()
    opa_dir = F.normalize(gaussians._opa_dir, p=2, dim=-1).detach() 
    cos_theta = torch.cos(gaussians._theta.clamp(max=torch.pi)).detach()
    beta = gaussians._beta.detach()

    aug_vertex_count = cos_theta.numel()
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    del gaussians
    torch.cuda.empty_cache()
    # # crop
    # aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
    # edges = aabb[3:] - aabb[:3]
    # sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    # sel = torch.where(sel)[0]
    # means, quats, scales, colors, opacities = (
    #     means[sel],
    #     quats[sel],
    #     scales[sel],
    #     colors[sel],
    #     opacities[sel],
    # )

    # # repeat the scene into a grid (to mimic a large-scale setting)
    # repeats = args.scene_grid
    # gridx, gridy = torch.meshgrid(
    #     [
    #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
    #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
    #     ],
    #     indexing="ij",
    # )
    # grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
    #     -1, 3
    # )
    # means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    # means = means.reshape(-1, 3)
    # quats = quats.repeat(repeats**2, 1)
    # scales = scales.repeat(repeats**2, 1)
    # colors = colors.repeat(repeats**2, 1, 1)
    # opacities = opacities.repeat(repeats**2)
    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, render_tab_state):
        width, height = render_tab_state.viewer_width, render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        # View-dependent opacities
        if aug_vertex_count > 0:
            camera_center = c2w[:3, 3].unsqueeze(0)  # [1, 3]
            view_dir = camera_center - means[-aug_vertex_count:]  # [aug_v, 3]
            view_dir = F.normalize(view_dir, dim=-1)  # [aug_v, 3]
            odv = torch.sum(opa_dir * view_dir, dim=-1)  # [aug_v, ]
            omx = (odv - cos_theta) / (1.0 - cos_theta)  # [aug_v, ]

            valid_mask = omx >= 0.0001
            opa_atten = torch.zeros(aug_vertex_count, device=means.device, dtype=means.dtype)
            opa_atten[valid_mask] = torch.pow(omx[valid_mask], torch.exp(beta[valid_mask]))
            opacities_buffer[-aug_vertex_count:] = opacities[-aug_vertex_count:] * opa_atten

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities_buffer,  # [N]
            colors,  # [N, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
            packed=False,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="path to the .ply file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
