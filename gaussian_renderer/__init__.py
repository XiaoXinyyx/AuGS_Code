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

import math

import torch
from torch.nn import functional as F
from gsplat import rasterization
from scene.gaussian_model import GaussianModel

def compute_opacity(opacity, opa_dir, theta, beta, camera_center, means3D):
    aug_vertex_count = beta.numel()

    view_dir = camera_center[None, ...] - means3D[-aug_vertex_count:]
    view_dir = F.normalize(view_dir, dim=-1)
    odv = torch.sum(opa_dir * view_dir, dim=-1) # [aug_vertex_count, ]
    
    # cos_theta = torch.cos(theta)         # [aug_vertex_count, ]
    # omx = (odv - cos_theta) / (1.0 - cos_theta) # [aug_vertex_count, ]
    # valid_mask = omx >= 0.0001
    # opa_atten = torch.zeros(aug_vertex_count, device=means3D.device, dtype=means3D.dtype)
    # opa_atten[valid_mask] = torch.pow(omx[valid_mask], torch.exp(beta[valid_mask]))

    alpha = torch.acos(odv.clip(-0.9999, 0.9999)) # Safe acos
    base = 0.5 * (torch.cos(torch.clip(alpha / theta, 0, torch.pi)) + 1)
    valid_mask = base >= 0.00001 # Safe pow
    opa_atten = torch.zeros(aug_vertex_count, device=means3D.device, dtype=means3D.dtype)
    opa_atten[valid_mask] = torch.pow(base[valid_mask], torch.exp(beta[valid_mask]))

    if opacity.requires_grad:
        final_opacity = torch.empty_like(opacity)
        final_opacity[:-aug_vertex_count] = opacity[:-aug_vertex_count]
        final_opacity[-aug_vertex_count:] = opacity[-aug_vertex_count:] * opa_atten.reshape(-1,1)
        opacity = final_opacity
    else:
        opacity[-aug_vertex_count:] *= opa_atten.reshape(-1,1)
    return opacity

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor,
           scaling_modifier=1.0, override_color=None, render_depth=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2.0 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2.0 * tanfovy)
    K = torch.tensor(
        [   # Camera intrinsics matrix
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )
    
    render_mode = "RGB+MD" if render_depth else "RGB"

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # View dependent opacity
    if pc.get_beta.numel() > 0:
        opacity = compute_opacity(opacity,
                                pc.get_opacity_dir, pc.get_theta, pc.get_beta,
                                viewpoint_camera.camera_center, means3D,)

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False, # Turn off memory efficient mode
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0,:,:,:3].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    is_used = info["is_used"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    depth = render_colors[0, :, :, 3:4].permute(2, 0, 1) if render_depth else None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
   
    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth" : depth,
        "is_used" : is_used,
    }

