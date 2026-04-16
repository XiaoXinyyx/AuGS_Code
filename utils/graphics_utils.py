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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getView2World(R, cam_pos):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = cam_pos
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def frustum_cull(points, proj):
    """
        points: [N, 3]
        proj: [4, 4]
        return: True if point is not culled. [N,]
    """
    P = torch.ones(points.shape[0], 1).to(device=points.device)
    points_hom = torch.cat([points, P], dim=1)
    points_proj = torch.matmul(points_hom, proj)
    
    mask = (points_proj[:, 0] + points_proj[:, 3] >= 0) & \
           (points_proj[:, 0] - points_proj[:, 3] <= 0) & \
           (points_proj[:, 1] + points_proj[:, 3] >= 0) & \
           (points_proj[:, 1] - points_proj[:, 3] <= 0) & \
           (points_proj[:, 3] != 0) & (points_proj[:, 2] / points_proj[:, 3] >= 0)
           # Cuda rasterizer does not apply far plane culling.
           # (points_proj[:, 2] - points_proj[:, 3] <= 0)
    
    return mask

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def depth2view(pix_coords:torch.Tensor, depth, z_near, fov_x, fov_y, W, H):
    """
        Get view/camera space coordinates according to pixel coordinates and depth
        pix_coords: [N, 2]
        depth: z coordinate in camera space [N,]
        return: [N, 3]
    """
    # NDC: [-1, 1] -> Pix [-0.5, L - 0.5]
    def Pix2NDC(pix, L):
        return (2.0 * pix + 1) / L - 1
    
    N = pix_coords.shape[0]

    ndc_x = Pix2NDC(pix_coords[:, 1], W) # [N,]
    ndc_y = Pix2NDC(pix_coords[:, 0], H) # [N,]
    assert ndc_x.min() >= -1 and ndc_x.max() <= 1
    assert ndc_y.min() >= -1 and ndc_y.max() <= 1

    tan_half_fov_x = math.tan(fov_x / 2)
    tan_half_fov_y = math.tan(fov_y / 2)

    r = tan_half_fov_x * z_near
    t = tan_half_fov_y * z_near

    # Position in view/camera space
    pos_view = torch.zeros(size=(N, 3)).float().to(device=pix_coords.device)
    pos_view[:, 0] = r * ndc_x * depth / z_near
    pos_view[:, 1] = t * ndc_y * depth / z_near
    pos_view[:, 2] = depth
 
    return pos_view

def find_normal(points):
    """
        Find normal vector of a set of points in 3D space.
        points: [N, 3]
        return: [3,]
    """
    # Normalization
    mean = torch.mean(points, axis=0)
    points = points - mean

    U, _, _ = torch.svd(points.T)

    # The normal vector is the last column of V
    return U[:, -1] # [3,]

def depth_to_normal(camera, depth):
    """
        camera: view camera
        depth: depthmap (z coordinates in view space)
        return : normal map [3, H, W]
    """
    W, H = camera.image_width, camera.image_height
    pix_coords = torch.meshgrid(torch.arange(H, device='cuda').float(), torch.arange(W, device='cuda').float(), indexing='ij')
    pix_coords = torch.cat([pix_coords[0].reshape(-1, 1), pix_coords[1].reshape(-1, 1)], dim=1)
    posView = depth2view(pix_coords, depth.reshape(-1), camera.znear, camera.FoVx, camera.FoVy, W, H)

    # World space positions [3, H, W]
    view_world_transform = torch.tensor(
        getView2World(camera.R, camera.camera_center.cpu().numpy())
    ).transpose(0, 1).cuda().double()
    posWorld = geom_transform_points(posView.double(), view_world_transform)
    posWorld = posWorld.reshape(H, W, 3)

    dx = torch.empty(size=(H, W, 3), device='cuda', dtype=torch.float32)
    dy = torch.empty(size=(H, W, 3), device='cuda', dtype=torch.float32)
    dx[:, :-1] = posWorld[:, 1:] - posWorld[:, :-1]
    dx[:, -1:] = dx[:, -2:-1]
    dy[:-1, :] = posWorld[1:, :] - posWorld[:-1, :]
    dy[-1:, :] = dy[-2:-1, :]

    normal_map = torch.nn.functional.normalize(torch.cross(dy, dx, dim=-1), dim=-1)
    normal_map = normal_map.permute(2, 0, 1).contiguous()
    return normal_map