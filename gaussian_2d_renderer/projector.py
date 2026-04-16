import math

import torch
import torchvision
import numpy as np
from wpca import WPCA
from scipy.spatial.transform import Rotation
from sklearn.cluster import AgglomerativeClustering

from utils.graphics_utils import depth2view, geom_transform_points, getView2World, frustum_cull
from utils.sh_utils import RGB2SH
from utils.general_utils import compute_cov2D

MIN_SCALE = 0.0002 # 0.001

class GaussianProjector:
    def __init__(self, gaussians_2d, camera, cam_idx, camera_extent, opa_inv_act, scale_inv_act):
        self.gaussians_2d = gaussians_2d
        
        self.opa_inv_act = opa_inv_act
        self.scale_inv_act = scale_inv_act

        # Camera
        self.cam_idx = cam_idx
        self.W = camera.image_width
        self.H = camera.image_height
        self.znear = camera.znear
        self.zfar = camera.zfar
        self.FoVx = camera.FoVx
        self.FoVy = camera.FoVy
        self.R = camera.R
        self.world2view = camera.world_view_transform.cpu()
        self.camera_center = camera.camera_center.cpu()
        self.camera_extent = camera_extent
        pass

    def compute_theta(self, posWorld, camera_centers, full_proj_transforms):
        """
            camera_centers: list of camera centers
        """
        N = posWorld.shape[0]
        M = camera_centers.shape[0]

        # Project to each camera. Do frustum culling
        visible_mask = torch.zeros(size=(N, M)).bool() # [N, M]
        for i in range(M):
            visible_mask[:, i] = frustum_cull(posWorld, full_proj_transforms[i])

        # View direction of each kernel. [N, M, 3]
        view_dir = camera_centers.unsqueeze(0) - posWorld.unsqueeze(1) # [1, M, 3] - [N, 1, 3] -> [N, M, 3]
        view_dir /= torch.norm(view_dir, dim=-1, keepdim=True) 

        # Dot product of view directions. 
        dot_prod = torch.zeros(size=(N, M)).float() # [N, M]
        for i in range(N):
            dot_prod[i:i+1] = torch.matmul(view_dir[i:i+1, self.cam_idx, :], view_dir[i, :, :].T)

        # Clamp to [0, 1]
        dot_prod = dot_prod.clamp(min=0.0, max=1.0)

        # Exclude myself and invisible cameras
        dot_prod[:, self.cam_idx] = -1
        dot_prod[~visible_mask] = -1

        # Find the maximum dot product. [N,]
        max_dot = torch.max(dot_prod, dim=-1)[0].clamp(min=0.0, max=1.0)

        theta = torch.acos(max_dot) # [0.0, pi/2]

        if theta.isnan().any():
            nan_mask = torch.isnan(theta)
            nan_indices = torch.nonzero(nan_mask)
            idx = nan_indices[0]
            print(f"Warning! theta is nan. max_dot: {max_dot[idx]}, dot_prod: {dot_prod[idx]}")
        # TODO: convert to cos(theta) to improve performance
        # theta = torch.cos(theta)

        return theta

    def get_rect(self, radius, W, H):
        """
            radius: [N, 1]
            return: rect_min, rect_max # [N, 2]
        """
        mean2D = self.gaussians_2d.get_xy # [N, 2]

        rect_min = mean2D - radius
        rect_min = torch.floor(torch.maximum(rect_min, torch.zeros((1,1), device=radius.device, dtype=torch.float32)))
        rect_max = mean2D + radius + 1
        rect_max = torch.ceil(torch.minimum(rect_max, torch.tensor([[H, W]], device=radius.device, dtype=torch.float32)))

        return rect_min.int(), rect_max.int()

    def compute_rotation(self, pix_coords, depth_map, depths, posWorld):
        posView = depth2view(pix_coords, depth_map.reshape(-1), self.znear, self.FoVx, self.FoVy, self.W, self.H)
        view_world_transform = torch.tensor(
            getView2World(self.R, self.camera_center.numpy())
        ).transpose(0, 1).float()
        world_pos_map = geom_transform_points(posView, view_world_transform)
        world_pos_map = world_pos_map.reshape(self.H, self.W, 3)

        # Compute 2d covariance matrix. [N, 3]
        cov2D = compute_cov2D(self.gaussians_2d.get_scale, self.gaussians_2d.get_rotation)
        h_var = 0.3
        cov2D[:, 0] += h_var
        cov2D[:, 2] += h_var

        # Inverse of covariance matrix
        det = (cov2D[:, 0] * cov2D[:, 2] - cov2D[:, 1] * cov2D[:, 1]).unsqueeze(1) # [N, 1]
        det_inv = 1.0 / det # [N, 1]
        conic = torch.cat([ cov2D[:, 2:3] * det_inv,
                           -cov2D[:, 1:2] * det_inv,
                            cov2D[:, 0:1] * det_inv,], dim=1) # [N, 3]

        # Compute extent in screen space (by finding eigenvalues of
        # 2D covariance matrix). Use extent to compute a bounding rectangle
        mid = 0.5 * (cov2D[:, 0] + cov2D[:, 2]).unsqueeze(1) # [N, 1]
        lambda1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        lambda2 = mid - torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        my_radius = torch.ceil(3.0 * torch.sqrt(torch.maximum(lambda1, lambda2))) # [N, 1]
        rect_min, rect_max = self.get_rect(my_radius, self.W, self.H)

        # Initialize rotations, variances and new depths
        N = self.gaussians_2d.size()
        rotation_matrix = torch.eye(3, dtype=torch.float32).repeat(N,1,1)
        variance = torch.ones((N, 3), dtype=torch.float32)
        new_depths = torch.zeros(N, dtype=torch.float32)

        # Compute rotation matrix using weighted PCA
        xy = self.gaussians_2d.get_xy # [N, 2]
        opacity = self.gaussians_2d.get_opacity # [N, 1]

        cluster_model = AgglomerativeClustering(n_clusters=None, linkage='single')

        # Iterate over each gaussian
        for i in range(N):
            tl, br = rect_min[i], rect_max[i] # [2]
            cur_conic = conic[i] # [3]

            # World positions
            points = world_pos_map[tl[0]:br[0], tl[1]:br[1], :].reshape(-1, 3).to(dtype=torch.float64)
            depth_samples = depth_map[0, tl[0]:br[0], tl[1]:br[1]].reshape(-1, 1)

            # Blending weights
            coords = torch.meshgrid([torch.arange(tl[0], br[0]).float(),
                                    torch.arange(tl[1], br[1]).float()],
                                    indexing='ij')
            coords = torch.stack(coords, dim=2).reshape(-1, 2).contiguous()
            d = xy[i, [1, 0]] - coords[:, [1, 0]]
            power = -0.5 * (cur_conic[0] * d[:, 0] * d[:, 0] + cur_conic[2] * d[:, 1] * d[:, 1]) \
                    - cur_conic[1] * d[:, 0] * d[:, 1]
            assert torch.all(power <= 0)
            # TODO: Consider antialising
            
            weight = torch.clamp(torch.exp(power) * opacity[i], max=0.99).unsqueeze(1)
            valid_mask = (weight >= 1.0 / 255.0).reshape(-1) & (depth_samples.flatten() < self.zfar)
            
            # Keep track of the pixel with maximum weight
            pivot_index = torch.argmax(weight).item()

            # Set up distance threshold
            if valid_mask[pivot_index]:
                dist_threshold = 5 * math.tan(self.FoVx * 0.5) * max(0.2, depths[i].item()) / (self.W * 0.5)
            else:
                new_depth = (depth_samples[valid_mask] * weight[valid_mask] / weight[valid_mask].sum()).sum()
                dist_threshold = 5 * math.tan(self.FoVx * 0.5) * max(0.2, new_depth.item()) / (self.W * 0.5)

            # Clip invalid pixels
            points = points[valid_mask]
            weight = weight[valid_mask]
            depth_samples = depth_samples[valid_mask]

            if points.shape[0] < 3:
                continue

            # Keep track of the pixel with maximum weight
            pivot_index = torch.argmax(weight).item()

            # Set cluster distance threshold according to depth and image resolution
            cluster_model.distance_threshold = dist_threshold
            
            # Clustering
            while True:
                labels = cluster_model.fit_predict(points)
                label_count = np.bincount(labels)
                target_label = labels[pivot_index]

                # No valid cluster. Increase distance threshold
                if max(label_count) < 3: 
                    cluster_model.distance_threshold *= 1.5
                    continue

                if label_count[target_label] < 3:
                    target_label = np.argmax(label_count)
                break

            valid_mask = torch.from_numpy(labels == target_label)
            if valid_mask[pivot_index]:
                new_depths[i] = depth_samples[pivot_index]
            else:
                new_depths[i] = (depth_samples[valid_mask] * weight[valid_mask] / weight[valid_mask].sum()).sum()

            # Clip invalid pixels by clustering
            points = points[valid_mask]
            weight = weight[valid_mask]
            # pivot_index -= (~valid_mask[:pivot_index]).sum()
            assert points.shape[0] >= 3

            # Get rotation by weighted PCA
            weight = weight.expand((weight.shape[0], 3))
            pca = WPCA(n_components=3).fit(points.numpy(), weights=weight.numpy())

            rotation_matrix[i] = torch.from_numpy(pca.components_.copy()).T
            variance[i] = torch.from_numpy(pca.explained_variance_.copy()).clamp_(min=1e-7)
            
        # Flip normals to point towards camera
        ndotv = torch.sum(rotation_matrix[:, :, 2] * (self.camera_center.reshape(1,3) - posWorld), dim=1)
        rotation_matrix[ndotv < 0, :, 2] *= -1

        # Make sure the rotation matrix is valid
        rotation_matrix[torch.linalg.det(rotation_matrix) < 0, :, 1] *= -1
        
        # Convert to  quaternions
        quaternion = torch.tensor(
            Rotation.from_matrix(rotation_matrix).as_quat()[:,[3,0,1,2]], dtype=torch.float32)

        return rotation_matrix, quaternion, variance, new_depths, cov2D

    def compute_scale(self, posView, rotation_matrix, variance, cov2D):
        """
            
        """
        tan_fovx = math.tan(self.FoVx * 0.5)
        tan_fovy = math.tan(self.FoVy * 0.5)
        focal_x = self.W / (2.0 * tan_fovx)
        focal_y = self.H / (2.0 * tan_fovy)

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        
        t = posView.clone()
        txtz = t[:, 0] / t[:, 2]
        tytz = t[:, 1] / t[:, 2]
        t[:, 0] = torch.clamp(txtz, -limx, limx) * t[:, 2]
        t[:, 1] = torch.clamp(tytz, -limy, limy) * t[:, 2]

        # Precompute J
        J = torch.zeros((posView.shape[0], 2, 3), dtype=torch.float32) # [N, 2, 3]
        J[:, 0, 0] = focal_x / t[:, 2]
        J[:, 0, 2] = -(focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
        J[:, 1, 1] = focal_y / t[:, 2]
        J[:, 1, 2] = -(focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])

        h_var = 0.3
        cov2D = cov2D.clone()
        cov2D[:, 0] -= h_var
        cov2D[:, 2] -= h_var
        cov2D = cov2D

        # world to view
        W = self.world2view.T[:3, :3]

        # Initialize rotation and scale
        scale = torch.empty((posView.shape[0], 3), dtype=torch.float32) # [N, 3]

        for i in range(posView.shape[0]):
            R = rotation_matrix[i]
            Q = R.T @ W.T @ J[i].T
            V = variance[i]

            q = torch.tensor([
                V[0] * Q[0,0]**2       + V[1] * Q[1,0]**2       + V[2] * Q[2,0]**2,
                V[0] * Q[0,0] * Q[0,1] + V[1] * Q[1,0] * Q[1,1] + V[2] * Q[2,0] * Q[2,1],
                V[0] * Q[0,0] * Q[0,1] + V[1] * Q[1,0] * Q[1,1] + V[2] * Q[2,0] * Q[2,1],
                V[0] * Q[0,1]**2       + V[1] * Q[1,1]**2       + V[2] * Q[2,1]**2,
            ], dtype=torch.float32)
            
            qtq = (q * q).sum()
            ctq = cov2D[i,0] * q[0] + 2 * cov2D[i,1] * q[1] + cov2D[i,2] * q[3]
            assert qtq > 0.0 and ctq > 0.0

            k = ctq / qtq

            # TODO: Add constraint
            
            scale[i] = torch.sqrt(k * V)

        return scale

    def depth_to_view_world(self, depths):
        mean2D = self.gaussians_2d.get_xy
        posView = depth2view(mean2D, depths, self.znear, self.FoVx, self.FoVy, self.W, self.H)
        view_world_transform = torch.tensor(
            getView2World(self.R, self.camera_center.numpy())
        ).transpose(0, 1)
        posWorld = geom_transform_points(posView, view_world_transform) # [N, 3]
        return posView, posWorld

    def project(self,
                camera_centers, full_proj_transforms, pix_coords, # Shared memory (CPU)
                depth_map, opa_inv_act, scale_inv_act):
        """
            depth_map: [1, H, W]
        """
        # Initialize parameters of 3D gaussians
        rst = {
            "xyz": None,
            "features_dc": None,
            "opacity": None,
            "opa_dir": None,
            "scale": None,
            "rot": None,
        }

        # Parameters of 2D gaussians
        mean2D = self.gaussians_2d.get_xy           # [N, 2]
        # scale2D = self.gaussians_2d.get_scale       # [N, 2]
        # rotation2D = self.gaussians_2d.get_rotation # [N, 1]
        opacity = self.gaussians_2d.get_opacity     # [N, 1]
        color = self.gaussians_2d.get_color         # [N, 3]

        kernel_count = mean2D.shape[0]
        if kernel_count == 0:
            return rst

        # Sample depth map (z coord in view space)
        # Get depth value of gaussian kernels
        grid = mean2D.view(1, kernel_count, 1, 2)
        grid = torch.cat([grid[..., 1:2], grid[..., 0:1]], dim=-1)
        grid[...,0] = (grid[...,0] + 0.5) / self.W * 2 - 1
        grid[...,1] = (grid[...,1] + 0.5) / self.H * 2 - 1
        depth = torch.nn.functional.grid_sample(
                    depth_map.unsqueeze(0), grid,
                    mode='nearest',
                    padding_mode='border',
                    align_corners=False).view(kernel_count)
        # depth -= self.camera_extent * 0.02

        # World position of gaussians
        posView, posWorld = self.depth_to_view_world(depth)

        # SH dc features
        features_dc = RGB2SH(color)
        
        # Compute rotation and coarse scale
        with torch.no_grad():
            rotation_matrix, rotation, variance, depth, cov2D \
                = self.compute_rotation(pix_coords, depth_map, depth, posWorld)
        
        # Recompute world position
        depth[depth < 0.2] = self.zfar
        posView, posWorld = self.depth_to_view_world(depth)

        # Theta values
        with torch.no_grad():
            theta = self.compute_theta(posWorld, camera_centers, full_proj_transforms).unsqueeze(1)

        # Opacity direction
        opa_dir = self.camera_center - posWorld
        opa_dir /= torch.norm(opa_dir, dim=-1, keepdim=True)

        # Finetune scale
        scale = self.compute_scale(posView, rotation_matrix, variance, cov2D)

        rst["xyz"] = posWorld
        rst["features_dc"] = features_dc
        rst["opacity"] = opa_inv_act(opacity)
        rst["opa_dir"] = opa_dir
        rst["theta"] = theta
        rst["scale"] = scale_inv_act(scale)
        rst["rotation"] = rotation

        return rst


