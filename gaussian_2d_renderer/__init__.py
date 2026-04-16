import math

import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from utils.general_utils import safe_inverse_sigmoid
from diff_gaussian_2d_rasterization import Gaussian2DRasterizer, Gaussian2DRasterizationSettings

from .projector import GaussianProjector

class Gaussian2DModel:

    def setup_functions(self):
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = safe_inverse_sigmoid

        self.scale_activation = torch.exp
        self.inverse_scale_activation = torch.log

    def __init__(self, depth_map = None):
        self._xy = nn.Parameter(torch.zeros((0, 2), dtype=torch.float, device="cuda").requires_grad_(True))
        self._color = nn.Parameter(torch.zeros((0, 3), dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.zeros((0, 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._scale = nn.Parameter(torch.zeros((0, 2), dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.zeros((0, 1), dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._depth = torch.zeros((0, 1), dtype=torch.float, device="cuda")
        self.depth_map = depth_map

        self.setup_functions()

    def capture(self):
        return (
            self._xy,
            self._color,
            self._opacity,
            self._scale,
            self._rotation
            # TODO : add depth
        )

    def restore(self, model_args):
        (   
            self._xy,
            self._color,
            self._opacity,
            self._scale,
            self._rotation
        ) = model_args
        return self

    def cpu_(self):
        self._xy = self._xy.cpu()
        self._color = self._color.cpu()
        self._opacity = self._opacity.cpu()
        self._scale = self._scale.cpu()
        self._rotation = self._rotation.cpu()
        self._depth = self._depth.cpu()
        return self

    def training_setup(self, training_args, image_scale):

        l = [
            {'params': [self._xy], 'lr': training_args.xy_2d_lr * image_scale, 'name': 'xy'},
            {'params': [self._color], 'lr': training_args.color_2d_lr, 'name': 'color'},
            {'params': [self._opacity], 'lr': training_args.opacity_2d_lr, 'name': 'opacity'},
            {'params': [self._scale], 'lr': training_args.scale_2d_lr, 'name': 'scale'},
            {'params': [self._rotation], 'lr': training_args.rotation_2d_lr, 'name': 'rotation'},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            
            optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, valid_points_mask):
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xy = optimizable_tensors["xy"]
        self._color = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scale = optimizable_tensors["scale"]
        self._rotation = optimizable_tensors["rotation"]

    def add_gaussian_kernels(self, new_xy, new_color, new_opacity, new_scale, new_rotation):
        with torch.no_grad():
            self._depth = torch.cat([self._depth, self.sample_depth_map(new_xy)], dim=0)

        tensors_dict = {
            "xy": new_xy,
            "color": new_color,
            "opacity": new_opacity,
            "scale": new_scale,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)

        self._xy = optimizable_tensors["xy"]
        self._color = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scale = optimizable_tensors["scale"]
        self._rotation = optimizable_tensors["rotation"]

    # return: [N, 1]
    def sample_depth_map(self, xy: torch.Tensor):
        _, H, W = self.depth_map.shape
        grid = xy.view(1, xy.shape[0], 1, 2)
        grid = torch.cat([grid[..., 1:2], grid[..., 0:1]], dim=-1)
        grid[...,0] = (grid[...,0] + 0.5) / W * 2 - 1
        grid[...,1] = (grid[...,1] + 0.5) / H * 2 - 1
        return grid_sample(
            self.depth_map.unsqueeze(0), grid, align_corners=False).view(xy.shape[0], 1)

    def end_optimizer_step(self):
        with torch.no_grad():
            self._color.data.clamp_(min=0.0, max=1.0)

            self._depth = self.sample_depth_map(self._xy)

            _, H, W = self.depth_map.shape
            self._xy[:, 0].clamp_(min=0.0, max=H-1)
            self._xy[:, 1].clamp_(min=0.0, max=W-1)

            # Clamp max scale to reduce time comsumption for clustering
            self._scale.clamp_(max=math.log(min(50, (H+W) * 0.01)))

    @property
    def get_xy(self):
        return self._xy

    @property
    def get_color(self):
        return self._color

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scale(self):
        return self.scale_activation(self._scale)

    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_depth(self):
        return self._depth

    def size(self):
        return self._xy.shape[0]

rasterizer = Gaussian2DRasterizer(None)

def render_2d(background, pc : Gaussian2DModel):

    raster_settings = Gaussian2DRasterizationSettings(
        image_height = background.shape[1],
        image_width = background.shape[2],
        background = background,
        antialiasing = True,
        debug = False,
    )

    rasterizer.set_raster_settings(raster_settings)

    means2D = pc.get_xy
    colors = pc.get_color
    opacities = pc.get_opacity
    scales = pc.get_scale
    rotations = pc.get_rotation
    depths = pc.get_depth

    rst = rasterizer(
        means2D = means2D,
        colors = colors,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        depths = depths,
    )
    
    return {"render": rst[0], "radii": rst[1], "pix_covered": rst[2]}