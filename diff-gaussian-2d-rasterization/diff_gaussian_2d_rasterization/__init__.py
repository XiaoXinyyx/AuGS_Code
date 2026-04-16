from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class Gaussian2DRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    background: torch.Tensor
    antialiasing: bool
    debug: bool

class Gaussian2DRasterizer(nn.Module):

    def __init__(self, raster_settings):
        super().__init__()
        if raster_settings is not None:
            self.set_raster_settings(raster_settings)
    
    def set_raster_settings(self, raster_settings: Gaussian2DRasterizationSettings):
        old_settings = getattr(self, "raster_settings", None)
        self.raster_settings = raster_settings

        # Update coordinates if needed
        if old_settings is None \
            or old_settings.image_height != raster_settings.image_height \
            or old_settings.image_width != raster_settings.image_width:

            rows, cols = torch.meshgrid(
                    torch.arange(raster_settings.image_height),
                    torch.arange(raster_settings.image_width), indexing='ij')
            self.coordinates = torch.stack([rows, cols], dim=-1).float().cuda()

    def forward(self, means2D, colors, opacities, scales, rotations, depths: torch.Tensor):
        raster_settings = self.raster_settings
        return _RasterizeGaussians2D.apply(means2D, colors, opacities, scales, rotations, depths, raster_settings)


class _RasterizeGaussians2D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2D,
        colors,
        opacities,
        scales,
        rotations,
        depths,
        raster_settings: Gaussian2DRasterizationSettings,
    ):
        args = (
            raster_settings.background,
            means2D,
            colors,
            opacities,
            scales,
            rotations,
            depths,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.antialiasing,
            raster_settings.debug,
        )
        
        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, radii, pix_covered, geomBuffer, binningBuffer, imgBuffer, \
                sampleBuffer \
                    = _C.rasterize_gaussians_2d(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Writing snapshot_fw.dump for debugging.\n")
                raise ex
        else:
            num_rendered, num_buckets, color, radii, pix_covered, geomBuffer, binningBuffer, imgBuffer, \
            sampleBuffer = _C.rasterize_gaussians_2d(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(means2D, colors, opacities, scales, rotations, depths,
                              radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)

        return (color, radii, pix_covered)
    
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_pix_covered):
        
        # Restore necessary values from context
        raster_settings = ctx.raster_settings
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        means2D, colors, opacities, scales, rotations, depths, \
        radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer  \
            = ctx.saved_tensors

        args = (
            num_rendered,
            raster_settings.background,
            means2D,
            colors,
            opacities,
            scales,
            rotations,
            depths,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sampleBuffer,
            num_buckets,
            grad_out_color,
            raster_settings.antialiasing,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors, grad_opacities, grad_scales, grad_rotations \
                    = _C.rasterize_gaussians_2d_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors, grad_opacities, grad_scales, grad_rotations \
                = _C.rasterize_gaussians_2d_backward(*args)

        return (
            grad_means2D,
            grad_colors,
            grad_opacities,
            grad_scales,
            grad_rotations,
            None,
            None,
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)


def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 