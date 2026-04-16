#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussians2DCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means2D,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& depths,
    const int image_height,
    const int image_width,
    const bool antialiasing,
    const bool debug
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussians2DBackwardCUDA(
    const int R,
    const torch::Tensor& background,
    const torch::Tensor& means2D,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& depths,
    const torch::Tensor& radii,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& sampleBuffer,
    const int B,
    const torch::Tensor& dL_dout_color,
    const bool antialiasing,
    const bool debug
);

void adamUpdate(
    torch::Tensor &param,
    torch::Tensor &param_grad,
    torch::Tensor &exp_avg,
    torch::Tensor &exp_avg_sq,
    torch::Tensor &visible,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M
);

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
        torch::Tensor& opacity_old,
        torch::Tensor& scale_old,
        torch::Tensor& N,
        torch::Tensor& binoms,
        const int n_max);