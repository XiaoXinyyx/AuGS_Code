/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>

#include "cuda_rasterizer/config.h"
#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include "cuda_rasterizer/utils.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
    const bool debug)
{
    const int P = means2D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts = means2D.options().dtype(torch::kInt32);
    auto float_opts = means2D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full(
        {NUM_CHANNELS, image_height, image_width}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, int_opts);
    torch::Tensor pix_covered = torch::full({P}, 0, int_opts);
    
    // Create useful buffers for the rasterizer
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
    std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

    int rendered = 0;
    int num_buckets = 0;
    if (P != 0) 
    {
        auto tup = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            sampleFunc,
            P, W, H,
            background.contiguous().data<float>(),
            means2D.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            opacities.contiguous().data<float>(),
            scales.contiguous().data<float>(),
            rotations.contiguous().data<float>(),
            depths.contiguous().data<float>(),
            out_color.contiguous().data<float>(),
            radii.contiguous().data<int32_t>(),
            pix_covered.contiguous().data<int32_t>(),
            antialiasing,
            debug);
        
        rendered = std::get<0>(tup);
        num_buckets = std::get<1>(tup);
    } 
    else 
    {
        out_color = background.clone();
    }
    return std::make_tuple(rendered, num_buckets, out_color, radii, pix_covered, geomBuffer, binningBuffer, imgBuffer, sampleBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussians2DBackwardCUDA(
    const int R, // num rendered
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
    const bool debug)
{
    const int P = means2D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    torch::Tensor dL_dmeans2D = torch::zeros({P, 2}, means2D.options()).contiguous();
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means2D.options()).contiguous();
    torch::Tensor dL_dopacities = torch::zeros({P, 1}, means2D.options()).contiguous();
    torch::Tensor dL_dscales = torch::zeros({P, 2}, means2D.options()).contiguous();
    torch::Tensor dL_drotations = torch::zeros({P, 1}, means2D.options()).contiguous();
    torch::Tensor dL_dconic = torch::zeros({P, 3}, means2D.options()).contiguous(); // 2D inverse covariance matrix

    if (P != 0)
    {
        CudaRasterizer::Rasterizer::backward(
            P, W, H, R, B,
            background.contiguous().data<float>(),
            means2D.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            opacities.contiguous().data<float>(),
            scales.contiguous().data<float>(),
            rotations.contiguous().data<float>(),
            radii.contiguous().data<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data<float>(),
            dL_dmeans2D.data<float>(),
            dL_dcolors.data<float>(),
            dL_dopacities.data<float>(),
            dL_dscales.data<float>(),
            dL_drotations.data<float>(),
            dL_dconic.data<float>(),
            antialiasing,
            debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacities, dL_dscales, dL_drotations);
}


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
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);
}