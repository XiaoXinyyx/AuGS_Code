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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        const float2* means2D,
        const float4* conic_opacity,
        const float* colors,
        const uint2* ranges,
        const uint32_t* point_list,
        const int W, const int H, const int B,
        const float* sampled_T,
        const float* sampled_ar,
        const float* final_Ts,
        const uint32_t* n_contrib,
        const uint32_t* per_tile_bucket_offset,
        const uint32_t* max_contrib,
        const float* pixel_colors,
        const uint32_t* bucket_to_tile,
        const float* dL_dpixels,
        float2* dL_dmean2D,
        float3* dL_dconic2D,
        float* dL_dopacity,
        float* dL_dcolors);

    void preprocess(
        const int P,
        const int* radii,
        const float* opacities,
        const glm::vec2* scales,
        const float* rotations,
        const float* dL_dconics, // gradients of inverse 2D covariance matrices
        float* dL_dopacity,
        float* dL_dscales,
        float* dL_drotations,
        const bool antialiasing);
}
#endif