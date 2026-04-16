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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    // Main rasterization method
    void render(
        const dim3 grid, const dim3 block,
        const int W, const int H,
        const uint2 *ranges,
        const uint32_t *point_list,
        const uint32_t* bucket_offsets, // bucket offsets per tile
        uint32_t* bucket_to_tile,
        float* sampled_T,
        float* sampled_ar,
        const float2 *means2D,
        const float4 *conic_opacity,
        const float *colors,
        const float *bg_color,
        float *final_T,
        uint32_t *n_contrib,
        uint32_t* max_contrib,
        int32_t* pix_covered,
        float *out_color);

    void preprocess(
        const int P, const int W, const int H,
        const float* means2D,
        const glm::vec2* scales,
        const float* rotations,
        const float* opacities,
        const float* depths,
        const dim3 tile_grid,
        int* radii,
        float4* conic_opacity,
        uint32_t* tiles_touched,
        const bool antialiasing);
}

#endif