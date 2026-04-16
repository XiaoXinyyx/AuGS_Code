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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ __forceinline__ float3 computeCov2D(const glm::vec2 scale, const float rotation)
{
    float cos_theta, sin_theta;
    sincos(rotation, &sin_theta, &cos_theta);
    const float a2 = scale.x * scale.x;
    const float b2 = scale.y * scale.y;

    return {a2 * cos_theta * cos_theta + b2 * sin_theta * sin_theta,
            (a2 - b2) * sin_theta * cos_theta,
            a2 * sin_theta * sin_theta + b2 * cos_theta * cos_theta};
}

template <uint32_t C>
__global__ void PerGaussianRenderCUDA(
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int W, const int H, const int B,
    const float* __restrict__ sampled_T,
    const float* __restrict__ sampled_ar,
    const float* __restrict__ final_Ts,
    const uint32_t* __restrict__ n_contrib,
    const uint32_t* __restrict__ per_tile_bucket_offset,
    const uint32_t* __restrict__ max_contrib,
    const float* __restrict__ pixel_colors,
    const uint32_t* __restrict__ bucket_to_tile,
    const float* __restrict__ dL_dpixels,
    float2* __restrict__ dL_dmean2D,
    float3* __restrict__ dL_dconic2D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors)
{
    // get the bucket index
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> my_warp = cg::tiled_partition<32>(block);
    uint32_t global_bucket_idx = block.group_index().x;
    bool valid_bucket = global_bucket_idx < (uint32_t) B;
    if (!valid_bucket) return;
    
    const uint32_t tile_id = bucket_to_tile[global_bucket_idx];
    const uint2 range = ranges[tile_id];
    const int num_splats_in_tile = range.y - range.x;
    // What is the number of buckets before me? what is my offset?
    const uint32_t bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
    const int bucket_idx_in_tile = global_bucket_idx - bbm;
    const int splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
    const int splat_idx_global = range.x + splat_idx_in_tile;
    const bool valid_splat = (splat_idx_in_tile < num_splats_in_tile);

    // if first gaussian in bucket is useless, then others are also useless
    if (bucket_idx_in_tile * 32 >= max_contrib[tile_id])
        return;

    // Load Gaussian properties into registers
    int gaussian_idx = 0;
    float2 xy = {0.0f, 0.0f};
    float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
    float c[C] = {0.0f}; // color of the gaussian
    if (valid_splat) {
        gaussian_idx = point_list[splat_idx_global];
        xy = {means2D[gaussian_idx].y, means2D[gaussian_idx].x};
        con_o = conic_opacity[gaussian_idx];
        for (int ch = 0; ch < C; ++ch)
            c[ch] = colors[gaussian_idx * C + ch];
    }

    // Gradient accumulation variables
    float Register_dL_dmean2D_x = 0.0f;
    float Register_dL_dmean2D_y = 0.0f;
    float Register_dL_dconic2D_x = 0.0f;
    float Register_dL_dconic2D_y = 0.0f;
    float Register_dL_dconic2D_z = 0.0f;
    float Register_dL_dopacity = 0.0f;
    float Register_dL_dcolors[C] = {0.0f};

    // tile metadata
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
    const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

    // values useful for gradient calculation
	float T;
    float T_final;
    float last_contributor; // starts from 1
    float ar[C];
    float dL_dpixel[C];

    // Iterate over all pixels in the tile
    for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
        // At this point, T already has my (1 - alpha) multiplied.
		// So pass this ready-made T value to next thread.
        T = my_warp.shfl_up(T, 1);
        last_contributor = my_warp.shfl_up(last_contributor, 1);
        T_final = my_warp.shfl_up(T_final, 1);
        for (int ch = 0; ch < C; ++ch) {
            ar[ch] = my_warp.shfl_up(ar[ch], 1);
            dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
        }

        // which pixel index should this thread deal with?
        int idx = i - my_warp.thread_rank();
        const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
        const uint32_t pix_id = W * pix.y + pix.x;
        const float2 pixf = {(float)pix.x, (float)pix.y};
        const bool valid_pixel = pix.x < W && pix.y < H;

        // every 32nd thread should read the stored state from memory
        if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE) {
            T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
            for (int ch = 0; ch < C; ++ch)
                ar[ch] = -pixel_colors[ch * H * W + pix_id] 
                       + sampled_ar[global_bucket_idx * BLOCK_SIZE * C + ch * BLOCK_SIZE + idx];
            T_final = final_Ts[pix_id];
            last_contributor = n_contrib[pix_id];
            for (int ch = 0; ch < C; ++ch)
                dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
        }

        // do work
        if (valid_splat && valid_pixel && idx >= 0 && idx < BLOCK_SIZE)
        {
            if (splat_idx_in_tile >= last_contributor) continue;

            // compute blending values
            const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) 
                                - con_o.y * d.x * d.y;
            if (power > 0.0f) continue;
            const float G = exp(power);
            float alpha = min(0.99f, con_o.w * G);
            if (alpha < 1.0f / 255.0f) continue;
            const float weight = alpha * T;

            // add the gradient contribution of this pixel's color to the gaussian
            float dL_dalpha = 0.0f;
            for (int ch = 0; ch < C; ++ch) {
                ar[ch] += weight * c[ch];
                const float &dL_dchannel = dL_dpixel[ch];
                Register_dL_dcolors[ch] += weight * dL_dchannel;
                dL_dalpha += ((c[ch] * T) + ar[ch] / (1.0f - alpha)) * dL_dchannel;
            }

			T *= (1.0f - alpha);

            // Helpful reusable temporary variables
            const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            // accumulate the gradients
            Register_dL_dmean2D_y += dL_dG * dG_ddelx;
            Register_dL_dmean2D_x += dL_dG * dG_ddely;

            Register_dL_dconic2D_x += -0.5f * gdx * d.x * dL_dG;
			Register_dL_dconic2D_y += -0.5f * gdx * d.y * dL_dG;
			Register_dL_dconic2D_z += -0.5f * gdy * d.y * dL_dG;
            Register_dL_dopacity += G * dL_dalpha;
        }
    }

    // finally add the gradients using atomics
    if (valid_splat) {
        atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
        atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
        atomicAdd(&dL_dconic2D[gaussian_idx].x, Register_dL_dconic2D_x);
        atomicAdd(&dL_dconic2D[gaussian_idx].y, Register_dL_dconic2D_y);
        atomicAdd(&dL_dconic2D[gaussian_idx].z, Register_dL_dconic2D_z);
        atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
        for (int ch = 0; ch < C; ++ch)
            atomicAdd(&dL_dcolors[gaussian_idx * C + ch], Register_dL_dcolors[ch]);
    }
}

template<int C>
__global__ void preprocessCUDA(
    const int P,
    const int* radii,
    const float* opacities,
    const glm::vec2* scales,
    const float* rotations,
    const float* dL_dconics, // gradients of inverse 2D covariance matrices
    float* dL_dopacity,
    float* dL_dscales,
    float* dL_drotations,
    const bool antialiasing)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0)) return;

    glm::vec2 scale = scales[idx];
    float rotation = rotations[idx];
    float3 dL_dconic = {dL_dconics[3 * idx], dL_dconics[3 * idx + 1], dL_dconics[3 * idx + 2]};

    // Compute 2D screen-space covariance matrix
    float3 cov2D = computeCov2D(scale, rotation);

    // Use helper variables for 2D covariance entries. More compact.
    float c_xx = cov2D.x;
	float c_xy = cov2D.y;
	float c_yy = cov2D.z;

    constexpr float h_var = 0.3f;
    float d_inside_root = 0.0f;
    if (antialiasing)
    {
        const float det_cov = c_xx * c_yy - c_xy * c_xy;
        c_xx += h_var;
        c_yy += h_var;
        const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
        const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));
		const float d_h_convolution_scaling = dL_dopacity[idx] * opacities[idx];
		dL_dopacity[idx] *= h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 
                        0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
    } 
    else
    {
        c_xx += h_var;
        c_yy += h_var;
    }
    
    float dL_dc_xx = 0;
    float dL_dc_xy = 0;
    float dL_dc_yy = 0;
    if (antialiasing)
    {
        // https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
    }

    const float denom = c_xx * c_yy - c_xy * c_xy;
	const float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
    if (denom2inv != 0)
    {
        // Gradients of loss w.r.t. entries of 2D covariance matrix,
        // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
        // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
        dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x 
                                + 2 * c_xy * c_yy * dL_dconic.y 
                                + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z 
                                + 2 * c_xx * c_xy * dL_dconic.y 
                                + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x 
                                    - (denom + 2 * c_xy * c_xy) * dL_dconic.y 
                                    + c_xx * c_xy * dL_dconic.z);
        // Gradients of loss w.r.t. scale and rotation
        const float a = scale.x;
        const float b = scale.y;
        const float a2 = sq(a);
        const float b2 = sq(b);
        float cos_theta, sin_theta;
        sincos(rotation, &sin_theta, &cos_theta);
        dL_drotations[idx] = (b2 - a2) * (dL_dc_xx * 2 * sin_theta * cos_theta
                                        + dL_dc_xy * (sin_theta * sin_theta - cos_theta * cos_theta)
                                        - dL_dc_yy * 2 * sin_theta * cos_theta);
        dL_dscales[idx * 2] = 2 * (dL_dc_xx * a * cos_theta * cos_theta
                                + dL_dc_xy * a * sin_theta * cos_theta
                                + dL_dc_yy * a * sin_theta * sin_theta);
        dL_dscales[idx * 2 + 1] = 2 * (dL_dc_xx * b * sin_theta * sin_theta
                                    - dL_dc_xy * b * sin_theta * cos_theta
                                    + dL_dc_yy * b * cos_theta * cos_theta); 
    }
}

void BACKWARD::render(
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
    float* dL_dcolors)
{
    PerGaussianRenderCUDA<NUM_CHANNELS> <<<B, 32>>> (
        means2D,
        conic_opacity,
        colors,
        ranges,
        point_list,
        W, H, B,
        sampled_T,
        sampled_ar,
        final_Ts,
        n_contrib,
        per_tile_bucket_offset,
        max_contrib,
        pixel_colors,
        bucket_to_tile,
        dL_dpixels,
        dL_dmean2D,
        dL_dconic2D,
        dL_dopacity,
        dL_dcolors);
}

void BACKWARD::preprocess(
    const int P,
    const int* radii,
    const float* opacities,
    const glm::vec2* scales,
    const float* rotations,
    const float* dL_dconics, // gradients of inverse 2D covariance matrices
    float* dL_dopacity,
    float* dL_dscales,
    float* dL_drotations,
    const bool antialiasing)
{
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P,
        radii,
        opacities,
        scales,
        rotations,
        dL_dconics, // gradients of 2D covariance matrices
        dL_dopacity,
        dL_dscales,
        dL_drotations,
        antialiasing);
}