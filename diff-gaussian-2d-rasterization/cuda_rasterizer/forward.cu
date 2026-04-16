#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "forward.h"
#include "auxiliary.h"

namespace cg = cooperative_groups;

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

// Perform initial steps for each Gaussian prior to rasterization.
template<uint32_t CHANNELS>
__global__ void preprocessCUDA(
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
    const bool antialiasing)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    float2 mean2D = {means2D[idx * 2], means2D[idx * 2 + 1]};
    glm::vec2 scale = scales[idx];
    float rotation = rotations[idx];
    float opacity = opacities[idx];

    // Compute 2D screen-space covariance matrix
    float3 cov2D = computeCov2D(scale, rotation);

    // Use helper variables for 2D covariance entries. More compact.
    float c_xx = cov2D.x;
    float c_xy = cov2D.y;
    float c_yy = cov2D.z;

    constexpr float h_var = 0.3f;
	const float det_cov = c_xx * c_yy - c_xy * c_xy;
	c_xx += h_var;
	c_yy += h_var;
	const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
    // max for numerical stability
	float h_convolution_scaling = antialiasing ? 
        sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)) : 1.0f;

    // Invert covariance (EWA algorithm)
    const float det = det_cov_plus_h_cov;

    // Covariance matrix is positive semi-definite.
    // When h_var is added to the diagonal, the determinant is positive.
    if (det <= 0.0f) 
        return;

    float det_inv = 1.f / det;
    float3 conic = {c_yy * det_inv, -c_xy * det_inv, c_xx * det_inv};

    // Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles
    float mid = 0.5f * (c_xx + c_yy);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = {mean2D.y, mean2D.x};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, tile_grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
    
    // Store some useful helper data for the next steps.
    radii[idx] = int(my_radius);

    // Inverse 2D covariance and opacity neatly pack into one float4
    conic_opacity[idx] = {conic.x, conic.y, conic.z, opacity * h_convolution_scaling};

    tiles_touched[idx] = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const int W, const int H,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const uint32_t* __restrict__ bucket_offsets, // bucket offsets per tile
    uint32_t* __restrict__ bucket_to_tile,
    float* __restrict__ sampled_T,
    float* __restrict__ sampled_ar,
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const float* __restrict__ bg_color,
    float* __restrict__ final_T,
    uint32_t* __restrict__ n_contrib,
    uint32_t* __restrict__ max_contrib,
    int32_t* __restrict__ pix_covered,
    float* __restrict__ out_color)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;

    // Load start/end range of IDs to process in bit sorted list.
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 range = ranges[tile_id];
    int toDo = inside ? range.y - range.x : 0;

    // what is the number of buckets before me? what is my offset?
    uint32_t bbm = tile_id == 0 ? 0 : bucket_offsets[tile_id - 1];
    int num_buckets = (range.y - range.x + 31) / 32;
    for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
        if (bucket_idx < num_buckets) {
            bucket_to_tile[bbm + bucket_idx] = tile_id;    
        }
    }

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float C[CHANNELS] = {0.0f};
    
    for(int i = 0; i < toDo; ++i)
    {
        // add incoming T value for every 32nd gaussian
        if (i % 32 == 0) {
            sampled_T[(bbm * BLOCK_SIZE) + block.thread_rank()] = T;
            for (int ch = 0; ch < CHANNELS; ++ch)
                sampled_ar[(bbm * BLOCK_SIZE * CHANNELS) + ch * BLOCK_SIZE + block.thread_rank()]
                    = C[ch];
            ++bbm;
        }

        int idx = point_list[range.x + i];
        float2 xy = {means2D[idx].y, means2D[idx].x};
        float4 con_o = conic_opacity[idx];
        
        // Keep track of current position in range
		contributor++;

        // Resample using conic matrix (cf. "Surface 
        // Splatting" by Zwicker et al., 2001)
        float2 d = {xy.x - pixf.x, xy.y - pixf.y};
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y)
                      - con_o.y * d.x * d.y;
        if (power > 0.0f) continue;
        
        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity
        // and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix). 
        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f) continue;

        // Eq. (3) from 3D Gaussian splatting paper.
        for(int ch = 0; ch < CHANNELS; ++ch)
            C[ch] += colors[idx * CHANNELS + ch] * alpha * T;
        
        // Keep track of last range entry to update this pixel.
        last_contributor = contributor;

        atomicAdd(&pix_covered[idx], 1);

        T = T * (1 - alpha);
        if (T < 0.0001f)
            break;
    }

    // All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
    if (inside)
    {
        final_T[pix_id] = T;
        n_contrib[pix_id] = last_contributor;
        for(int ch = 0; ch < CHANNELS; ++ch)
            out_color[ch * H * W + pix_id] = C[ch] + bg_color[ch * H * W + pix_id] * T;
    }

    // max reduce the last contributor
    typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    last_contributor = BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
    if (block.thread_rank() == 0) {
		max_contrib[tile_id] = last_contributor;
	}
}

void FORWARD::preprocess(
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
    const bool antialiasing)
{
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P, W, H,
        means2D,
        scales,
        rotations,
        opacities,
        depths,
        tile_grid,
        radii,
        conic_opacity,
        tiles_touched,
        antialiasing);
}


void FORWARD::render(
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
    float *out_color)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        W, H,
        ranges,
        point_list,
        bucket_offsets,
        bucket_to_tile,
        sampled_T,
        sampled_ar,
        means2D,
        conic_opacity,
        colors,
        bg_color,
        final_T,
        n_contrib,
        max_contrib,
        pix_covered,
        out_color);
}