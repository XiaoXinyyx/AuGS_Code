#pragma once

#include <cuda_runtime_api.h>
#include "rasterizer.h"

namespace CudaRasterizer
{
    /**
     * @brief 
     * 
     * @tparam T 
     * @param chunk Input: Pointer to the start of the buffer. Output: Offsetted pointer after allocation.
     * @param ptr   Output: Pointer to the (memory aligned) allocated memory.
     * @param count Element count.
     * @param alignment 
     */
    template <typename T>
    static void obtain(char *&chunk, T *&ptr, std::size_t count, std::size_t alignment)
    {
        std::uintptr_t aligned_ptr = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
        ptr = reinterpret_cast<T *>(aligned_ptr);
        chunk = reinterpret_cast<char *>(ptr + count);
    }

    struct GeometryState
    {
        size_t scan_size;     // Size of scanning_space
        char* scanning_space; // Space for temporary storage.
        float4* conic_opacity;
        uint32_t* point_offsets;
        uint32_t* tiles_touched;

        // Construct a new GeometryState object from a chunk of memory.
        static GeometryState fromChunk(char *&chunk, size_t count);
    };

    struct ImageState
    {
        uint32_t* bucket_count;
        uint32_t* bucket_offsets; // Inclusive sum of bucket count per tile
        size_t bucket_count_scan_size;
        char* bucket_count_scanning_space;
        float* pixel_colors;
        uint32_t* max_contrib;    // max contributor per tile. (start from 1)
        
        uint2* ranges;
        uint32_t* n_contrib;
        float* accum_alpha;

        static ImageState fromChunk(char *&chunk, size_t pix_count, size_t tile_count);
        static size_t required(size_t pix_count, size_t tile_count);
    };

    struct BinningState
    {
        size_t sorting_size;      // Size of list_sorting_space
        char* list_sorting_space; // Temporary space for sorting.
        uint64_t* point_list_keys_unsorted;
        uint64_t* point_list_keys;
        uint32_t* point_list_unsorted;
        uint32_t* point_list;

        static BinningState fromChunk(char *&chunk, size_t count);
    };

    struct SampleState
    {
        uint32_t* bucket_to_tile;
        float* T;
        float* ar; // Sampled color

        static SampleState fromChunk(char*& chunk, size_t C);
    };

    // Calculate the required memory size in bytes.
    template <typename T>
    size_t required(size_t P)
    {
        char *size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }
}