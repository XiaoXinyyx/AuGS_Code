#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

namespace CudaRasterizer
{
    class Rasterizer
    {
    public:
        static std::tuple<int,int> forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            std::function<char* (size_t)> sampleBuffer,
            const int P, const int W, const int H,
            const float* background,
            const float* means2D,
            const float* colors,
            const float* opacities,
            const float* scales,
            const float* rotations,
            const float* depths,
            float* out_color,
            int* radii,
            int* pix_covered,
            const bool antialiasing,
            const bool debug);

        static void backward(
            const int P, const int W, const int H, const int R, const int B,
            const float* background,
            const float* means2D,
            const float* colors,
            const float* opacities,
            const float* scales,
            const float* rotations,
            const int* radii,
            char* geom_buffer,
            char* binning_buffer,
            char* img_buffer,
            char* sample_buffer,
            const float* dL_dpix,
            float* dL_dmean2D,
            float* dL_dcolor,
            float* dL_dopacity,
            float* dL_dscales,
            float* dL_drotations,
            float* dL_dconic,
            const bool antialiasing,
            const bool debug);
    };
}



#endif 