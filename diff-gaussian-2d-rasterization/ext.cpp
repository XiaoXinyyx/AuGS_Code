#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians_2d", &RasterizeGaussians2DCUDA);
  m.def("rasterize_gaussians_2d_backward", &RasterizeGaussians2DBackwardCUDA);
  m.def("adamUpdate", &adamUpdate);
  m.def("compute_relocation", &ComputeRelocationCUDA);
}