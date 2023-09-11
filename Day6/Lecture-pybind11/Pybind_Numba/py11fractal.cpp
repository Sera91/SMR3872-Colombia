#include <complex>
#include <vector>
#include <pybind11/numpy.h>
 
py::array_t<int> quick(int height, int width, int maxiterations) {
 
     py::array_t<int> fractal({height, width});
 
     auto fractal_uc = fractal.mutable_unchecked<2>();
 
     for (int h = 0;  h < height;  h++) {
         for (int w = 0;  w < width;  w++) {
 
             std::complex<double> ci{
                 double(h-1)/height - 1,
                 1.5 * (double(w-1)/width - 1)};
 
             std::complex<double> z = ci;
             fractal_uc(h,w) = maxiterations;
             for (int i = 0;  i < maxiterations;  i++) {
                 z = z * z + ci;
                 if (std::abs(z) > 2) {
                     fractal_uc(h, w) = i;
                     break;
                 }
             }
         }
     }
 
     return fractal;
}
 
PYBIND11_MODULE(py11fractal, m) {
     m.def("quick", quick);
}
