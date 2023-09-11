#include <pybind11/pybind11.h>
#include <fibonacci.h>

unsigned int Fibonacci_cpp(const unsigned int n);

namespace py = pybind11;

PYBIND11_MODULE(fibonacci_example, mod) {
    mod.def("fibonacci_cpp", &Fibonacci_cpp, "Recursive Fibonacci algorithm.");
}
