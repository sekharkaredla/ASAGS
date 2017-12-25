/*

*/
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) {
    return x * x;
}

PYBIND11_PLUGIN(temp) {
    pybind11::module m("temp", "auto-compiled c++ extension");
    m.def("square", &square);
    return m.ptr();
}
