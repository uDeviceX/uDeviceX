#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybop.h"

namespace py = pybind11;

PYBIND11_MODULE(pybop, m) {
    m.doc() = "bop wrapper for python";

    py::class_<PyBop>(m, "PyBop")
        .def(py::init<>())
        .def("alloc",     &PyBop::alloc)
        .def("reset",     &PyBop::reset)
        .def("set_n",     &PyBop::set_n)
        .def("set_vars",  &PyBop::set_vars)
        .def("set_type",  &PyBop::set_type)
        .def("set_data",  &PyBop::set_dataf)
        .def("set_data",  &PyBop::set_datad)
        .def("set_data",  &PyBop::set_datai)
        .def("get_n",     &PyBop::get_n)
        .def("get_vars",  &PyBop::get_vars)
        .def("get_type",  &PyBop::get_type)
        .def("get_int_data", &PyBop::get_datai)
        .def("get_float_data", &PyBop::get_dataf)
        .def("get_double_data", &PyBop::get_datad)
        .def("read",  &PyBop::read)
        .def("write", &PyBop::write);
}
 
