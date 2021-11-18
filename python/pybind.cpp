
// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).
// All rights reserved.

// Copyright (C) 2021 Jiarui Fang (fangjiarui123@gmail.com).  All rights reserved.

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pybind11/pybind11.h"
#include "core/tensor.h"

namespace ps_tensor {
namespace python {

namespace py = pybind11;

static void DLPack_Capsule_Destructor(PyObject *data) {
  auto *dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  if (dlMTensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlMTensor->deleter(const_cast<DLManagedTensor *>(dlMTensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}
PYBIND11_MODULE(ps_tensor_cxx, m) {
py::class_<core::Tensor>(m, "Tensor")
      .def_static("from_dlpack",
                  [](py::capsule capsule) -> std::unique_ptr<core::Tensor> {
                    auto tensor = (DLManagedTensor *)(capsule);
                    PyCapsule_SetName(capsule.ptr(), "used_tensor");
                    return std::make_unique<core::Tensor>(tensor);
                  })
      .def("to_dlpack",
           [](core::Tensor &tensor) -> py::capsule {
             auto *dlpack = tensor.ToDLPack();
             return py::capsule(dlpack, "dltensor", DLPack_Capsule_Destructor);
           })
      .def("n_dim", &core::Tensor::n_dim)
      .def("shape", &core::Tensor::shape)
      .def("move_gpu", &core::Tensor::move_gpu)
      .def("float_data", &core::Tensor::data<float>)
      .def("print", &core::Tensor::print_data)
      .def_static("create_empty", [] { return core::Tensor(nullptr); });
}
}  // namespace python
}  // namespace ps_tensor
