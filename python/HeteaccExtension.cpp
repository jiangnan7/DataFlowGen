//===- dataflowExtension.cpp - Extension module -------------------------------===//

#include "heteacc-c/Dialect/DataFlow.h"
#include "PybindAdaptors.h"
#include <pybind11/pybind11.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/../../lib/Bindings/Python/IRModule.h"
#include "mlir/CAPI/IR.h"
#include "llvm-c/ErrorHandling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Signals.h"

#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;


//===----------------------------------------------------------------------===//
// PybindUtils
//===----------------------------------------------------------------------===//

pybind11::error_already_set
mlir::python::SetPyError(PyObject *excClass, const llvm::Twine &message) {
  auto messageStr = message.str();
  PyErr_SetString(excClass, messageStr.c_str());
  return pybind11::error_already_set();
}


//===----------------------------------------------------------------------===//
// Numpy array retrieval utils
//===----------------------------------------------------------------------===//

static void getVectorFromUnsignedNpArray(PyObject *object,
                                         SmallVectorImpl<unsigned> &vector) {
  _import_array();
  if (!PyArray_Check(object))
    throw SetPyError(PyExc_ValueError, "expect numpy array");
  auto array = reinterpret_cast<PyArrayObject *>(object);
  if (PyArray_TYPE(array) != NPY_INT64 || PyArray_NDIM(array) != 1)
    throw SetPyError(PyExc_ValueError, "expect single-dimensional int64 array");

  auto dataBegin = reinterpret_cast<int64_t *>(PyArray_DATA(array));
  auto dataEnd = dataBegin + PyArray_DIM(array, 0);

  vector.clear();
  for (auto i = dataBegin; i != dataEnd; ++i) {
    auto value = *i;
    if (value < 0)
      throw SetPyError(PyExc_ValueError, "expect non-negative array element");
    vector.push_back((unsigned)value);
  }
}

PYBIND11_MODULE(_heteacc, m) {
   m.doc() = "Heteacc Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();
  auto dataflow_m = m.def_submodule("dataflow");

  dataflow_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__dataflow__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}