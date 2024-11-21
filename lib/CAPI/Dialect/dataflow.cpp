//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The heteacc Authors.
//
//===----------------------------------------------------------------------===//

#include "heteacc-c/Dialect/DataFlow.h"
#include "mlir/CAPI/Registration.h"
#include "heteacc/Dialect/DataFlow/DataFlow.h"

using namespace mlir;
using namespace heteacc;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DataFlow, dataflow, mlir::heteacc::dataflow::DataFlowDialect)
