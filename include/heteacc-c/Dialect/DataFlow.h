#ifndef HETEACC_C_DIALECT_HETEACC_H
#define HETEACC_C_DIALECT_HETEACC_H

#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"


#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DataFlow, dataflow);

#ifdef __cplusplus
}
#endif

#endif