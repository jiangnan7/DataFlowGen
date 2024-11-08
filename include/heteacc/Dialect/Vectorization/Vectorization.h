
#ifndef DIALECT_VECTORIZATION_IR_TD
#define DIALECT_VECTORIZATION_IR_TD

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "heteacc/Dialect/Vectorization/VectorizationOpsDialect.h.inc"

// namespace heteacc {
// namespace mlir{
// namespace Vectorization{

//   class VectorizationDialect;
// } // end namespace Vectorization
// // } // end namespace heteacc

// }

#define GET_OP_CLASSES
#include "heteacc/Dialect/Vectorization/VectorizationOps.h.inc"

#endif // DIALECT_VECTORIZATION_IR_TD
