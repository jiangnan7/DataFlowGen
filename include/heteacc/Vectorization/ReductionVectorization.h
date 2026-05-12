#ifndef MLIR_VECTORIZATION_REDUCTION_H
#define MLIR_VECTORIZATION_REDUCTION_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace heteacc {

/// Vectorize conditional and unconditional reduction loops.
/// Transforms unrolled reduction chains (select or addi chains) into
/// vector accumulator + vector.reduction patterns.
void vectorizeConditionalReduction(ModuleOp module, unsigned factor);

} // namespace heteacc
} // namespace mlir

#endif // MLIR_VECTORIZATION_REDUCTION_H
