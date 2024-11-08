#ifndef DATAFLOW_UTILS_H
#define DATAFLOW_UTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "heteacc/Dialect/DataFlow/DataFlow.h"

namespace mlir {
    // 
namespace heteacc {

using namespace dataflow;

//===----------------------------------------------------------------------===//
// Dataflow utils
//===----------------------------------------------------------------------===//

void applyBranchPrediction(dataflow::ExecutionBlockOp exeop);

/// Wrap the operations in the block with dispatch op.
LaunchOp launchBlock(Block *block);

TaskOp fuseOpsIntoTask(ArrayRef<Operation *> ops,
                                 PatternRewriter &rewriter,
                                 bool insertToLastOp = false);
/// Wrap the operations in the block with dispatch op.
ExecutionBlockOp executionBlock(Block *block);
}
}


#endif // DATAFLOW_UTILS_H

