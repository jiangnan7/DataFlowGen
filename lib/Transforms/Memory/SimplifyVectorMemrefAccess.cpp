#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "heteacc/Transforms/Passes.h"

using namespace mlir;
using namespace heteacc;


namespace {
struct MemoryLoadPattern : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    
    if (llvm::all_of(op.getIndices(), [&](Value operand) {
          return isValidDim(operand) || isValidSymbol(operand);
        })) {
      rewriter.replaceOpWithNewOp<vector::LoadOp>(
        op, op.getVectorType(), op.getSource(), op.getIndices());
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
struct MemoryStorePattern : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    
    if (llvm::all_of(op.getIndices(), [&](Value operand) {
            return isValidDim(operand) || isValidSymbol(operand);
            })) {
        rewriter.replaceOpWithNewOp<vector::StoreOp>(
            op, op.getVector(), op.getSource(), op.getIndices());
        return success();
        }
        return failure();

  }
};
} // namespace

namespace {
struct SimplifyVectorMemrefAccess
    : public SimplifyVectorMemrefAccessBase<SimplifyVectorMemrefAccess> {
  void runOnOperation() override { 
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<MemoryLoadPattern>(context);
    patterns.add<MemoryStorePattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
   }
};
} // namespace

std::unique_ptr<Pass> heteacc::createSimplifyVectorMemrefAccessPass() {
  return std::make_unique<SimplifyVectorMemrefAccess>();
}
