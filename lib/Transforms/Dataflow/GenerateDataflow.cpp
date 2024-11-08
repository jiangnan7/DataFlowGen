#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "heteacc/Dialect/DataFlow/Utils.h"
#include "heteacc/Misc/Utils.h"
#include "heteacc/Transforms/Passes.h"
using namespace mlir;
using namespace heteacc;
using namespace dataflow;


namespace {
struct TaskPartition : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LaunchOp launch,
                                PatternRewriter &rewriter) const override { 
    if (llvm::any_of(launch.getOps(), [](Operation &op) {
          return isa<bufferization::BufferizationDialect, tosa::TosaDialect,
                     tensor::TensorDialect, linalg::LinalgDialect>(
                     op.getDialect()) ||
                 isa<func::CallOp, LaunchOp, TaskOp>(op);
        }))
      return failure();
    auto &block = launch.getRegion().front();

    // Fuse operations into dataflow tasks. TODO: We need more case study to
    // figure out any other operations need to be separately handled. For
    // example, how to handle AffineIfOp?
    SmallVector<Operation *, 4> opsToFuse;
    unsigned taskIdx = 0;
    for (auto &op : llvm::make_early_inc_range(block)) {
      if (hasEffect<MemoryEffects::Allocate>(&op)) {
        // Memory allocs are moved to the begining and skipped.
        op.moveBefore(&block, block.begin());

      } else if (isa<AffineForOp, scf::ForOp>(op)) {//, dataflow::ForOp, scf::IfOp
        // We always take loop as root operation and fuse all the collected
        // operations so far.
        opsToFuse.push_back(&op);
        fuseOpsIntoTask(opsToFuse, rewriter);
        opsToFuse.clear();
        taskIdx++;

      } else if (&op == block.getTerminator()) {
        // If the block will only generate one task, stop it.
        if (opsToFuse.empty() || taskIdx == 0)
          continue;
        fuseOpsIntoTask(opsToFuse, rewriter);
        opsToFuse.clear();
        taskIdx++;

      } else {
        // Otherwise, we push back the current operation to the list.
        opsToFuse.push_back(&op);
      }
    }
    return success();
  }
};
} // namespace


struct GenerateDataflow
    : public GenerateDataflowBase<GenerateDataflow> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    launchBlock(&func.front());
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands, /*allowHavingChilds=*/true);

    int loop_band = 0;
    for (auto &band : llvm::reverse(targetBands)){
      int loop_level = 0;
      // executionBlock(band.back().getBody());
      for(auto &loop : band){
        OpBuilder b(loop);
        loop->setAttr("Loop_Band", b.getI32IntegerAttr(loop_band));
        loop->setAttr("Loop_Level", b.getI32IntegerAttr(loop_level));
        loop_level++;
      }
      loop_band++;
    }
      

    mlir::RewritePatternSet patterns(context);
    patterns.add<TaskPartition>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};






std::unique_ptr<Pass> heteacc::createGenerateDataflowPass() {
  return std::make_unique<GenerateDataflow>();
}
