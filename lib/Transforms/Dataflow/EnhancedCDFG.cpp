#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "heteacc/Dialect/DataFlow/Utils.h"
#include "heteacc/Misc/Utils.h"
#include "heteacc/Transforms/Passes.h"
using namespace mlir;
using namespace heteacc;
using namespace dataflow;

static bool hasLoopStartState(dataflow::TaskOp taskop) {
  for (auto &op : taskop.getBody().front()) {
    if (auto state = dyn_cast<dataflow::StateOp>(op)) {
      if (state.getInstanceName().str().find("loop_start") != std::string::npos)
        return true;
      if (auto enableAttr = state->getAttrOfType<StringAttr>("Enable")) {
        if (enableAttr.getValue().find("Loop_Start") != std::string::npos)
          return true;
      }
    }
  }
  return false;
}

static bool hasLoopExitState(dataflow::ForOp forop) {
  for (auto &op : forop.getLoopBody().front()) {
    if (auto exeop = dyn_cast<dataflow::ExecutionBlockOp>(op)) {
      for (auto &exeInnerOp : exeop.getBody().front()) {
        if (auto state = dyn_cast<dataflow::StateOp>(exeInnerOp)) {
          if (state.getInstanceName().str().find("loop_exit") !=
              std::string::npos)
            return true;
          if (auto exeAttr = state->getAttrOfType<StringAttr>("Exe")) {
            if (exeAttr.getValue().find("Loop") != std::string::npos)
              return true;
          }
        }
      }
    }
  }
  return false;
}

static bool hasStaticLoopBounds(dataflow::ForOp forop) {
  return forop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>() &&
         forop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>() &&
         forop.getStep().getDefiningOp<arith::ConstantIndexOp>();
}

static bool needsExplicitLoopExitState(dataflow::ForOp forop) {
  if (!hasStaticLoopBounds(forop))
    return true;
  return static_cast<bool>(forop->getParentOfType<dataflow::ForOp>());
}

namespace {
struct RefineFunc : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcop,
                                PatternRewriter &rewriter) const override {
    // TODO Refine the EXEblock.
    for (auto &op : funcop.getBody().front())
      if (isa<dataflow::ForOp>(op)) { // TODO Support more op.
        auto loc = funcop.getLoc();   // rewriter.setInsertionPoint(&op);
        Value loopSignal = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq,
            rewriter.create<arith::ConstantIndexOp>(loc, 0), // TODO index/int
            dyn_cast<dataflow::ForOp>(op).getLowerBound());
        rewriter.create<dataflow::EnableOp>(loc, loopSignal.getType(),
                                            loopSignal);
        return success();
      }
    return failure();
  }
};
} // namespace

struct EnhancedCDFG : public EnhancedCDFGBase<EnhancedCDFG> {
  void runOnOperation() override {
    auto func = getOperation();

    // executionBlock(&func.front());

    func.walk([&](dataflow::ForOp forop) {
      executionBlock(&forop.getLoopBody().front());
    });

    func.walk([&](dataflow::TaskOp taskop) {
      if (hasLoopStartState(taskop))
        return WalkResult::advance();

      mlir::Operation *countOp = nullptr;
      uint32_t count = 0;
      for (auto &op : taskop.getBody().front()) {
        countOp = &op;
        if (isa<dataflow::ForOp, AffineForOp, scf::IfOp, dataflow::IfOp>(op)) {
          break;
        } else if (isa<arith::ConstantIntOp, arith::ConstantOp,
                       arith::ConstantIndexOp, memref::AllocaOp,
                       arith::IndexCastOp>(op)) {
          continue;
        } else {
          count += 1;
          countOp = taskop->getParentOp();
        }
      }

      if (!countOp)
        return WalkResult::advance();

      OpBuilder builder(countOp);
      auto trueSignal =
          builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
      auto loopStart =
          StringAttr::get(builder.getContext(), "loop_start");
      auto nullState = StringAttr::get(builder.getContext(), "null");

      if (count == 0 && isa<dataflow::ForOp>(countOp)) {
        builder.setInsertionPoint(countOp);
      } else {
        builder.setInsertionPointToStart(&taskop.getBody().front());
      }

      auto state = builder.create<dataflow::StateOp>(builder.getUnknownLoc(),
                                                     trueSignal, loopStart,
                                                     nullState);
      state->setAttr("Enable",
                     StringAttr::get(builder.getContext(), "Loop_Start"));
      return WalkResult::advance();
    });

    func.walk([&](dataflow::ForOp forop) {
      if (hasLoopExitState(forop))
        return WalkResult::advance();

      dataflow::ExecutionBlockOp exeop;
      for (auto &opiter : forop.getRegion().front()) {
        if (auto candidate = dyn_cast<dataflow::ExecutionBlockOp>(opiter)) {
          exeop = candidate;
          break;
        }
      }
      if (!exeop)
        return WalkResult::advance();

      OpBuilder builder(exeop);
      builder.setInsertionPoint(exeop.getBody().front().getTerminator());

      auto ivnew = builder.create<arith::AddIOp>(builder.getUnknownLoc(),
                                                 forop.getInductionVar(),
                                                 forop.getStep());
      ivnew->setAttr("Exe", StringAttr::get(builder.getContext(), "Loop"));

      if (needsExplicitLoopExitState(forop)) {
        Value loopSignal = builder.create<arith::CmpIOp>(
            builder.getUnknownLoc(), arith::CmpIPredicate::eq, ivnew,
            forop.getUpperBound());
        loopSignal.getDefiningOp()->setAttr(
            "Exe", StringAttr::get(builder.getContext(), "Loop"));
        auto loopBack = StringAttr::get(forop.getContext(), "loop_back");
        auto loopExit = StringAttr::get(forop.getContext(), "loop_exit");
        auto state = builder.create<dataflow::StateOp>(
            builder.getUnknownLoc(), loopSignal, loopExit, loopBack);
        state->setAttr("Exe", StringAttr::get(builder.getContext(), "Loop"));
      }
      forop.getRegion().front().getTerminator()->replaceUsesOfWith(
          forop.getInductionVar(), ivnew);
      return WalkResult::advance();
    });
    // mlir::RewritePatternSet patterns(context);
    // patterns.add<RefineFunc>(context);
    // patterns.add<RefineLoop>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

std::unique_ptr<Pass> heteacc::createEnhancedCDFGPass() {
  return std::make_unique<EnhancedCDFG>();
}
