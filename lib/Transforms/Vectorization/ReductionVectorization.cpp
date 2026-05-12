#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include "heteacc/Dialect/DataFlow/DataFlow.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Vectorization/ReductionVectorization.h"

using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

void heteacc::vectorizeConditionalReduction(ModuleOp module, unsigned factor) {
  MLIRContext *context = module.getContext();
  IRRewriter rewriter(context);

  module.walk([&](dataflow::ForOp forOp) {
    // Skip nested loops (only handle innermost).
    bool hasNestedFor = false;
    forOp.getLoopBody().walk([&](dataflow::ForOp) { hasNestedFor = true; });
    if (hasNestedFor)
      return;

    // Only handle loops with exactly one iter_arg of integer type.
    if (forOp.getIterOperands().size() != 1)
      return;
    Value iterInit = *forOp.getIterOperands().begin();
    Type iterType = iterInit.getType();
    if (!iterType.isa<IntegerType>())
      return;

    // Check the loop body for the pattern:
    // Pattern A: select chain (conditional reduction)
    // Pattern B: addi chain (unconditional reduction)
    Block &body = forOp.getLoopBody().front();
    auto yieldOp = dyn_cast<dataflow::YieldOp>(body.getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
      return;

    // Find the chain of selects (Pattern A) or addis (Pattern B).
    SmallVector<dataflow::SelectOp, 4> selectChain;
    SmallVector<arith::CmpIOp, 4> cmpOps;
    SmallVector<arith::AddIOp, 4> addOps;
    SmallVector<Value, 4> accumulatedValues;
    bool isConditional = false;

    Value current = yieldOp.getOperand(0);

    // Try Pattern A: select chain.
    while (auto selectOp = current.getDefiningOp<dataflow::SelectOp>()) {
      selectChain.push_back(selectOp);
      Value trueVal = selectOp.getTrueValue();
      Value falseVal = selectOp.getFalseValue();
      Value condition = selectOp.getCondition();

      auto addOp = trueVal.getDefiningOp<arith::AddIOp>();
      if (!addOp)
        break;

      auto cmpOp = condition.getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        break;

      addOps.push_back(addOp);
      cmpOps.push_back(cmpOp);

      if (addOp.getLhs() == falseVal) {
        accumulatedValues.push_back(addOp.getRhs());
      } else if (addOp.getRhs() == falseVal) {
        accumulatedValues.push_back(addOp.getLhs());
      } else {
        break;
      }
      current = falseVal;
    }

    if (selectChain.size() == factor) {
      isConditional = true;
      auto pred = cmpOps[0].getPredicate();
      for (auto cmpOp : cmpOps) {
        if (cmpOp.getPredicate() != pred)
          return;
      }
    } else {
      // Try Pattern B: unconditional addi chain.
      selectChain.clear();
      addOps.clear();
      accumulatedValues.clear();

      current = yieldOp.getOperand(0);
      while (auto addOp = current.getDefiningOp<arith::AddIOp>()) {
        addOps.push_back(addOp);
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();

        if (rhs.getDefiningOp<arith::AddIOp>() || rhs == body.getArgument(1)) {
          accumulatedValues.push_back(lhs);
          current = rhs;
        } else if (lhs.getDefiningOp<arith::AddIOp>() ||
                   lhs == body.getArgument(1)) {
          accumulatedValues.push_back(rhs);
          current = lhs;
        } else {
          break;
        }
      }

      if (addOps.size() != factor)
        return;

      std::reverse(accumulatedValues.begin(), accumulatedValues.end());
      std::reverse(addOps.begin(), addOps.end());
    }

    // Check that all accumulated values have slp.group.
    for (auto val : accumulatedValues) {
      if (!val.getDefiningOp() || !val.getDefiningOp()->hasAttr("slp.group"))
        return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found "
                            << (isConditional ? "conditional" : "unconditional")
                            << " reduction pattern with "
                            << accumulatedValues.size() << " values\n");

    // === Transform ===
    auto intType = iterType.cast<IntegerType>();
    auto vecType = VectorType::get({static_cast<int64_t>(factor)}, intType);

    rewriter.setInsertionPoint(forOp);
    auto zeroAttr =
        DenseElementsAttr::get(vecType, rewriter.getIntegerAttr(intType, 0));
    Value zeroVec =
        rewriter.create<arith::ConstantOp>(forOp.getLoc(), vecType, zeroAttr);

    SmallVector<Value> newIterOperands = {zeroVec};
    auto newForOp = rewriter.create<dataflow::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newIterOperands,
        [](OpBuilder &, Location, Value, ValueRange) {});

    for (auto attr : forOp->getAttrs()) {
      if (attr.getName() != "operand_segment_sizes")
        newForOp->setAttr(attr.getName(), attr.getValue());
    }

    Block &newBody = newForOp.getLoopBody().front();
    rewriter.setInsertionPointToStart(&newBody);
    Location loc = forOp.getLoc();

    Value accVec = newBody.getArgument(1);
    Value iv = newBody.getArgument(0);

    // Find the memref from original loads.
    Value memref;
    for (auto &op : body) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        memref = loadOp.getMemref();
        break;
      }
    }
    if (!memref) {
      rewriter.eraseOp(newForOp);
      return;
    }

    Value padVal = rewriter.create<arith::ConstantOp>(
        loc, intType, rewriter.getIntegerAttr(intType, 0));
    Value loadVec = rewriter.create<vector::TransferReadOp>(
        loc, vecType, memref, ValueRange{iv}, padVal);

    // Reconstruct the computation chain on vectors.
    Value computeResult = loadVec;
    auto *firstAccOp = accumulatedValues[0].getDefiningOp();
    SmallVector<Operation *, 8> computeChain;

    std::function<void(Operation *)> traceChain = [&](Operation *op) {
      if (!op || isa<memref::LoadOp>(op))
        return;
      for (auto operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (isa<memref::LoadOp>(defOp) ||
              defOp->hasTrait<OpTrait::ConstantLike>())
            continue;
          if (defOp->getBlock() == &body && defOp->hasAttr("slp.group"))
            traceChain(defOp);
        }
      }
      computeChain.push_back(op);
    };
    traceChain(firstAccOp);

    // Replay the compute chain with vector types.
    DenseMap<Operation *, Value> vecMapping;
    for (auto *op : computeChain) {
      SmallVector<Value, 2> newOperands;
      for (auto operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (isa<memref::LoadOp>(defOp)) {
            newOperands.push_back(loadVec);
          } else if (defOp->hasTrait<OpTrait::ConstantLike>()) {
            newOperands.push_back(
                rewriter.create<vector::BroadcastOp>(loc, vecType, operand));
          } else if (vecMapping.count(defOp)) {
            newOperands.push_back(vecMapping[defOp]);
          } else {
            newOperands.push_back(operand);
          }
        } else {
          newOperands.push_back(accVec);
        }
      }
      Value result = rewriter
                         .create(loc, op->getName().getIdentifier(),
                                 newOperands, vecType, op->getAttrs())
                         ->getResult(0);
      vecMapping[op] = result;
      computeResult = result;
    }

    // Final accumulation.
    Value finalResult;
    if (isConditional) {
      Value addVec =
          rewriter.create<arith::AddIOp>(loc, vecType, computeResult, accVec);
      auto cmpVecType = VectorType::get({static_cast<int64_t>(factor)},
                                         rewriter.getI1Type());
      Value cmpConst;
      for (auto operand : cmpOps[0]->getOperands()) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
          cmpConst = rewriter.create<vector::BroadcastOp>(loc, vecType,
                                                           constOp.getResult());
          break;
        }
      }
      if (!cmpConst) {
        // Fallback: broadcast a zero constant if no constant operand found.
        cmpConst = rewriter.create<vector::BroadcastOp>(
            loc, vecType,
            rewriter.create<arith::ConstantOp>(
                loc, intType, rewriter.getIntegerAttr(intType, 0)));
      }
      Value cmpVec = rewriter.create<arith::CmpIOp>(
          loc, cmpVecType, cmpOps[0].getPredicate(), computeResult, cmpConst);
      finalResult = rewriter.create<dataflow::SelectOp>(loc, vecType, cmpVec,
                                                         addVec, accVec);
    } else {
      finalResult =
          rewriter.create<arith::AddIOp>(loc, vecType, computeResult, accVec);
    }

    rewriter.create<dataflow::YieldOp>(loc, ValueRange{finalResult});

    // After the loop: reduce vector to scalar.
    rewriter.setInsertionPointAfter(newForOp);
    Value loopResult = newForOp.getResult(0);
    Value reduced = rewriter.create<vector::ReductionOp>(
        loc, vector::CombiningKind::ADD, loopResult);

    if (auto constInit = iterInit.getDefiningOp<arith::ConstantOp>()) {
      auto initAttr = constInit.getValue().dyn_cast<IntegerAttr>();
      if (initAttr && initAttr.getInt() != 0) {
        reduced = rewriter.create<arith::AddIOp>(loc, reduced, iterInit);
      }
    }

    forOp.getResult(0).replaceAllUsesWith(reduced);
    rewriter.eraseOp(forOp);
  });
}
