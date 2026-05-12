#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"

#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Transforms/Passes.h"
using namespace mlir;
using namespace heteacc;
using namespace vectorization;

/// Given the LHS and RHS of an AddFOp, if one is defined by a MulFOp,
/// return a tuple (lhs, rhs, acc) for a fused multiply-add replacement.
static std::optional<std::tuple<Value, Value, Value>>
extractMACOperandsFromAddOperands(Value addLhs, Value addRhs) {
  auto lhsDefOp = addLhs.getDefiningOp();
  auto rhsDefOp = addRhs.getDefiningOp();
  arith::MulFOp mulOp = nullptr;
  Value acc;
  if (lhsDefOp) {
    mulOp = dyn_cast<arith::MulFOp>(lhsDefOp);
    acc = addRhs;
  }
  if (!mulOp && rhsDefOp) {
    mulOp = dyn_cast<arith::MulFOp>(rhsDefOp);
    acc = addLhs;
  }
  if (!mulOp)
    return {};
  return std::make_tuple(mulOp.getLhs(), mulOp.getRhs(), acc);
}

bool heteacc::applyOpFusion(func::FuncOp func) {
  func->walk([&](arith::AddIOp addOp) {
    VectorType resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return WalkResult::interrupt();
    auto res =
        extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
    if (!res)
      return WalkResult::interrupt();
    (void)*res;
    return WalkResult::advance();
  });
  return true;
}

namespace {

/// Fuse arith.mulf + arith.addf into vector.fma on vector types.
struct ConvertMulAddToAIEVecFMAElemOpPattern
    : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp addOp,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = dyn_cast<VectorType>(addOp.getType());
    if (!resultType)
      return failure();

    auto res =
        extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    rewriter.setInsertionPoint(addOp);
    auto fmaElemOp = rewriter.create<vector::FMAOp>(
        addOp.getLoc(), addOp.getType(), lhs, rhs, acc);
    rewriter.replaceOp(addOp, fmaElemOp.getResult());
    return success();
  }
};

struct OperationFusion
    : public PassWrapper<OperationFusion, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperationFusion)

  StringRef getArgument() const final { return "operation-fusion"; }
  StringRef getDescription() const final {
    return "Fusion of vector-related operations for greater efficiency";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ConvertMulAddToAIEVecFMAElemOpPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> heteacc::createOperationFusionPass() {
  return std::make_unique<OperationFusion>();
}
