#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "heteacc/Transforms/Passes.h"
#include "heteacc/Misc/VecUtils.h"

using namespace mlir;
using namespace heteacc;
using namespace vectorization;



// Given the LHS and RHS of an `arith::AddIOp`, if one of them is defined by an
// `arith::MulIOp`, return a tuple with the `lhs`, `rhs`, and `acc` of the MAC
// operation that can replace them.
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

// This pattern replaces `arith.muli`+`arith.addi` on vectors with
// `aievec.mac_elem`. This pattern works for aie-ml.
// namespace {
//   struct ConvertMulAddToAIEVecFMAElemOpPattern
//     : public OpRewritePattern<arith::AddIOp> {
//   using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

//   // ConvertMulAddToAIEVecFMAElemOpPattern(MLIRContext *context)
//   //     : OpConversionPattern<arith::AddIOp>(context) {}

//   LogicalResult matchAndRewrite(arith::AddIOp addOp,
//                   PatternRewriter &rewriter) const override {
//     // Verify it's a vector operation
//     VectorType resultType = dyn_cast<VectorType>(addOp.getType());

//     if (!resultType)
//       return failure();

//     // // Verify it can be replaced by a MAC
//     auto res =
//         extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
//     // auto lhsDefOp = addOp.getLhs().getDefiningOp();
//     // auto rhsDefOp = addOp.getRhs().getDefiningOp();
//     // arith::MulIOp mulOp = nullptr;

//     // if (lhsDefOp) {
//     //   mulOp = dyn_cast<arith::MulIOp>(lhsDefOp);
//     //   // acc = addRhs;
//     // }
//     // if (!mulOp && rhsDefOp) {
//     //   mulOp = dyn_cast<arith::MulIOp>(rhsDefOp);
//     //   // acc = addLhs;
//     // }
//     if (!res)
//       return failure();
//     auto [lhs, rhs, acc] = *res;

//     // // Verify the vector type is supported by AIEML
//     unsigned resultElWidth =
//         resultType.getElementType().getIntOrFloatBitWidth();
//     unsigned laneSize = getVectorLaneSize(resultType);
//     llvm::outs() << "callInIfwithop.size() " <<resultElWidth<< "\n";
//     llvm::outs() << "callInIfwithop() " <<laneSize<< "\n";
//     // if ((laneSize != 32 || resultElWidth != 16) &&
//     //     (laneSize != 16 || resultElWidth != 32))
//     //   return failure();
//     // Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
//     //                                    /*AIEML =*/true);
//     // auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc,
//     //                                             shiftParam);//accType,
//     rewriter.setInsertionPoint(addOp);
//     auto fmaElemOp = rewriter.create<vector::FMAOp>(
//         addOp.getLoc(), addOp.getType(), lhs, rhs, acc);
//     rewriter.replaceOpWithNewOp<vector::FMAOp>(
//         addOp, resultType, fmaElemOp.getResult());
//     //  rewriter.eraseOp(mulOp);

//     // rewriter.replaceOpWithNewOp(addOp, fmaElemOp.getResult()); //   rewriter.eraseOp(mulOp);
//     return success();
//   }

//   // unsigned shiftParam;
// };

// }

bool heteacc::applyOpFusion(func::FuncOp func){
  auto builder = OpBuilder(func);
  auto context = func.getContext();

  func->walk([&] (arith::AddIOp addOp) {
     VectorType resultType = dyn_cast<VectorType>(addOp.getType());
     if(!resultType)
      return WalkResult::interrupt();
     auto res =
        extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
      if (!res)
        return WalkResult::interrupt();
      auto [lhs, rhs, acc] = *res; llvm::outs() << "return " << "\n";
       lhs.getDefiningOp()->dump();
        rhs.getDefiningOp()->dump();
         acc.getDefiningOp()->dump();
  });

}

namespace {


 struct ConvertMulAddToAIEVecFMAElemOpPattern
    : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern<arith::AddFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddFOp addOp,
                  PatternRewriter &rewriter) const override {
    // Verify it's a vector operation
    VectorType resultType = dyn_cast<VectorType>(addOp.getType());

    if (!resultType)
      return failure();

    // // Verify it can be replaced by a MAC
    auto res =
        extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
    // auto lhsDefOp = addOp.getLhs().getDefiningOp();
    // auto rhsDefOp = addOp.getRhs().getDefiningOp();
    // arith::MulIOp mulOp = nullptr;

    // if (lhsDefOp) {
    //   mulOp = dyn_cast<arith::MulIOp>(lhsDefOp);
    //   // acc = addRhs;
    // }
    // if (!mulOp && rhsDefOp) {
    //   mulOp = dyn_cast<arith::MulIOp>(rhsDefOp);
    //   // acc = addLhs;
    // }
    if (!res)
      return failure();
    auto [lhs, rhs, acc] = *res;

    // // Verify the vector type is supported by AIEML
    unsigned resultElWidth =
        resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = getVectorLaneSize(resultType);
    // llvm::outs() << "callInIfwithop.size() " <<resultElWidth<< "\n";
    // llvm::outs() << "callInIfwithop() " <<laneSize<< "\n";
    // if ((laneSize != 32 || resultElWidth != 16) &&
    //     (laneSize != 16 || resultElWidth != 32))
    //   return failure();

    // Type accType = getVectorOpDestType(cast<VectorType>(acc.getType()),
    //                                    /*AIEML =*/true);
    // auto upsOp = rewriter.create<aievec::UPSOp>(addOp.getLoc(), accType, acc,
    //                                             shiftParam);//accType,


    //  VectorType resultType = dyn_cast<VectorType>(addOp.getType());
    //  if(!resultType)
    //   return WalkResult::interrupt();
    //  auto res =
    //     extractMACOperandsFromAddOperands(addOp.getLhs(), addOp.getRhs());
    //   if (!res)
    //     return WalkResult::interrupt();
    //   auto [lhs, rhs, acc] = *res; llvm::outs() << "return " << "\n";
    //    lhs.getDefiningOp()->dump();
    //     rhs.getDefiningOp()->dump();
    //      acc.getDefiningOp()->dump();
 
    rewriter.setInsertionPoint(addOp);
    auto fmaElemOp = rewriter.create<vector::FMAOp>(
        addOp.getLoc(), addOp.getType(), lhs,rhs, acc);
    // rewriter.replaceOpWithNewOp<vector::FMAOp>(
    //     addOp, resultType, fmaElemOp.getResult());
    //  rewriter.eraseOp(mulOp);

    rewriter.replaceOp(addOp, fmaElemOp.getResult()); //   rewriter.eraseOp(mulOp);
    return success();
  }

  // unsigned shiftParam;
};

struct OperationFusion : public PassWrapper<OperationFusion, OperationPass<func::FuncOp>>  {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperationFusion)
  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final { return "operation-fusion"; }
  StringRef getDescription() const final {
    return "Fusion of vector-related operations for greater efficiency";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, 
                    arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    // applyOpFusion(func);
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    // target.addIllegalOp<Arith::MulOp, Arith::AddOp>();
    // target.addLegalOp<func::FuncOp, func::ReturnOp, func::CallOp>();
    patterns.add<ConvertMulAddToAIEVecFMAElemOpPattern>(context);
(void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    // if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    //   signalPassFailure();
    // }
  }
};
} // namespace

std::unique_ptr<Pass> heteacc::createOperationFusionPass() {
  return std::make_unique<OperationFusion>();
}
