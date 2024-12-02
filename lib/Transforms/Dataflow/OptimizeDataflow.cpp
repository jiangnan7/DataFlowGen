#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "heteacc/Dialect/DataFlow/Utils.h"
#include "heteacc/Misc/Utils.h"
#include "heteacc/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
using namespace mlir;
using namespace heteacc;
using namespace dataflow;

// Reference to mlir/lib/Conversion/AffineToStandard/AffineToStandard.cpp
class LowerAffineFor : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lowerBound;
    Value upperBound;
    Value step; 
    // if(op.hasConstantUpperBound()){
    //   lowerBound = rewriter.create<arith::ConstantIntOp>(loc, op.getConstantLowerBound(), 32);
    //   upperBound = rewriter.create<arith::ConstantIntOp>(loc, op.getConstantUpperBound(), 32);
    //   step = rewriter.create<arith::ConstantIntOp>(loc, op.getStep(), 32);
    // } else {
    //   lowerBound = lowerAffineLowerBound(op, rewriter);
    //   upperBound = lowerAffineUpperBound(op, rewriter);
    //   step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
    // }
    lowerBound = lowerAffineLowerBound(op, rewriter);
    upperBound = lowerAffineUpperBound(op, rewriter);
    step = rewriter.create<arith::ConstantIndexOp>(loc, op.getStep());
    auto forop = rewriter.create<dataflow::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());

    // Pass loop band and level to the generated loop.
    for (auto attr : op->getAttrs()) {
        if (attr.getName().str() == "Loop_Band" || attr.getName().str() == "Loop_Level") {
            forop->setAttr(attr.getName(), attr.getValue());
        }
    }

    // if(op.hasConstantUpperBound())
    // auto *conditionBlock = &op.getRegion().front();
    // auto iv = conditionBlock->getArgument(0);
    // auto *lastBodyBlock  = &op.getRegion().back();

    // auto stepped = rewriter.create<arith::AddIOp>(loc, iv, step).getResult();
    // if (!stepped)
    //   return failure();
    // auto comparison = rewriter.create<arith::CmpIOp>(
    //           loc, arith::CmpIPredicate::eq, stepped, upperBound);

    rewriter.eraseBlock(forop.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), forop.getRegion(),
                                forop.getRegion().end());
    rewriter.replaceOp(op, forop.getResults());
    return success();
  }
};

class SCFForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value lowerBound = op.getLowerBound();
  Value upperBound = op.getUpperBound();
  Value step = op.getStep();
  
  auto forop = rewriter.create<dataflow::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getIterOperands());

  rewriter.eraseBlock(forop.getBody());
  rewriter.inlineRegionBefore(op.getRegion(), forop.getRegion(),
                              forop.getRegion().end());
  rewriter.replaceOp(op, forop.getResults());

  return success();

  }
};

class SCFIfOpConversion : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

  rewriter.setInsertionPoint(op);
  SmallVector<Type, 4> resultTypes(op.getResultTypes());

  for (auto &type : resultTypes)
    if (type.isa<IndexType>())
      type = IntegerType::get(getContext(), 32);
  bool hasElseRegion = !op.getElseRegion().empty();
  auto newOp = rewriter.create<dataflow::IfOp>(op.getLoc(), resultTypes,
                                            op.getCondition(), hasElseRegion);
  // rewriter.eraseBlock(newOp.getThenRegion());
  // rewriter.createBlock(&newOp.getThenRegion());
  rewriter.inlineRegionBefore(op.getThenRegion(), &newOp.getThenRegion().back());
  rewriter.eraseBlock(&newOp.getThenRegion().back());

  if (hasElseRegion) {
    // rewriter.eraseBlock(newOp.getElseRegion());
    // rewriter.createBlock(&newOp.getElseRegion());
    rewriter.inlineRegionBefore(op.getElseRegion(), &newOp.getElseRegion().back());
    rewriter.eraseBlock(&newOp.getElseRegion().back());
  }

  rewriter.replaceOp(op, newOp.getResults());

  return success();

  }
};

/// Affine yields ops are removed.
class SCFYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
  auto loc = rewriter.getUnknownLoc();
    
      llvm::SmallVector<Value, 8> inputs;

      for(auto a : op.getOperands()){
      inputs.push_back(a);
    }
    rewriter.replaceOpWithNewOp<dataflow::YieldOp>(op, ValueRange{inputs});
  return success();
  }
};

/// Affine yields ops are removed.
class AffineYieldOpLowering : public OpRewritePattern<AffineYieldOp> {
public:
  using OpRewritePattern<AffineYieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) const override {
 
  auto loc = rewriter.getUnknownLoc();
    
      llvm::SmallVector<Value, 8> inputs;
      for(auto a : op.getOperands()){
      inputs.push_back(a);
    } 
    rewriter.replaceOpWithNewOp<dataflow::YieldOp>(op, ValueRange{inputs});
    return success();
  }
};

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build vector.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(),
                                                *resultOperands);
    return success();
  }
};

class AffineIfLowering : public OpRewritePattern<AffineIfOp> {
public:
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = op.getIntegerSet();
    Value zeroConstant = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value, 8> operands(op.getOperands());
    auto operandsRef = llvm::makeArrayRef(operands);

    // Calculate cond as a conjunction without short-circuiting.
    Value cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                         operandsRef.take_front(numDims),
                                         operandsRef.drop_front(numDims));
      if (!affResult)
        return failure();
      auto pred =
          isEquality ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value cmpVal =
          rewriter.create<arith::CmpIOp>(loc, pred, affResult, zeroConstant);
      cond = cond
                 ? rewriter.create<arith::AndIOp>(loc, cond, cmpVal).getResult()
                 : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<arith::ConstantIntOp>(loc, /*value=*/1,
                                                        /*width=*/1);

    bool hasElseRegion = !op.getElseRegion().empty();
    auto ifOp = rewriter.create<dataflow::IfOp>(loc, op.getResultTypes(), cond,
                                           hasElseRegion);
    rewriter.inlineRegionBefore(op.getThenRegion(),
                                &ifOp.getThenRegion().back());
    rewriter.eraseBlock(&ifOp.getThenRegion().back());
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(op.getElseRegion(),
                                  &ifOp.getElseRegion().back());
      rewriter.eraseBlock(&ifOp.getElseRegion().back());
    }

    // Replace the Affine IfOp finally.
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};


/// Convert an "affine.apply" operation into a sequence of arithmetic
/// operations using the StandardOps dialect.
class AffineApplyLowering : public OpRewritePattern<AffineApplyOp> {
public:
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                        llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
    return success();
  }
};

class RemsiToAndPattern : public OpRewritePattern<arith::RemSIOp> {
public:
  using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::RemSIOp remOp, PatternRewriter &rewriter) const override {
      auto rhs = remOp.getRhs();
      if (auto constant = rhs.getDefiningOp<arith::ConstantOp>()) {

        APInt rhsValue = constant.getValue().cast<IntegerAttr>().getValue();
        if (rhsValue.isPowerOf2()) {
          auto inputType = remOp.getType();
          Value constantValue;
          if (inputType.isa<mlir::IndexType>()) {
            constantValue = rewriter.create<arith::ConstantOp>(
                remOp.getLoc(), rewriter.getIndexAttr(rhsValue.getSExtValue() - 1));
          } else if (auto intType = inputType.dyn_cast<mlir::IntegerType>()) {
            constantValue = rewriter.create<arith::ConstantOp>(
                remOp.getLoc(), inputType,
                rewriter.getIntegerAttr(intType, rhsValue - 1));
          } else {
            return failure();
          }
          auto andOp = rewriter.create<arith::AndIOp>(
              remOp.getLoc(), remOp.getLhs(), constantValue);
          rewriter.replaceOp(remOp, andOp.getResult());
          return success();
        }
      }
      return failure();
    }
};

// class ArithSelectConversion : public OpConversionPattern<arith::SelectOp> {
// public:
//   using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

//   LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
//                                 ConversionPatternRewriter &rewriter) const override {
//     Value condition = op.getCondition();
//     Value trueValue = op.getTrueValue();
//     Value falseValue = op.getFalseValue();
//     rewriter.replaceOpWithNewOp<dataflow::SelectOp>(
//         op, op.getResult().getType(), condition, trueValue, falseValue);

//     return success();
//   }
// };
struct OptimizeDataflow
    : public OptimizeDataflowBase<OptimizeDataflow> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    
    patterns.add<AffineApplyLowering>(context, /*benefit=*/1);
    patterns.add<AffineLoadLowering>(context, /*benefit=*/1);
    
    patterns.add<AffineStoreLowering>(context, /*benefit=*/1);
    patterns.add<LowerAffineFor>(context, /*benefit=*/1);
    patterns.add<AffineIfLowering>(context, /*benefit=*/1);
    patterns.add<AffineYieldOpLowering>(context, /*benefit=*/1);
    patterns.add<SCFForOpConversion>(context, /*benefit=*/1);
    patterns.add<SCFIfOpConversion>(context, /*benefit=*/1);
    patterns.add<SCFYieldOpConversion>(context, /*benefit=*/1);
    // patterns.add<ArithSelectConversion>(context, /*benefit=*/1);

    ConversionTarget target(*context);
    target.addIllegalDialect<mlir::AffineDialect,scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, heteacc::dataflow::DataFlowDialect,
                          vector::VectorDialect>();
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();


    mlir::RewritePatternSet operator_patterns(context);
    operator_patterns.insert<RemsiToAndPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(operator_patterns));
  }
};






std::unique_ptr<Pass> heteacc::createOptimizeDataflowPass() {
  return std::make_unique<OptimizeDataflow>();
}
