///@xcgao
#include "mlir/Pass/Pass.h"

#include "mlir/IR/Operation.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "heteacc/Transforms/Passes.h"

#include <queue>

using namespace mlir;
using namespace heteacc;
using namespace dataflow;


namespace {

// using mlir::affine::AffineForOp;
// using mlir::affine::AffineLoadOp;
// using mlir::affine::AffineStoreOp;
using mlir::func::FuncOp;

class memrefLoadLowering : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {

    // Build memref.store valueToStore, memref[expandedMap.results].
    Attribute affineCoeff = op->getAttr("affineCoeff");
    Attribute affineOffset = op->getAttr("affineOffset");
    auto Indices = op.getIndices();
    auto memRefShape = op.getMemRefType().getShape();
    auto resultType = op.getResult().getType();

    if(affineCoeff){
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::InputOp>(op, resultType, op.getMemRef(), 
                  affineCoeff.dyn_cast<ArrayAttr>(), affineOffset.dyn_cast<IntegerAttr>());
    }else{
      llvm::SmallVector<Value, 8> dims;
      for(auto a : Indices){
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::LoadOp>(op, resultType, addrOp.getResult());//op.getMemRef(),
    }
    
    return success();
  }
};

class memrefStoreLowering : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {

    // Build memref.store valueToStore, memref[expandedMap.results].
    Attribute affineCoeff = op->getAttr("affineCoeff");
    Attribute affineOffset = op->getAttr("affineOffset");
    auto Indices = op.getIndices();
    auto memRefShape = op.getMemRefType().getShape();

    if(affineCoeff){
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::OutputOp>(op, op.getValue(), op.getMemRef(), 
                  affineCoeff.dyn_cast<ArrayAttr>(), affineOffset.dyn_cast<IntegerAttr>());
    }else{
      llvm::SmallVector<Value, 8> dims;
      for(auto a : Indices){
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::StoreOp>(op, op.getValue(), addrOp.getResult());//op.getMemRef(), 
    }
    
    return success();
  }
};

// A pass that manually walks the IR
struct GenerateGEP : GenerateGEPBase<GenerateGEP> {

  void runOnOperation(){
    auto func = getOperation();
    auto context = func.getContext();

    mlir::RewritePatternSet patterns(context);
    
    patterns.add<memrefLoadLowering>(context, /*benefit=*/1);
    patterns.add<memrefStoreLowering>(context, /*benefit=*/1);

    ConversionTarget target(*context);
    target.addIllegalDialect<mlir::AffineDialect, scf::SCFDialect>();
    target.addIllegalOp<memref::LoadOp, memref::StoreOp>();
    target.addLegalDialect<arith::ArithDialect, heteacc::dataflow::DataFlowDialect,
                          vector::VectorDialect>();
    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      return signalPassFailure();
  }
};

}

std::unique_ptr<Pass> heteacc::createGenerateGEPPass() {
  return std::make_unique<GenerateGEP>();
}
