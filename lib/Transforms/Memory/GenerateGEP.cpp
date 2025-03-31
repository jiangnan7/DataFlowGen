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

bool support_linear_memory = false;

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

    if(affineCoeff && support_linear_memory){
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::InputOp>(op, resultType, op.getMemRef(), 
                  affineCoeff.dyn_cast<ArrayAttr>(), affineOffset.dyn_cast<IntegerAttr>());
    }else{
      llvm::SmallVector<Value, 8> dims;
      for(auto a : Indices){
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::LoadOp>(op, resultType, addrOp.getResult(), op->getAttrs());//op.getMemRef(),
      // auto newOp = rewriter.create<memref::LoadOp>(op.getLoc(), resultType, addrOp.getResult(), op->getAttrs());
      // newOp->setAttrs(op->getAttrs());
      // rewriter.replaceOp(op, newOp->getResults());
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

    if(affineCoeff && support_linear_memory){
      auto newOp = rewriter.replaceOpWithNewOp<dataflow::OutputOp>(op, op.getValue(), op.getMemRef(), 
                  affineCoeff.dyn_cast<ArrayAttr>(), affineOffset.dyn_cast<IntegerAttr>());
    }else{
      llvm::SmallVector<Value, 8> dims;
      for(auto a : Indices){
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));
      auto newOp = rewriter.create<dataflow::StoreOp>(op.getLoc(), op.getValue(), addrOp.getResult());//op.getMemRef(), 
      newOp->setAttrs(op->getAttrs());
      rewriter.eraseOp(op);
    }
    
    return success();
  }
};

struct vectorTransferReadLowering 
    : public OpRewritePattern<vector::TransferReadOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    
    auto vecTy = op.getType().dyn_cast<VectorType>();
    if (!vecTy) {
      return failure(); 
    }

    auto memRefShape = op.getSource().getType().getShape();
    unsigned laneCount = vecTy.getNumElements();

    llvm::SmallVector<Value, 8> dims;
    for(auto a : op.getIndices()){
      dims.push_back(a);
    }
    auto addrOp = rewriter.create<AddressOp>(
          op.getLoc(), rewriter.getI32Type(), op.getSource(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));
      addrOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount)); 
    
    auto loadOp = rewriter.replaceOpWithNewOp<dataflow::LoadOp>(
          op, op.getVectorType(), addrOp.getResult(), op->getAttrs());
      loadOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount)); 

    return success();
  }
};


struct vectorTransferWriteLowering 
    : public OpRewritePattern<vector::TransferWriteOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {

    auto vecTy = op.getVector().getType().dyn_cast<VectorType>();
    if (!vecTy) {
      return failure(); 
    }

    unsigned laneCount = vecTy.getNumElements();

    auto memRefShape = op.getSource().getType().getShape();


    bool static_flag = true;    
    llvm::SmallVector<Value, 8> dims;
    for(auto a : op.getIndices()){
      dims.push_back(a);
      if(!a.getDefiningOp<arith::ConstantOp>())
        static_flag = false;
    }

    Value vecValue = op.getVector();

    auto addrOp = rewriter.create<AddressOp>(
    op.getLoc(), rewriter.getI32Type(), op.getSource(), ValueRange{dims}, rewriter.getI64ArrayAttr(memRefShape));

    auto newOp = rewriter.create<dataflow::StoreOp>(op.getLoc(), op.getVector(), addrOp.getResult());
    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount)); 
    if(static_flag){
      newOp->setAttr("loadNums", rewriter.getI32IntegerAttr(1)); 
    }
    rewriter.eraseOp(op);

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
    patterns.add<vectorTransferReadLowering>(context, /*benefit=*/1);
    patterns.add<vectorTransferWriteLowering>(context, /*benefit=*/1);
    // patterns.add<vectorIndexLoadLowering>(context, /*benefit=*/1);
    // patterns.add<vectorIndexStoreLowering>(context, /*benefit=*/1);

    ConversionTarget target(*context);
    target.addIllegalDialect<mlir::AffineDialect, scf::SCFDialect>();
    target.addIllegalOp<memref::LoadOp, memref::StoreOp>();
    target.addIllegalOp<vector::TransferReadOp, vector::TransferWriteOp>();
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
