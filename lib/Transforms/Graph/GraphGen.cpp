#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "heteacc/Graph/Graph.h"
#include "heteacc/Graph/GraphGen.h"
#include "heteacc/Graph/Node.h"
#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Transforms/Passes.h"
using namespace mlir;
using namespace heteacc;
using namespace dataflow;
#define DEBUG_TYPE "graph"
bool Linear_memory = false;
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

    if (affineCoeff && Linear_memory) {
      rewriter.replaceOpWithNewOp<dataflow::InputOp>(
          op, resultType, op.getMemRef(), affineCoeff.dyn_cast<ArrayAttr>(),
          affineOffset.dyn_cast<IntegerAttr>());
    } else {
      llvm::SmallVector<Value, 8> dims;
      for (auto a : Indices) {
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(
          op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims},
          rewriter.getI64ArrayAttr(memRefShape));
      rewriter.replaceOpWithNewOp<dataflow::LoadOp>(
          op, resultType, addrOp.getResult(),
          op->getAttrs()); // op.getMemRef(),
      // auto newOp = rewriter.create<memref::LoadOp>(op.getLoc(), resultType,
      // addrOp.getResult(), op->getAttrs()); newOp->setAttrs(op->getAttrs());
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

    if (affineCoeff && Linear_memory) {
      rewriter.replaceOpWithNewOp<dataflow::OutputOp>(
          op, op.getValue(), op.getMemRef(), affineCoeff.dyn_cast<ArrayAttr>(),
          affineOffset.dyn_cast<IntegerAttr>());
    } else {
      llvm::SmallVector<Value, 8> dims;
      for (auto a : Indices) {
        dims.push_back(a);
      }
      auto addrOp = rewriter.create<AddressOp>(
          op.getLoc(), rewriter.getI32Type(), op.getMemRef(), ValueRange{dims},
          rewriter.getI64ArrayAttr(memRefShape));
      auto newOp = rewriter.create<dataflow::StoreOp>(
          op.getLoc(), op.getValue(), addrOp.getResult()); // op.getMemRef(),
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

    auto vecTy = op.getType().dyn_cast<mlir::VectorType>();
    if (!vecTy) {
      return failure();
    }
    bool static_flag = true;
    auto memRefShape = op.getSource().getType().getShape();
    unsigned laneCount = vecTy.getNumElements();

    llvm::SmallVector<Value, 8> dims;
    for (auto a : op.getIndices()) {
      dims.push_back(a);
      if (!a.getDefiningOp<arith::ConstantOp>())
        static_flag = false;
    }
    auto addrOp = rewriter.create<AddressOp>(
        op.getLoc(), rewriter.getI32Type(), op.getSource(), ValueRange{dims},
        rewriter.getI64ArrayAttr(memRefShape));
    addrOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));

    auto loadOp = rewriter.replaceOpWithNewOp<dataflow::LoadOp>(
        op, op.getVectorType(), addrOp.getResult(), op->getAttrs());
    loadOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));
    if (static_flag) {
      addrOp->setAttr("loadNums", rewriter.getI32IntegerAttr(laneCount));
      loadOp->setAttr("loadNums", rewriter.getI32IntegerAttr(laneCount));
    }
    return success();
  }
};

struct vectorTransferWriteLowering
    : public OpRewritePattern<vector::TransferWriteOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {

    auto vecTy = op.getVector().getType().dyn_cast<mlir::VectorType>();
    if (!vecTy) {
      return failure();
    }

    unsigned laneCount = vecTy.getNumElements();

    auto memRefShape = op.getSource().getType().getShape();

    bool static_flag = true;
    llvm::SmallVector<Value, 8> dims;
    ValueRange indices = op.getIndices();
    for (auto a : indices) {
      dims.push_back(a);
      if (!a.getDefiningOp<arith::ConstantOp>())
        static_flag = false;
    }

    Value vecValue = op.getVector();

    auto addrOp = rewriter.create<AddressOp>(
        op.getLoc(), rewriter.getI32Type(), op.getSource(), ValueRange{dims},
        rewriter.getI64ArrayAttr(memRefShape));
    addrOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));
    auto newOp = rewriter.create<dataflow::StoreOp>(op.getLoc(), vecValue,
                                                    addrOp.getResult());
    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));
    if (static_flag) {
      addrOp->setAttr("storeNums", rewriter.getI32IntegerAttr(laneCount));
      newOp->setAttr("storeNums", rewriter.getI32IntegerAttr(laneCount));
    }
    rewriter.eraseOp(op);

    return success();
  }
};

struct vectorIndexLoadLowering
    : public OpRewritePattern<dataflow::VectorIndexLoadOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(dataflow::VectorIndexLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = op.getIndices().getType().dyn_cast<mlir::VectorType>();
    if (!vecTy) {
      return failure();
    }

    unsigned laneCount = vecTy.getNumElements();
    auto memRefShape = op.getSource().getType().getShape();

    auto addrOp = rewriter.create<AddressOp>(
        op.getLoc(), VectorType::get(vecTy.getShape(), rewriter.getI32Type()),
        op.getSource(), ValueRange{op.getIndices()},
        rewriter.getI64ArrayAttr(memRefShape));
    addrOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));
    auto loadOp = rewriter.replaceOpWithNewOp<dataflow::LoadOp>(
        op, op.getVector().getType(), addrOp.getResult(), op->getAttrs());

    loadOp->setAttrs(op->getAttrs());
    loadOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));

    return success();
  }
};

struct vectorIndexStoreLowering
    : public OpRewritePattern<dataflow::VectorIndexStoreOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(dataflow::VectorIndexStoreOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "\nVectorIndexStoreOp\n ");
    auto vecTy = op.getValueToStore().getType().dyn_cast<mlir::VectorType>();
    if (!vecTy) {
      return failure();
    }

    unsigned laneCount = vecTy.getNumElements();
    auto memRefShape = op.getSource().getType().getShape();

    auto addrOp = rewriter.create<AddressOp>(
        op.getLoc(), VectorType::get(vecTy.getShape(), rewriter.getI32Type()),
        op.getSource(), ValueRange{op.getIndices()},
        rewriter.getI64ArrayAttr(memRefShape));
    addrOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));
    auto newOp = rewriter.create<dataflow::StoreOp>(
        op.getLoc(), op.getValueToStore(), addrOp.getResult());
    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("laneNums", rewriter.getI32IntegerAttr(laneCount));

    rewriter.eraseOp(op);

    return success();
  }
};

bool GraphGen::applyGraphInit(func::FuncOp func, bool isTopFunc) {

  if (isTopFunc) {
    this->dependency_graph->top_function = func;
    for (Value operand : func.front().getArguments()) {
      // auto node =
      // this->dependency_graph->getArgCall()->insertLiveInArgument(operand,
      // ArgumentNode::LiveIn);

      this->dependency_graph->funArgValue.push_back(operand);
      if (MemRefType memRefType = operand.getType().dyn_cast<MemRefType>()) {
        ArrayRef<int64_t> shape =
            operand.getType().cast<MemRefType>().getShape();
        int64_t totalElements = std::accumulate(shape.begin(), shape.end(), 1,
                                                std::multiplies<int64_t>());
        (void)memRefType;
        (void)totalElements;
      }

      this->map_value_node[operand] =
          this->dependency_graph->getArgCall()->insertLiveInArgument(
              operand, ArgumentNode::ArgumentType::LiveIn);
      // arg_mem0
    }
    buildMemoryGraph(func);
    // this->dependency_graph->getMemoryUnit()->setMemSize(size);
    // TODO Global Value.
  }
  auto context = func.getContext();

  mlir::RewritePatternSet patterns(context);

  patterns.add<memrefLoadLowering>(context, /*benefit=*/1);
  patterns.add<memrefStoreLowering>(context, /*benefit=*/1);
  patterns.add<vectorTransferReadLowering>(context, /*benefit=*/1);
  patterns.add<vectorTransferWriteLowering>(context, /*benefit=*/1);
  patterns.add<vectorIndexLoadLowering>(context, /*benefit=*/1);
  patterns.add<vectorIndexStoreLowering>(context, /*benefit=*/1);

  ConversionTarget target(*context);
  target.addIllegalDialect<mlir::AffineDialect, scf::SCFDialect>();
  target.addIllegalOp<memref::LoadOp, memref::StoreOp>();
  target.addIllegalOp<vector::TransferReadOp, vector::TransferWriteOp>();
  target.addIllegalOp<dataflow::VectorIndexLoadOp,
                      dataflow::VectorIndexStoreOp>();
  target
      .addLegalDialect<arith::ArithDialect, heteacc::dataflow::DataFlowDialect,
                       vector::VectorDialect>();
  if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
    return false;
  }

  // Visit all op.
  LLVM_DEBUG(llvm::dbgs() << "\nVisit all op. \n ");
  func.walk([&](Operation *op) {
    if (!isa<func::FuncOp>(op)) {
      this->dispatchVisitor(op);
    }
  });

  this->buildLoopGraph(func);
  this->dependencyAnalyze(func);

  this->connectingBranch(func);
  this->dependency_graph->connectingGraph(func);

  // // Printing the graph

  // func.walk([&] (func::ReturnOp returnop){
  //   this->dependency_graph->setOutputNode(this->map_op_node[returnop.getOperation()]);
  // });
  // if(){
  //   this->dependency_graph->setOutputNode(this->map_op_node[returnop.getOperation()]);
  // }
  LLVM_DEBUG(llvm::dbgs() << "dumpGraph\n";);
  this->dependency_graph->dumpGraph(PrintType::Scala);

  this->dependency_graph->dumpECDFG();

  return true;
}

namespace {
struct GraphInit : public GraphInitBase<GraphInit> {
  GraphInit() = default;
  GraphInit(std::string hlsTopFunc) {
    topFunc = hlsTopFunc;
    ;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto isTop = func.getName() == topFunc;
    std::error_code errc;
    if (isTop) {
      std::string generator = "./hardware/src/main/scala/generator";
      std::string str = std::string(func.getOperationName().data(),
                                    func.getOperationName().size());
      llvm::raw_fd_ostream out("./output/" + topFunc + ".scala", errc,
                               llvm::sys::fs::OpenFlags::OF_None);
      GraphGen graphGen(NodeInfo(0, topFunc), out);
      graphGen.applyGraphInit(func, isTop);
    }
  }
};
} // namespace

std::unique_ptr<Pass> heteacc::createGraphInitPass(std::string hlsTopFunc) {
  return std::make_unique<GraphInit>(hlsTopFunc);
}
