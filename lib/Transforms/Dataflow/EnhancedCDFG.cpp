#include "mlir/Transforms/DialectConversion.h"
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
struct RefineFunc : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcop,
                                PatternRewriter &rewriter) const override { 
  //TODO Refine the EXEblock.
  for(auto &op : funcop.getBody().front())
    if(isa<dataflow::ForOp>(op)){ //TODO Support more op.
      auto loc = funcop.getLoc();//rewriter.setInsertionPoint(&op);
      Value loopSignal = rewriter.create<arith::CmpIOp>( 
        loc, arith::CmpIPredicate::eq, 
        rewriter.create<arith::ConstantIndexOp>(loc, 0), //TODO index/int
        dyn_cast<dataflow::ForOp>(op).getLowerBound());
      Value enable = rewriter.create<dataflow::EnableOp>(
            loc, loopSignal.getType(), loopSignal);   
      return success();
    }
  }
};
} // namespace

struct EnhancedCDFG
    : public EnhancedCDFGBase<EnhancedCDFG> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();


    // executionBlock(&func.front());

    func.walk([&](dataflow::ForOp forop) {
      executionBlock(&forop.getLoopBody().front());
    });


    uint32_t count = 0;
    mlir::Operation* countOp;

    //TODO: Add a execution block for more operations.
    func.walk([&](dataflow::TaskOp taskop) {
      for(auto &op : taskop.getBody().front()){
        countOp = &op;
        if(isa<dataflow::ForOp, AffineForOp, scf::IfOp, dataflow::IfOp>(op))
          break;
        else if(isa<arith::ConstantIntOp, arith::ConstantOp, arith::ConstantIndexOp, memref::AllocaOp, arith::IndexCastOp>(op))
          continue;
        else {
          count += 1;
          countOp = taskop->getParentOp();
        }
      }
    });

    mlir::Operation* control_signal;
    if(count == 0){
        if(isa<dataflow::ForOp>(countOp)){
          // The bounds are constants omitting the arith::cmp.
          // If it is not then a cmp is built, thus supporting dynamic bounds.(whileop)
          OpBuilder builder(countOp);
          if(dyn_cast<dataflow::ForOp>(countOp).getLowerBound().getType().isIndex()){
          auto true_signal = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
          auto loop_stast = StringAttr::get(countOp->getContext(), "loop_start");
          auto null = StringAttr::get(countOp->getContext(), "null");
          builder.setInsertionPoint(countOp);
          auto state = builder.create<dataflow::StateOp>(builder.getUnknownLoc(), true_signal , loop_stast, null);
          state->setAttr("Enable", StringAttr::get(builder.getContext(), "Loop_Start"));
          // auto enable = builder.create<dataflow::EnableOp>(
          //     builder.getUnknownLoc(), true_signal.getType(), true_signal);  
          //     enable->setAttr("Enable", StringAttr::get(builder.getContext(), "Loop_Start"));
            // control_signal = enable.getOperation();
          } else {
            //TODO Supporting  whileop.

            // Value loopSignal = builder.create<arith::CmpIOp>( 
            // builder.getUnknownLoc(), arith::CmpIPredicate::slt, 
            // dyn_cast<dataflow::ForOp>(countOp).getLowerBound(), dyn_cast<dataflow::ForOp>(countOp).getUpperBound());
            // auto enable = builder.create<dataflow::EnableOp>(
            //   builder.getUnknownLoc(), loopSignal.getType(), loopSignal);  
            //   enable->setAttr("Enable", StringAttr::get(builder.getContext(), "Loop_Start"));
            // control_signal = enable.getOperation();
          }
        }

    } else {
      OpBuilder builder(countOp);
      auto true_signal = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
      auto loop_stast = StringAttr::get(countOp->getContext(), "loop_start");
      auto null = StringAttr::get(countOp->getContext(), "null");
      builder.setInsertionPointToStart(&countOp->getRegion(0).front());
      auto state = builder.create<dataflow::StateOp>(builder.getUnknownLoc(), true_signal , loop_stast, null);
      state->setAttr("Enable", StringAttr::get(builder.getContext(), "Loop_Start"));
      //TODO
    }
    
    llvm::SmallVector<Value, 8> carry_vec;
    llvm::SmallVector<mlir::Operation* , 8> carry_new_op;
    llvm::DenseMap<Value, mlir::Operation* > carry2select;
    func.walk([&](dataflow::ForOp forop) {
      for(auto& opiter: forop.getRegion().front()){
        if(auto exeop = dyn_cast<dataflow::ExecutionBlockOp>(opiter)){
          OpBuilder builder(exeop);

          builder.setInsertionPointToStart(&exeop.getBody().front());
          Value ivSel;
          auto isInBlock = [&](OpOperand &use) {
            return exeop->isAncestor(use.getOwner());
          };
          //Supporting Carry Value.
          if(isa<dataflow::ForOp>(exeop->getParentOp())){
            int i = 3;
            
            // for(const auto &carry: forop.getRegionIterArgs()){
            //   auto carry_select = builder.create<dataflow::MergeOp>(builder.getUnknownLoc(), 
            //   carry.getType(), forop.getOperation()->getOperand(i++), carry);
            //   carry_vec.push_back(carry);
            //   carry2select[carry] = carry_select;
            //   carry_new_op.push_back(carry_select);
            //   carry_select->setAttr("Select", StringAttr::get(builder.getContext(), "Loop_Signal"));
            // }

          } else {
            // ivSel = builder.create<dataflow::MergeOp>(builder.getUnknownLoc(), 
            // forop.getInductionVar().getType(), forop.getLowerBound(), forop.getInductionVar());
            // ivSel.getDefiningOp()->setAttr("Select", StringAttr::get(builder.getContext(), "Loop_Signal"));//Loop_Level
          }
          
          builder.setInsertionPoint(exeop.getBody().front().getTerminator());
          // Fixed: Added explicit increment for the loop counter.
          auto ivnew =  builder.create<arith::AddIOp>( builder.getUnknownLoc(), forop.getInductionVar(), forop.getStep());
          
          Value loopSignal = builder.create<arith::CmpIOp>( 
                builder.getUnknownLoc(), arith::CmpIPredicate::eq, ivnew, forop.getUpperBound());
          ivnew->setAttr("Exe", StringAttr::get(builder.getContext(), "Loop"));//Loop_Level
          loopSignal.getDefiningOp()->setAttr("Exe", StringAttr::get(builder.getContext(), "Loop"));//Loop_Level      
          auto loop_back = StringAttr::get(forop.getContext(), "loop_back");
          auto loop_exit = StringAttr::get(forop.getContext(), "loop_exit");
          auto state = builder.create<dataflow::StateOp>(builder.getUnknownLoc(), loopSignal , loop_exit, loop_back);
          state->setAttr("Exe", StringAttr::get(builder.getContext(), "Loop"));//Loop_Level
          forop.getRegion().front().getTerminator()->replaceUsesOfWith(forop.getInductionVar(), ivnew);
          return WalkResult::advance(); 
        }
      }
    });

    func.walk([&](mlir::Operation* op) {
      for(auto &carry_value: carry2select){ 
        // for(auto &carry_op: carry_new_op){ 
          if(op != carry_value.getSecond()){// && !dyn_cast<dataflow::AddressOp>(op)
            op->replaceUsesOfWith(carry_value.getFirst(), carry_value.getSecond()->getResult(0));
          // }
        }
      }
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
