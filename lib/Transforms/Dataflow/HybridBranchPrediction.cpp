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
struct SelectPrediction : public OpRewritePattern<dataflow::IfOp> {
  using OpRewritePattern<dataflow::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dataflow::IfOp ifop,
                                PatternRewriter &rewriter) const override { 
  
  //The IF-THEN-ELSE region is in one of the execution regions, 
  //and this execution region enables the IF Dual Issue region.

  bool hasElseRegion = !ifop.getElseRegion().empty();
  
  if(hasElseRegion){
    //The IF-ELSE region simply returns the value that has not been updated, 
    //optimizing the IF-THEN-ELSE structure for dataflow.select.
    if(ifop.getElseRegion().front().getOperations().size() == 1){
      auto &block = ifop.getThenRegion().front();
      for (auto &op : llvm::make_early_inc_range(block)) {
        if(&op == block.getTerminator()){
          continue;
        } else {
          op.moveBefore(ifop);
        }
      }

      Value conditionValue = ifop.getCondition();
      auto condop = conditionValue.getDefiningOp();
      // condop->dump();
      rewriter.setInsertionPointAfter(ifop);
      
      //for(int i=0; i < ifop.getThenRegion().front().getTerminator().getNumOperands(); i++){
      
      //Handles the case of having more than one carry variable and yield value.
      auto ifSel = rewriter.create<dataflow::SelectOp>(rewriter.getUnknownLoc(), 
            ifop.getResults().getType(), condop->getResult(0), //lowerBound, lowerBound);
            ifop.getThenRegion().front().getTerminator()->getOperand(0), 
            ifop.getElseRegion().front().getTerminator()->getOperand(0));//如果有多个，就循环复制
            

      rewriter.replaceOp(ifop, ifSel.getResult());
    } else {
      //TODO

    }

  } 


  return success();
  }
};
} // namespace


namespace {
struct BranchPrediction : public OpRewritePattern<dataflow::IfOp> {
  using OpRewritePattern<dataflow::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dataflow::IfOp ifop,
                                PatternRewriter &rewriter) const override { 
  int ifopNum = 0;
  //Only one ifop.
  mlir::Operation* topOp;
  int branchStauts = 0;
  llvm::SmallVector<mlir::Operation*, 6> branchOps;
  llvm::DenseMap<mlir::Operation*, llvm::SmallSetVector<mlir::Block *, 4>> branch2block;
  llvm::DenseMap<int , mlir::Block *> order2block;

  
  return success();
  }
};
} // namespace


struct HybridBranchPrediction
    : public HybridBranchPredictionBase<HybridBranchPrediction> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();


    mlir::RewritePatternSet patterns(context);
    //Folding simple branching statements.
    patterns.add<SelectPrediction>(context);
    // patterns.add<BranchPrediction>(context);



    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

    //We need to determine how many branches states.
    func->walk([&](dataflow::ExecutionBlockOp exeop){
      applyBranchPrediction(exeop);
    });

    // mlir::RewritePatternSet new_patterns(context);
    // new_patterns.add<BranchPrediction>(context);
    // (void)applyPatternsAndFoldGreedily(func, std::move(new_patterns));

    //FIXME walker
    func->walk([&](dataflow::IfOp ifop){
      //Creating the issue block.
      // if(ifop.getResults().size() == 0){
      //   return WalkResult::interrupt();
      // }
      OpBuilder builder(ifop);
      builder.setInsertionPointAfter(ifop);
      executionBlock(&ifop.getThenRegion().front());
      if(!ifop.getElseRegion().empty()){
        executionBlock(&ifop.getElseRegion().front());
        // for(int i=0; i < ifop.getResults().size(); i++){
        //   auto value = ifop.getThenRegion().front().getTerminator()->getOperand(i);
        //   auto data_select = builder.create<dataflow::SelectOp>(builder.getUnknownLoc(), 
        //   value.getType(), ifop.getCondition(), value, ifop.getElseRegion().front().getTerminator()->getOperand(i));
        //   data_select->setAttr("Select", StringAttr::get(builder.getContext(), "Data"));
        // }
        llvm::SmallVector <Value, 8> yieldValue;
        for(const auto &result : ifop.getResults()){
          auto data_select = builder.create<dataflow::SelectOp>(builder.getUnknownLoc(), 
          result.getType(), ifop.getCondition(), result, result);
          data_select->setAttr("Select", StringAttr::get(builder.getContext(), "Data"));
          data_select->setAttr("Data", StringAttr::get(builder.getContext(), "IF-THEN-ELSE"));
          yieldValue.push_back(data_select.getResult());
        }
        ifop.getOperation()->getBlock()->getTerminator()->setOperands(ValueRange{yieldValue});
        } else {
        //?
        int i = 3;
        for(const auto &result : ifop.getResults()){
           auto data_select = builder.create<dataflow::SelectOp>(builder.getUnknownLoc(), 
            result.getType(), ifop.getCondition(), result, ifop->getParentOp()->getOperand(i++));
            data_select->setAttr("Select", StringAttr::get(builder.getContext(), "Data"));
        }
        auto if_then = StringAttr::get(ifop->getContext(), "if_then");
        auto if_else_null = StringAttr::get(ifop->getContext(), "null");
        builder.setInsertionPoint(ifop);
        builder.create<dataflow::StateOp>(builder.getUnknownLoc(), ifop.getCondition(), if_then, if_else_null);
       
      }


      return WalkResult::advance();
    });
    
  }
};






std::unique_ptr<Pass> heteacc::createHybridBranchPredictionPass() {
  return std::make_unique<HybridBranchPrediction>();
}
