#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "heteacc/Dialect/DataFlow/Utils.h"
#include "mlir/Analysis/Liveness.h"

using namespace mlir;
using namespace heteacc;
using namespace dataflow;



/// Wrap the operations in the block with launch op.
LaunchOp heteacc::launchBlock(Block *block) {
  if (!block->getOps<LaunchOp>().empty() ||
      !isa<func::FuncOp, mlir::AffineForOp>(block->getParentOp()))
    return LaunchOp();

  OpBuilder builder(block, block->begin());
  ValueRange returnValues(block->getTerminator()->getOperands());
  TypeRange  returnType(block->getTerminator()->getOperandTypes());
  auto loc = builder.getUnknownLoc();
  
  auto launch = builder.create<LaunchOp>(loc,  returnValues);
  auto &launchBlock = launch.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&launchBlock);
  builder.create<YieldOp>(loc, returnValues);

  auto &LaunchOps = launchBlock.getOperations();
  auto &parentOps = block->getOperations();
  LaunchOps.splice(launchBlock.begin(), parentOps,
                     std::next(parentOps.begin()), std::prev(parentOps.end()));
  block->getTerminator()->setOperands(launch.getResults());
  return launch;
}

// Reference to Scalehls.
TaskOp heteacc::fuseOpsIntoTask(ArrayRef<Operation *> ops,
                                 PatternRewriter &rewriter,
                                 bool insertToLastOp) {
  assert(!ops.empty() && "must fuse at least one op");
  llvm::SmallDenseSet<Operation *, 4> opsSet(ops.begin(), ops.end());

  // Collect output values. This is not sufficient and may lead to empty-used
  // outputs, which will be removed during canonicalization.
  llvm::SetVector<Value> outputValues;
  for (auto op : ops)
    for (auto result : op->getResults())
      if (llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return !opsSet.count(user); }))
        outputValues.insert(result);
        
  // Create new graph task with all inputs and outputs.
  auto loc = rewriter.getUnknownLoc();
  if (!insertToLastOp)
    rewriter.setInsertionPoint(ops.front());
  else
    rewriter.setInsertionPoint(ops.back());
    Type resultType;
  auto task =
      rewriter.create<TaskOp>(loc, ValueRange(outputValues.getArrayRef()));
  auto taskBlock = rewriter.createBlock(&task.getBody());

  // Move each targeted op into the new graph task.
  rewriter.setInsertionPointToEnd(taskBlock);
  auto yield = rewriter.create<YieldOp>(loc, outputValues.getArrayRef());
  for (auto op : ops)
    op->moveBefore(yield);

  // Replace external output uses with the task results.
  unsigned idx = 0;
  for (auto output : outputValues)
    output.replaceUsesWithIf(task.getResult(idx++), [&](OpOperand &use) {
      return !task->isProperAncestor(use.getOwner());
    });

  // Inline all sub-tasks.
  for (auto subTask : llvm::make_early_inc_range(task.getOps<TaskOp>())) {
    auto &subTaskOps = subTask.getBody().front().getOperations();
    auto &taskOps = task.getBody().front().getOperations();
    taskOps.splice(subTask->getIterator(), subTaskOps, subTaskOps.begin(),
                   std::prev(subTaskOps.end()));
    rewriter.replaceOp(subTask, subTask.getYieldOp()->getOperands());
  }
  return task;
}


ExecutionBlockOp heteacc::executionBlock(Block *block){
  
  if (!block->getOps<ExecutionBlockOp>().empty() ||
      !isa<dataflow::ForOp, mlir::AffineForOp, dataflow::IfOp>(block->getParentOp()))
    return ExecutionBlockOp();
  llvm::SmallVector<Value, 8> inputs;
  llvm::SmallVector<Location, 8> inputLocs;
  llvm::SmallVector<Type, 8> inputsType;
  auto liveins = Liveness(block->getParentOp()).getLiveIn(block);
  for (auto livein : liveins) {
    if (block->getParent()->isAncestor(livein.getParentRegion()))
      continue;
    inputs.push_back(livein);
    inputLocs.push_back(livein.getLoc());
    inputsType.push_back(livein.getType());;
  }

  auto isInDispatch = [&](OpOperand &use) {
      return block->getParentOp()->isAncestor(use.getOwner());
  };

  OpBuilder builder(block, block->begin());
  auto loc = builder.getUnknownLoc();
  ValueRange returnValues(block->getTerminator()->getOperands());
  TypeRange typeValues;


  auto exeOp = builder.create<ExecutionBlockOp>(loc, returnValues);//, returnValues);

  OpBuilder b(exeOp);
  auto UnknownLoc = b.getUnknownLoc();

  auto &exeBlock = exeOp.getBody().emplaceBlock();
  b.setInsertionPointToStart(&exeBlock);
  auto exeReturnse =b.create<arith::ConstantIntOp>(loc, 1, 1);


  builder.setInsertionPointToEnd(&exeBlock);
  auto yield = builder.create<YieldOp>(loc, returnValues);
  yield->setAttr("execution_block",builder.getI32IntegerAttr(1));

  auto &exeOps = exeBlock.getOperations();
  auto &parentOps = block->getOperations();
  exeOps.splice(exeBlock.begin(), parentOps,
                     std::next(parentOps.begin()), std::prev(parentOps.end()));
         
  
  block->getTerminator()->setOperands(exeOp.getResults());
  return exeOp;
}

void heteacc::applyBranchPrediction(dataflow::ExecutionBlockOp exeop) {
  int ifopNum = 0;
  //Only one ifop.
  mlir::Operation* topOp;
  int branchStauts = 0;
  llvm::SmallVector<mlir::Operation*, 6> branchOps;
  llvm::DenseMap<mlir::Operation*, llvm::SmallSetVector<mlir::Block *, 4>> branch2block;
  llvm::DenseMap<int , mlir::Block *> order2block;


  for(auto &op : exeop.getBody().front()){
    //the top ifop.
    if(isa<dataflow::IfOp>(op)){

      bool hasElseRegion = !dyn_cast<dataflow::IfOp>(op).getElseRegion().empty();

      if(!hasElseRegion){
        break;
      }

      // getBranchStatus(&op);
      topOp = &op;
      ifopNum++;

      branchStauts += 1;
      branchOps.push_back(&op);
      order2block[branchStauts] = &dyn_cast<dataflow::IfOp>(op).getThenRegion().front();
      branch2block[&op].insert(&dyn_cast<dataflow::IfOp>(op).getThenRegion().front());
      //TODO: Using a better way to capture branch states and blocks.
      dyn_cast<dataflow::IfOp>(op).getElseRegion().front().walk([&] (dataflow::IfOp ifop){
        branchStauts += 1;
        
        branchOps.push_back(ifop.getOperation());
        branch2block[ifop.getOperation()].insert(&ifop.getThenRegion().front());
        order2block[branchStauts] = &ifop.getThenRegion().front();

        if(!ifop.getElseRegion().empty()){
          branchStauts += 1;
          order2block[branchStauts] = &ifop.getElseRegion().front();
          branch2block[ifop.getOperation()].insert(&ifop.getElseRegion().front());
        }

      });   
    } 
  }
  int blocksNum = 0;
  for(auto &b2b: branch2block){
    blocksNum += b2b.second.size();
    // for (auto &op : llvm::make_early_inc_range(block)) {
    //   if(&op == block.getTerminator()){
    //     continue;
    //   } else {
    //     op.moveBefore(ifop);
    //   }
    // }

  }
  if(ifopNum == 0)
    return ;
  assert(ifopNum == 1 && blocksNum == branchStauts);
// Value conditionValue = ifop.getCondition();
//       auto condop = conditionValue.getDefiningOp();

  OpBuilder builder(topOp);
  auto if_then = StringAttr::get(topOp->getContext(), "if_then");
  auto if_else = StringAttr::get(topOp->getContext(), "if_else");
  auto if_else_null = StringAttr::get(topOp->getContext(), "null");
  builder.setInsertionPoint(topOp);
  builder.create<dataflow::StateOp>(builder.getUnknownLoc(), dyn_cast<dataflow::IfOp>(topOp).getCondition(), if_then, if_else);


  for(auto &b2b: branch2block){
    if(b2b.first == topOp)
      continue;
    builder.setInsertionPoint(b2b.first);
    if(b2b.second.size() > 1){
      builder.create<dataflow::StateOp>(builder.getUnknownLoc(), dyn_cast<dataflow::IfOp>(b2b.first).getCondition(), if_then, if_else);
    } else {
      builder.create<dataflow::StateOp>(builder.getUnknownLoc(), dyn_cast<dataflow::IfOp>(b2b.first).getCondition(), if_then, if_else_null);
    }
  }
}