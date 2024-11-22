#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "heteacc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/GraphGen.h"
#include "heteacc/Graph/Node.h"
#include "heteacc/Graph/Graph.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

using namespace mlir;
using namespace heteacc;
#define DEBUG_TYPE "graph"

void GraphGen::buildEnhancedControlDataFlowGraph(func::FuncOp func){
  uint32_t c_id = 0;
}

bool GraphGen::applyGraphInit(func::FuncOp func, bool isTopFunc) {
    
    
    if(isTopFunc){
        this->dependency_graph->top_function = func;
        for(Value operand : func.front().getArguments()){ 
          // auto node = this->dependency_graph->getArgCall()->insertLiveInArgument(operand, ArgumentNode::LiveIn);
          this->dependency_graph->funArgValue.push_back(operand);
          this->map_value_node[operand] =  this->dependency_graph->getArgCall()->insertLiveInArgument(operand, ArgumentNode::ArgumentType::LiveIn);
          //arg_mem0
        }
        //TODO Global Value.
    }


    //Visit all op. 
    LLVM_DEBUG(llvm::dbgs() << "\nVisit all op. \n ");
    func.walk([&](Operation* op) {  
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
    topFunc = hlsTopFunc;;
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto isTop = func.getName() == topFunc;
    std::error_code errc;
    if(isTop) {
      std::string generator = "./hardware/src/main/scala/generator";
      std::string str = std::string(func.getOperationName().data(), func.getOperationName().size());
      llvm::raw_fd_ostream out("./output/" + topFunc + ".scala", errc, llvm::sys::fs::OpenFlags::OF_None);
      GraphGen graphGen(NodeInfo(0, topFunc), out);
      graphGen.applyGraphInit(func, isTop);

    }
    


  }
};
} // namespace

std::unique_ptr<Pass>
heteacc::createGraphInitPass(std::string hlsTopFunc) {
  return std::make_unique<GraphInit>(hlsTopFunc);
}
