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
#include "mlir/Analysis/Liveness.h"
using namespace mlir;
using namespace heteacc;
#define DEBUG_TYPE "graph"


LoopInfo GraphGen::analyzeLoopNode(dataflow::ForOp dataflowForop){
    LLVM_DEBUG(llvm::dbgs() << "\nAnalyze Loop Node \n ");
    LoopInfo Info("loop_" + std::to_string(this->dependency_graph->getLoopNodeNums()));
    //check op is a special form.

    if (auto loopBandAttr = dataflowForop->getAttr("Loop_Band")) {
      Info.loop_band = loopBandAttr.cast<IntegerAttr>().getInt();
    }
    if (auto loopLevelAttr = dataflowForop->getAttr("Loop_Level")) {
      Info.loop_level = loopLevelAttr.cast<IntegerAttr>().getInt();
    }
    if(Info.loop_band == 999 && Info.loop_level == 999)
      Info.loop_structure = false;
    LLVM_DEBUG(llvm::dbgs() << "Loop_Level: "<< Info.loop_level << "\n");

    Value forInduction = dataflowForop.getInductionVar();

    // It is the outermost loop.

    //If there are no other operations in front of the loop, 
    //besides constant and alloca, etc., then it is enable. 

    //FIXME
    dataflowForop->getParentOp()->getParentRegion()->front().walk([&](mlir::Operation* outer){
      if(auto state = dyn_cast<dataflow::StateOp>(outer)){
        if(state.getInstanceName().str().find("loop_start") != std::string::npos){
          Info.enable = outer;
          outer->dump();
          blacklist_control_edge[outer].push_back(dataflowForop.getOperation());
          return WalkResult::interrupt();
        }
      } 
    });
    if(Info.enable == nullptr){
      Info.enable = dataflowForop.getOperation()->getParentOp();
      blacklist_control_edge[dataflowForop.getOperation()->getParentOp()].push_back(dataflowForop.getOperation());
    }
    for(auto &op: dataflowForop.getRegion().front()){
      if(dyn_cast<dataflow::ExecutionBlockOp>(op)){
        Info.exe_block_op = &op;
      }
    }
    for(auto &for_exe_op: Info.exe_block_op->getRegion(0).front()){//dataflowForop.getRegion().front().walk([&](mlir::Operation* for_exe_op){
      if(auto state = dyn_cast<dataflow::StateOp>(for_exe_op)){
          {
            if(state.getInstanceName().str().find("loop_exit") == std::string::npos)
              continue;
              // return WalkResult::advance();
            Info.loop_back = &for_exe_op;
            Info.loop_finish = &for_exe_op;
            blacklist_control_edge[&for_exe_op].push_back(dataflowForop.getOperation());
            // return WalkResult::interrupt();
            break;
          }
        }  

    // });
    }
    // Info.enable->dump();
    // Info.loop_back->dump();
    // llvm::outs() <<  "\n" << "End !!!! " << "\n";

    llvm::SmallVector<Value, 8> carry_value;
    carry_value.push_back(forInduction);
    auto regionIterArgs = dataflowForop.getRegionIterArgs();
    for (auto arg : regionIterArgs) {
        carry_value.push_back(arg);
    }
    // Analyze all the live-ins.
    this->op2traversal[dataflowForop.getOperation()] = true;

    // auto liveins = Liveness(dataflowForop.getOperation()).getLiveIn(&dataflowForop.getRegion().front());
    // llvm::DenseMap<Value, llvm::SmallVector<mlir::Operation*, 8>> livein2op;
    // for (auto livein : liveins) {
    //    llvm::outs() << livein << " <-livein \n";
    //   for(auto &user: livein.getUses()){
    //       livein2op[livein].push_back(user.getOwner());//->dump();
    //       user.getOwner()->dump();
    //   }
    // }
    
    dataflowForop.getOperation()->walk([&] (mlir::Operation* region_op){
      if(isa<dataflow::ExecutionBlockOp>(region_op) || this->op2traversal.find(region_op) != this->op2traversal.end()) 
        return WalkResult::advance(); //dataflow.state

      // region_op->dump();
      this->op2traversal[region_op] = true;

      for(const auto &operand : region_op->getOperands()){
        LLVM_DEBUG(llvm::dbgs() << operand << "\n";);

        if (auto selectAttr = region_op->getAttrOfType<StringAttr>("Select")) {
          // Retrieve the string value of the attribute
          llvm::StringRef selectValue = selectAttr.getValue();
          // Check if the value is "Loop"
          if (selectValue.find("Loop_Signal") != std::string::npos) {
              Info.selectVec.push_back(region_op);
              return WalkResult::advance(); 
          }
        }

        // It is an argument from the function, not the operation.
        // Notes that carry values are not included.
        auto isBlockArgOrNotAllocaOp = [&](const mlir::Value &VAL) -> bool {
          auto iter = std::find(this->dependency_graph->funArgValue.begin(), this->dependency_graph->funArgValue.end(), VAL);
          auto carry_iter = std::find(carry_value.begin(), carry_value.end(), VAL);
          return VAL.isa<mlir::BlockArgument>() && iter != this->dependency_graph->funArgValue.end();
        };

        bool is_live_in = isBlockArgOrNotAllocaOp(operand);//It is a block arg.

        if(auto t = operand.getDefiningOp()){
          //Data dependencies come from outside and are not special operations.
          if(!dataflowForop->isAncestor(t) && !isa<arith::ConstantOp>(t)){
            is_live_in = true;
          }
        }
        
        // Find the live-in value.
        if(is_live_in){
          LLVM_DEBUG(llvm::dbgs() << "Find the live-in value.\n ";);
          bool loop2loop = false;
          // auto iter = std::find(livein2op.begin(), this->dependency_graph->funArgValue.end(), operand);
          // llvm::SmallVector<mlir::Operation*, 8> livein2op_set = livein2op[operand];
          // if(livein2op_set)
          // llvm::outs() << "xxx" << "\n";
          // for(auto &op : livein2op_set){
          //   op->dump();
          // }
          //Here we have not considered memory dependencies between loops. 
          dataflowForop.getOperation()->walk([&](dataflow::ForOp sub_loop){
            if(dataflowForop == sub_loop) return WalkResult::advance();
            // llvm::outs() << sub_loop << "   subloop \n";

            // sub_loop.getOperation()->walk([&](mlir::Operation* sub_loop_op ){
            for(auto &sub_loop_op : sub_loop.getLoopBody().front()){

              for(const auto &sub_loop_operand : sub_loop_op.getOperands()){

                if(&operand == &sub_loop_operand){
                  loop2loop = true;
                  LLVM_DEBUG(llvm::dbgs() << "Find the live-in value to loop1. \n ";);

            //       mlir::Value* nonConstOperand = const_cast<mlir::Value*>(operand);
            // nonConstOperand->dump();
                  LLVM_DEBUG(llvm::dbgs() << "operand: " << operand << "\n ";);
                  // operand.dump();
                  Info.live_in_out_loop[operand].insert(sub_loop);
                  // sub_loop_op.dump();
                  // break;
                }
                if(isa<mlir::BlockArgument>(sub_loop_operand)){
                  LLVM_DEBUG(llvm::dbgs() << "Find the live-in value to loop2.   " << operand  << "   \n";);
                  Info.live_in_out_loop[operand].insert(sub_loop);
                  // sub_loop_op.dump();
                }
              }
            }
            return WalkResult::advance();
          });

          Info.live_in_out_ins[operand].insert(region_op);
          // %12 = memref.load %arg0[%arg3, %arg5] : memref<32x32xi32>
          // memref<32x32xi32>: 0; index: 0

          //Has a parent loop.
          // if(Info.loop_level != 0){
          //   // if (!isa<mlir::BlockArgument>(operand)) {
          //     if(isBlockArgOrNotAllocaOp(operand)){
          //       Info.live_in_in_ins.insert(operand);
          //     }
          //     else {
          //       dataflowForop->getParentOp()->walk([&](dataflow::ForOp parent_loop){
          //       // if(&dataflowForop == &parent_loop) return WalkResult::advance();
          //       for(auto &parent_loop_op : parent_loop.getLoopBody().front()){
          //         for(const auto &par_operand : parent_loop_op.getOperands()){
          //           if(&par_operand == &operand)
          //             Info.live_in_in_ins.insert(operand);
          //         }
          //       }
          //       return WalkResult::interrupt();
          //     });
          //     }
          //   // }

          // } else {
          //   Info.live_in_in_ins.insert(operand);
          // }
          Info.live_in_in_ins.insert(operand);
          // Adding to data edge blacklist
          this->blacklist_loop_live_in_data_edge[operand].push_back(region_op);
        }
      }
    });
    
    // for(auto iter = op2traversal.begin(); iter != op2traversal.end(); ++iter){
    //   iter->second = false;
    // }
    // this->op2traversal[dataflowForop.getOperation()] = true;
    // dataflowForop.getOperation()->walk([&] (mlir::Operation* region_op){

    //   if(isa<dataflow::ExecutionBlockOp, arith::ConstantOp>(region_op) || this->op2traversal[region_op]) 
    //     return WalkResult::advance(); //dataflow.state

    //   llvm::outs() <<  "\\output: \n";
    //   region_op->dump();
    //   this->op2traversal[region_op] = true;

    //   for(const auto &result : region_op->getResults()){
    //     bool is_live_out = false;
    //   //   for(auto &user : result.getUses()){
    //   //     user.getOwner()->dump();

    //   //     is_live_out = !dataflowForop->isAncestor(user.getOwner()) ? true : false;
    //   //     if(is_live_out){

    //   //     }
    //   //   }
    //   }
      
    // });
    if(dataflowForop.getOperation()->getNumResults() > 0){
      auto forYield = dataflowForop.getLoopBody().front().getTerminator()->getOperand(0);
      
      if(auto exeop = dyn_cast<dataflow::ExecutionBlockOp>(forYield.getDefiningOp())){
        auto exeYield = exeop.getBody().front().getTerminator()->getOperand(0);

        bool loop2loop = false;
        dataflowForop.getOperation()->walk([&](dataflow::ForOp sub_loop){
          if(dataflowForop == sub_loop) return WalkResult::advance();
        
          for(auto &sub_loop_op : sub_loop.getLoopBody().front()){

            for(const auto &sub_loop_operand : sub_loop_op.getOperands()){
              if(&exeYield == &sub_loop_operand){
                loop2loop = true;

                Info.live_out_in_loop[exeYield].insert(sub_loop);

              }
              if(isa<mlir::BlockArgument>(sub_loop_operand)){
                Info.live_in_out_loop[exeYield].insert(sub_loop);
              }
            }
          }
          return WalkResult::advance();
        });
        Info.live_out_in_ins.insert(exeYield);
        // this->blacklist_loop_live_out_data_edge[exeYield].push_back(this->dependency_graph->returnOp);

        // Info.live_out_out_ins[exeYield].insert(use.getOwner());
        // Info.live_out_out_loop[exeYield].insert(dataflowForop);
        for(const auto &use: forYield.getDefiningOp()->getParentOp()->getResults()){
          // llvm::outs() <<  "\\output: \n";
          // use.getDefiningOp()->dump();
          // Info.live_out_in_ins.insert(use);
          if(dyn_cast<dataflow::TaskOp>(use.getDefiningOp()->getParentOp())){
            auto outermostValue = use.getDefiningOp()->getParentOp()->getParentOp()->getResult(0);
            LLVM_DEBUG(llvm::dbgs() << "outermostValue: "<< outermostValue << "\n";);
            for(auto &user: outermostValue.getUses()){
              Info.live_out_out_ins[outermostValue].insert(user.getOwner());
              this->blacklist_loop_live_out_data_edge[outermostValue].push_back(user.getOwner());
            
            // user.getOwner()->dump();
            }
            
          } else {
            //TODO
            for(auto &forResultUser: use.getUses()){
              forResultUser.getOwner()->dump();
              if(isa<dataflow::ForOp>(forResultUser.getOwner())){
                continue;
              }
              Info.live_out_out_ins[use].insert(forResultUser.getOwner());
              this->blacklist_loop_live_out_data_edge[forInduction].push_back(forResultUser.getOwner());
            }

            // Info.live_out_out_ins[use].insert(use.getDefiningOp());
            // this->blacklist_loop_live_out_data_edge[use].push_back(use.getDefiningOp());
          }
          
        }
        
      }
    }

    //Carry loop dependency
    for(auto &use: forInduction.getUses()){
      // use.getOwner()->dump();
      if(isa<dataflow::ForOp>(use.getOwner())){
        continue;
      }
      Info.carry_dependencies[forInduction].push_back(use.getOwner());
      this->blacklist_carry_dependency_data_edge[forInduction].push_back(use.getOwner());
    }
    // for(const auto &operand: Info.exe_block_op->getRegion(0).front().getTerminator()->getOperands()){
    //   // use.getOwner()->dump();
    //   // if(isa<dataflow::ForOp>(use.getOwner())){
    //   //   continue;
    //   // }
    //   Info.carry_dependencies[operand].push_back(operand.getDefiningOp());
    //   this->blacklist_carry_dependency_data_edge[operand].push_back(operand.getDefiningOp());
    // }

   
    int carry_index =0;
    for(const auto &iter_value : dataflowForop.getRegionIterArgs()){
      
      for(auto &use: iter_value.getUses()){
        // if(isa<dataflow::ForOp>(use.getOwner())){
        //   continue;
        // }
        auto yield_op_value = Info.exe_block_op->getRegion(0).front().getTerminator()->getOperand(carry_index);

        Info.carry_dependencies[iter_value].push_back(use.getOwner());
        Info.carry_dependencies[iter_value].push_back(yield_op_value.getDefiningOp());
        Info.carry_dependencies_without_connection[iter_value].push_back(yield_op_value.getDefiningOp());

        Info.carry_dependencies_map[use.getOwner()] = iter_value;
        Info.carry_dependencies_map[yield_op_value.getDefiningOp()] = iter_value;

        this->blacklist_carry_dependency_data_edge[iter_value].push_back(use.getOwner());
        // this->blacklist_carry_dependency_data_edge[iter_value].push_back(yield_op_value.getDefiningOp());
      }
      carry_index += 1;
    }

    if(false){
      LLVM_DEBUG(llvm::dbgs() << "live_in_out_ins:\n");
      for (auto &pair : Info.live_in_out_ins) {
          LLVM_DEBUG(llvm::dbgs() << "Value: " << pair.first << ", Operations: ");
          for (auto *op : pair.second)
              LLVM_DEBUG(llvm::dbgs() << op->getName() << " " ; op->dump(););
          LLVM_DEBUG(llvm::dbgs() << "\n");
      }


      LLVM_DEBUG(llvm::dbgs() << "live_in_out_loop:\n");
      for (auto &pair : Info.live_in_out_loop) {
          LLVM_DEBUG(llvm::dbgs() << "Value: " << pair.first << ", Loops: ");
          for (auto loop : pair.second)
              LLVM_DEBUG(llvm::dbgs() << loop->getName() << " " << loop << " \n");
          LLVM_DEBUG(llvm::dbgs() << "\n");
      }

      LLVM_DEBUG(llvm::dbgs() << "live_in_in_ins:\n");
      for (auto value : Info.live_in_in_ins) {
          LLVM_DEBUG(llvm::dbgs() << "Value: " << value << "\n");
      }

      LLVM_DEBUG(llvm::dbgs() << "live_in_in_loop:\n");
      for (auto &pair : Info.live_in_in_loop) {
          LLVM_DEBUG(llvm::dbgs() << "Value: " << pair.first << ", Loops: ");
          for (auto loop : pair.second)
              LLVM_DEBUG(llvm::dbgs() << loop->getName() << " ");
          LLVM_DEBUG(llvm::dbgs() << "\n");
      }

      LLVM_DEBUG(llvm::dbgs() << "carry_dependencies:\n");
      for (auto &pair : Info.carry_dependencies) {
          LLVM_DEBUG(llvm::dbgs() << "Value: " << pair.first << ", Operations: ");
          
          for (auto *op : pair.second){
            if(dyn_cast<func::FuncOp>(op)){
              LLVM_DEBUG(llvm::dbgs() << " is funcop.");
            }
            LLVM_DEBUG(llvm::dbgs() << op->getName() << " ");
          } 
          LLVM_DEBUG(llvm::dbgs() << "\n");
      }
    }


    return Info;
}


void GraphGen::buildLoopGraph(func::FuncOp func){
  uint32_t c_id = 0;
  LLVM_DEBUG(llvm::dbgs() << "\nBuild Loop Graph \n ";);

  func.walk([&](dataflow::ForOp forOp) {  

    //Find the loop node.
    LoopNode* currentLoopNode = this->for_op_node[forOp];
    LoopInfo currLoopInfo = this->for_op_info[forOp];

    if(currentLoopNode == nullptr) assert(!"LoopNode is empty !");

    //enable
    
    // currLoopInfo.enable->dump();
    this->map_op_node[currLoopInfo.enable]->addControlOutputPort(currentLoopNode);
    currentLoopNode->setControlLoop(this->map_op_node[currLoopInfo.enable]);

    
    currentLoopNode->setActiveOutputLoopSignal(this->map_op_node[currLoopInfo.exe_block_op]);
    this->map_op_node[currLoopInfo.exe_block_op]->addControlInputPort(currentLoopNode);
    

    static_cast<StateBranchNode*>(this->map_op_node[currLoopInfo.loop_back])->addFalseBranch(currentLoopNode);
    currentLoopNode->setControlLoop(this->map_op_node[currLoopInfo.loop_back]);
    static_cast<StateBranchNode*>(this->map_op_node[currLoopInfo.loop_back])->addTrueBranch(currentLoopNode);
    currentLoopNode->setControlLoop(this->map_op_node[currLoopInfo.loop_back]);

    auto loop_cmp = this->map_value_node[dyn_cast<dataflow::StateOp>(currLoopInfo.loop_back).getCond()];
    static_cast<StateBranchNode*>(this->map_op_node[currLoopInfo.loop_back])->addDataInputPort(loop_cmp);
    loop_cmp->addDataOutputPort(this->map_op_node[currLoopInfo.loop_back]);
 
    currentLoopNode->setActiveBackSignal(this->map_op_node[currLoopInfo.exe_block_op]);
    this->map_op_node[currLoopInfo.exe_block_op]->addControlInputPort(currentLoopNode);
 
    // Connecting loop exit signals
    if(dyn_cast<dataflow::TaskOp>(forOp->getParentOp())){
      if(!this->dependency_graph->hasreturnValue){

      } else {
        this->map_value_node[this->dependency_graph->returnValue]->addControlInputPort(currentLoopNode);
        currentLoopNode->setActiveExitSignal(this->map_value_node[this->dependency_graph->returnValue]);
      }
    } else {
      ;
    }

    // return WalkResult::advance();
  });

  func.walk([&](dataflow::ForOp forOp) {  

    //Find the loop node.
    LoopNode* currentLoopNode = this->for_op_node[forOp];
    LoopInfo currLoopInfo = this->for_op_info[forOp];

    // Conneting live-in values.
    LLVM_DEBUG(llvm::dbgs() << "Conneting live-in values. \n";);

    for(auto in: currLoopInfo.live_in_in_ins){
      LLVM_DEBUG(llvm::dbgs() << "in: " << in << "\n";);

      auto node = currentLoopNode->findLiveInNode(in);
      for(auto &use: in.getUses()){
        use.getOwner()->dump();
      }
      if(node == nullptr){
        auto new_live_in = currentLoopNode->insertLiveInArgument(in, ArgumentNode::ArgumentType::LoopLiveIn);
        if(this->map_value_node.find(in) == this->map_value_node.end()){
          this->map_value_node[in] = new_live_in;
        }
        new_live_in->setParentNode(this->map_value_node[in]);
      }
      this->live_in_ins_loop_edge[in].insert(forOp);
    }

    for (auto in : currLoopInfo.live_in_out_loop) {
      auto node = currentLoopNode->findLiveInNode(in.getFirst());
      if(node == nullptr){
        auto new_live_in = currentLoopNode->insertLiveInArgument(
            in.getFirst(), ArgumentNode::ArgumentType::LoopLiveIn);
        new_live_in->setParentNode(map_value_node[in.getFirst()]);
      }
      for(auto &loop : in.getSecond()){
        auto _tmp = this->for_op_node[loop]->findLiveInNode(in.getFirst());
        if (_tmp == nullptr) {
          auto new_loop_live_in = this->for_op_node[loop]->insertLiveInArgument(
              in.getFirst(), ArgumentNode::ArgumentType::LoopLiveIn);
          new_loop_live_in->setParentNode(map_value_node[in.getFirst()]);
        }

        this->live_in_loop_loop_edge[in.getFirst()].insert(std::make_pair(forOp, loop));
      }
    }

    for (auto src : currLoopInfo.live_in_out_ins) {
      for (auto tar : src.getSecond()) {
        this->live_in_loop_ins_edge[forOp].insert(std::make_pair(src.getFirst(), tar));
      }
    }
    //FIX different Live out
    // Conneting live-out values.
    LLVM_DEBUG(llvm::dbgs() << "Conneting live-out values. \n";);

    for (auto _live_out : currLoopInfo.live_out_in_ins) {
      auto _new_live_out_node = currentLoopNode->findLiveOutNode(_live_out);
      if (_new_live_out_node == nullptr) {
        _new_live_out_node =
            currentLoopNode->insertLiveOutArgument(_live_out, ArgumentNode::ArgumentType::LoopLiveOut);
        LLVM_DEBUG(llvm::dbgs() << "_new_live_out_node is " <<  _new_live_out_node->getName() << "\n";);
        if(this->map_value_node.find(_live_out) == this->map_value_node.end()){
          this->map_value_node[_live_out] = _new_live_out_node;
        }
        static_cast<ArgumentNode*>(_new_live_out_node)
            ->setParentNode(this->map_value_node[_live_out]);
      }

      this->live_out_ins_loop_edge[_live_out].insert(forOp);
    }
    LLVM_DEBUG(llvm::dbgs() << "Conneting live-out values. Edge type 2 \n";);

    // Edge type 2
    for (auto _live_out : currLoopInfo.live_out_in_loop) {
      auto _node = currentLoopNode->findLiveOutNode(_live_out.getFirst());
      if (_node == nullptr) {
        auto _new_live_out_node = currentLoopNode->insertLiveOutArgument(
            _live_out.getFirst(), ArgumentNode::ArgumentType::LoopLiveOut);
        _new_live_out_node->setParentNode(map_value_node[_live_out.getFirst()]);
        LLVM_DEBUG(llvm::dbgs() << "_new_live_out_node is " <<  _new_live_out_node->getName() << "\n";);
      }
      for (auto _n : _live_out.getSecond()) {
        auto _tmp = this->for_op_node[_n]->findLiveOutNode(_live_out.getFirst());
        if (_tmp == nullptr) {
          this->for_op_node[_n]->insertLiveOutArgument(_live_out.getFirst(),
                                                           ArgumentNode::ArgumentType::LoopLiveOut);
        }
        this->loop_loop_edge_lout_map[_live_out.getFirst()].push_back(std::make_pair(forOp, _n));
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Conneting live-out values. Edge type 3 \n";);

    // Edge type 3
    // FIXME
    // FIXME

    LLVM_DEBUG(llvm::dbgs() << "currLoopInfo.live_out_out_ins.size: " <<  currLoopInfo.live_out_out_ins.size()<< "\n";);
    for (auto _live_out_edge : currLoopInfo.live_out_out_ins) {
      auto _node = currentLoopNode->findLiveOutNode(_live_out_edge.getFirst());
        if (_node == nullptr) {
          auto _new_live_out_node = currentLoopNode->insertLiveOutArgument(
              _live_out_edge.getFirst(), ArgumentNode::ArgumentType::LiveOut);
          _new_live_out_node->setParentNode(map_value_node[_live_out_edge.getFirst()]);

          LLVM_DEBUG(llvm::dbgs() << "_new_live_out_node " <<  _new_live_out_node->getName() << "\n";);
        }
      for (auto _inst : _live_out_edge.getSecond()) {
        this->live_out_loop_ins_edge[forOp].insert(
            std::make_pair(_live_out_edge.getFirst(), _inst));
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Edge type 3 " <<  currLoopInfo.carry_dependencies.size()<< "\n";);

    // Connecting carry values
    for (auto carry_value : currLoopInfo.carry_dependencies) {
      auto new_carry_depen = currentLoopNode->insertCarryDepenArgument(carry_value.getFirst(), 
                                               ArgumentNode::ArgumentType::CarryDependency);
      if(this->map_value_node.find(carry_value.getFirst()) == this->map_value_node.end()){
        this->map_value_node[carry_value.getFirst()] = new_carry_depen;
      }

      new_carry_depen->setParentNode(this->map_value_node[carry_value.getFirst()]);

        for (auto use : carry_value.getSecond()) {


          bool find_carry_out = llvm::any_of(use->getBlock()->back().getOperands(), 
            [&](mlir::Value value) { return use->getResult(0) == value; });
          bool find_carry_in = llvm::any_of(use->getOperands(), 
            [&](mlir::Value value) { return carry_value.getFirst() == value; });

          this->loop_edge_map[std::make_pair(carry_value.getFirst(), use)] = new_carry_depen;

          // Only for AddIOp.
          if (find_carry_in && !find_carry_out){
            if (auto addAttr = use->getAttrOfType<StringAttr>("Exe")) {
              // Retrieve the string value of the attribute
              llvm::StringRef addValue = addAttr.getValue();
              // Check if the value is "Loop"
              if (addValue.find("Loop") != std::string::npos) {
                  this->edge_direction_map[std::make_pair(carry_value.getFirst(), use)] = 3;//op --> carry value
              }
            }
          }

          if(dyn_cast<dataflow::MergeOp>(use) && find_carry_in){
            this->edge_direction_map[std::make_pair(carry_value.getFirst(), use)] = 1;//carry value --> op
          } else if (find_carry_out && !find_carry_in) {
            this->edge_direction_map[std::make_pair(carry_value.getFirst(), use)] = 2;//op --> carry value
          } else if (find_carry_in && find_carry_out){
            this->edge_direction_map[std::make_pair(carry_value.getFirst(), use)] = 3;// op <--> carry value
          } 
      }
    }

  });

  for(auto edge : this->live_in_ins_loop_edge){
    LLVM_DEBUG(llvm::dbgs() << "value: " << edge.first << "\n";);

    auto from = this->map_value_node.find(edge.first);
    if (from == this->map_value_node.end()) {
      assert(!"WRONG");
    }

    for (auto loop : edge.second) {
      auto loop_node = this->for_op_node[loop];
      LLVM_DEBUG(llvm::dbgs() << "loop: " << loop_node->getName() << "\n";);

      auto to = loop_node->findLiveInNode(edge.first);

      if (to == nullptr)
        assert(!"There is a bug in loop connections!");
      LLVM_DEBUG(llvm::dbgs() << "from: " << from->second->getName()<< "\n";);
      
      from->second->addDataOutputPort(to);
      to->addDataInputPort(from->second);
    }

  }

  for(auto edge : this->live_in_loop_loop_edge){
    for (auto _loop_edge : edge.second) {
      auto _node_src =
          this->for_op_node[_loop_edge.first]->findLiveInNode(edge.first);
      auto _node_tar =
          this->for_op_node[_loop_edge.second]->findLiveInNode(edge.first);

      _node_src->addDataOutputPort(_node_tar);
      _node_tar->addDataInputPort(_node_src);
    }
  }

  for (auto _edge : this->live_out_ins_loop_edge) {
    LLVM_DEBUG(llvm::dbgs() << "live_out value: " << _edge.first << "\n";);

    auto _node_src = this->map_value_node[_edge.first];
    for (auto _tar : _edge.second) {
      auto _loop_dest = this->for_op_node[_tar];
      auto _node_dest = _loop_dest->findLiveOutNode(_edge.first);

      _node_src->addDataOutputPort(_node_dest);
      _node_dest->addDataInputPort(_node_src);
    }
  }


}



void GraphGen::dependencyAnalyze(mlir::func::FuncOp func){

    LLVM_DEBUG(llvm::dbgs() << "\nDependency Analyze. \n ";);
    int i=0;

    //1. Analyze data dependency.
    // Connecting function arguments to the spliter
    for(auto &value : this->funArgValue){
      auto fine_arg_node   = static_cast<ArgumentNode*>(this->map_value_node[value]);
      auto argSpliter      = this->dependency_graph->getArgCall();

      argSpliter->addDataOutputPort(fine_arg_node);
      fine_arg_node->addDataInputPort(argSpliter);
    }

    //Enable 
    func.walk([&](dataflow::StateOp operation) {  
      if(operation.getOperation()->hasAttr("Enable")){
        auto en_branch = static_cast<StateBranchNode*>(this->map_op_node[operation.getOperation()]);
        this->dependency_graph->getArgCall()->addControlOutputPort(en_branch);
        en_branch->addControlInputPort(this->dependency_graph->getArgCall());
      }
    });

    mlir::Operation* currExeOp;
    LLVM_DEBUG(llvm::dbgs() << "\nBranch->Exe Block \n";);

    // Branch->Exe Block
    
    func.walk([&](mlir::Operation* operation) {  
      // operation->dump();
      if(isa<arith::ConstantOp>(operation) || isa<dataflow::EnableOp>(operation) 
        || isa<dataflow::ForOp>(operation) || isa<dataflow::StateOp>(operation)){
        return WalkResult::advance();
      }

      for(uint32_t i = 0; i < operation->getNumOperands(); ++i){
        auto operand = operation->getOperand(i);

        // LLVM_DEBUG(llvm::dbgs() << "operand: " << operand << "\n";);

        // Connect const to operation.
        Node* const_node= nullptr;
        bool find_const = false;
        //
        if(isa<mlir::BlockArgument>(operand)){
          ;
        } else if(isa<arith::ConstantOp>(operand.getDefiningOp())){

          DataType datatype = DataType::IntegerType;
          if(isa<arith::ConstantFloatOp>(operand.getDefiningOp()))
            datatype = DataType::FloatType;

          const_node = this->dependency_graph->insertConstNode(operand, datatype);
          // this->map_value_node[operand] = const_node;
          find_const = true;
          const_node->addControlInputPort(this->map_op_node[operation->getParentOp()]);
          this->map_op_node[operation->getParentOp()]->addControlOutputPort(const_node);
          static_cast<ExecutionBlockNode*>(this->map_op_node[operation->getParentOp()])
                ->addconstNode(static_cast<ConstNode*>(const_node));

          
        } 

        if(find_const){
          auto to = this->map_op_node.find(operation);
          const_node->addDataOutputPort(to->second);
          to->second->addDataInputPort(const_node, i); //Fixed: Resolved an issue with incorrect port handling.
        }
      }
     return WalkResult::advance();
    });


    func.walk([&](mlir::Operation* operation) {  
      // LLVM_DEBUG(llvm::dbgs() << " peration Analyze. \n ";);
      // operation->dump();
      if(isa<arith::ConstantOp>(operation) || isa<dataflow::EnableOp>(operation) 
        || isa<dataflow::ForOp>(operation) || isa<dataflow::StateOp>(operation)){
        return WalkResult::advance();
      }
      if(isa<dataflow::ExecutionBlockOp>(operation)){
        currExeOp = operation;
        // currExeOp->dump();
      }
      // return WalkResult::advance();
      // if(operation->getNumResults() > 0){
      //   llvm::outs() << operation->getName() << " <-name \n";
      //   // auto op_from = this->map_op_node.find(operation);  
      // }
      // auto from = this->map_op_node.find(operation);
      LLVM_DEBUG( operation->dump(););
      llvm::SmallVector<mlir::Operation*> conflict_op;

      for(uint32_t i = 0; i < operation->getNumOperands(); ++i){
        auto operand = operation->getOperand(i);
        LLVM_DEBUG(llvm::dbgs() << "operand: " << operand << "\n";);

        //
        if(isa<mlir::BlockArgument>(operand)){
            ;
        } else if(isa<arith::ConstantOp>(operand.getDefiningOp())){
          continue;
        } 
        auto from  = this->map_value_node.find(operand);
        auto to    = this->map_op_node.find(operation);

        if (from == this->map_value_node.end()  || to == this->map_op_node.end())
          return WalkResult::advance();
        auto from_node = from->second;
        auto to_node   = to->second;
        // Live in
        LLVM_DEBUG(llvm::dbgs() << "live_in check!\n" ;);
        bool find_live_in = false;
        for (auto &data_scc : this->blacklist_loop_live_in_data_edge) {
          for (auto _data_edge : data_scc.getSecond()) {
            if ((data_scc.getFirst() == operand) && (_data_edge == operation)) {
              find_live_in = true;
            }
          }
        }

        if(find_live_in){
          for (auto _loop_edge : this->live_in_loop_ins_edge) {
            auto _loop_node = this->for_op_node[_loop_edge.getFirst()];

            auto _edge = std::find_if(
                _loop_edge.getSecond().begin(),
                _loop_edge.getSecond().end(),
                [operand, operation](auto _f_edge) {
                  return ((_f_edge.first == operand) && (_f_edge.second == operation));
                });

            if (_edge != _loop_edge.getSecond().end()) {
              LLVM_DEBUG(llvm::dbgs() << _edge->first;);

              auto _node_src = _loop_node->findLiveInNode(_edge->first);
              auto _node_tar = this->map_op_node[_edge->second];
              _edge->second->dump();
              LLVM_DEBUG(llvm::dbgs() << _node_src->getName() << " " << _node_tar->getName() << "\n";); 
              _node_src->addDataOutputPort(_node_tar);
              _node_tar->addDataInputPort(_node_src, i); //Fixed: Resolved an issue with incorrect port handling.
            }

          }
          continue;
        }

        bool find_live_out = false;
        LLVM_DEBUG(llvm::dbgs() << "live_out check!\n";);

        for (auto _data_src : this->blacklist_loop_live_out_data_edge) {
          
          for (auto _data_edge : _data_src.getSecond()) {
            
            if ((_data_src.getFirst() == operand) && (_data_edge == operation)) {
              operation->dump();
              LLVM_DEBUG(llvm::dbgs() << _data_edge;);
              
              find_live_out = true;
            }
          }
        }

        if (find_live_out) {
          LLVM_DEBUG(llvm::dbgs() << "find_live_out!\n " ;);
          LLVM_DEBUG(llvm::dbgs() << live_out_loop_ins_edge.size() << "  \n" ;);
          for (auto _loop_edge : this->live_out_loop_ins_edge) {
            auto _loop_node = this->for_op_node[_loop_edge.getFirst()];

            auto _edge = std::find_if(
                _loop_edge.getSecond().begin(),
                _loop_edge.getSecond().end(),
                [operand, operation](auto _f_edge) {
                  return ((_f_edge.first == operand) && (_f_edge.second == operation));
                });
                LLVM_DEBUG(llvm::dbgs() << "live_out!\n" ;);

                operation->dump();
            if (_edge != _loop_edge.getSecond().end()) {
              auto _node_src = _loop_node->findLiveOutNode(_edge->first);
              auto _node_tar = this->map_op_node[_edge->second];
              if(_node_src == nullptr) continue;
              LLVM_DEBUG(llvm::dbgs() << _node_src->getName() << " " << _node_tar->getName()  << "\n";);
              _node_src->addDataOutputPort(_node_tar);
              _node_tar->addDataInputPort(_node_src, i);//Fixed: Resolved an issue with incorrect port handling.
            }
          }

         continue;
        }

        bool find_carry = false;
        bool find_carry_out = false;
        LLVM_DEBUG(llvm::dbgs() << "carry check!\n";);

        for (auto &data_scr : this->blacklist_carry_dependency_data_edge) {
          // Find the carry variable for the iteration.
          if ((data_scr.getFirst() == operand)){

            for (auto data_edge : data_scr.getSecond()) {
              data_edge->dump();
              operation->dump();
              if  (data_edge == operation)
                find_carry = true;
            }
          } 
          // The operands do not directly contain carry variables, 
          // and the result of the operation is used as input to the loop iteration.
          else if(!llvm::is_contained(conflict_op, operation)){
            find_carry_out = llvm::any_of(operation->getBlock()->back().getOperands(), 
                          [&](mlir::Value value) { return operation->getResult(0) == value; }) &&
                          llvm::none_of(this->blacklist_carry_dependency_data_edge,
                                       [&](const auto &blacklist_entry) {
                                           return llvm::is_contained(
                                               operation->getOperands(),
                                               blacklist_entry.getFirst());
                                       });

            conflict_op.push_back(operation);
          }
        }
        
        if (find_carry) {
          auto _carry = this->loop_edge_map[std::make_pair(operand, operation)];
          //from -- operand; to -- operation
          //Operation --> Carry
          LLVM_DEBUG(llvm::dbgs() << from->second->getName() << " " << _carry->getName() << " " << to->second->getName() << "\n";);

          int direction = this->edge_direction_map[std::make_pair(operand, operation)];
          if((direction == 3)){
            to->second->addDataOutputPort(from->second);
            from->second->addDataInputPort(to->second);
          }
          
          _carry->addDataOutputPort(to->second);
          to->second->addDataInputPort(_carry, i); //Fixed: Resolved an issue with incorrect port handling.
          continue;
        } else if (find_carry_out){
          
          auto curr_carry = this->for_op_info[dyn_cast<dataflow::ForOp>(operation->getParentOp()->getParentOp())].carry_dependencies_map[operation];
          assert(this->edge_direction_map[std::make_pair(curr_carry, operation)] >= 2);
          auto _carry = this->loop_edge_map[std::make_pair(curr_carry, operation)];
          LLVM_DEBUG(llvm::dbgs() << "find_carry_out: " << _carry->getName() << " " << to->second->getName() << "\n";);
          to->second->addDataOutputPort(_carry);
          _carry->addDataInputPort(to->second);
          // continue;
        }

        //Fixed: Resolved an issue with incorrect port handling.
        from->second->addDataOutputPort(to->second);
        LLVM_DEBUG(llvm::dbgs() << to->second->getName() << " " << i << "\n";);
        to->second->addDataInputPort(from->second, i);
      }


    });
    
    func.walk([&](mlir::Operation* operation) {  
      //Memory System.
      auto op_node = this->map_op_node.find(operation);
      if (op_node == this->map_op_node.end())
        return WalkResult::advance();
      auto memory_node = static_cast<LSNode*>(op_node->second); 
      if(isa<dataflow::LoadOp>(operation)){
        //TODO: Dynamic memory management.
        this->dependency_graph->getMemoryUnit()->addReadMemoryReqPort(memory_node);
        this->dependency_graph->getMemoryUnit()->addReadMemoryRespPort(memory_node);
        memory_node->addReadMemoryReqPort(this->dependency_graph->getMemoryUnit());
        memory_node->addReadMemoryRespPort(this->dependency_graph->getMemoryUnit());


      } else if(isa<dataflow::StoreOp>(operation)){
        this->dependency_graph->getMemoryUnit()->addWriteMemoryReqPort(memory_node);
        this->dependency_graph->getMemoryUnit()->addWriteMemoryRespPort(memory_node);
        memory_node->addWriteMemoryReqPort(this->dependency_graph->getMemoryUnit());
        memory_node->addWriteMemoryRespPort(this->dependency_graph->getMemoryUnit());
      }

      return WalkResult::advance();
    });

    
    // LLVM_DEBUG(llvm::dbgs() << "Execution Block.\n " ;);
    //Exeblockop 
    func.walk([&](dataflow::ExecutionBlockOp exeop) {  

      if(auto exeblock = static_cast<ExecutionBlockNode*>(this->map_op_node[exeop.getOperation()])){
        // LLVM_DEBUG(llvm::dbgs() << "exeop.\n " ;);
        // exeop.dump();
        for(auto &inner : exeop.getBody().front()){
          // if(isa<arith::IndexCastOp>(inner)){//dataflow::IfOp, dataflow::ExecutionBlockOp, 
          //   continue;
          // }
          auto iter_node = this->map_op_node.find(&inner);
          if (iter_node == map_op_node.end())
            continue;
          if (auto operation_node = static_cast<OperationNode*>(this->map_op_node[&inner])) {
             exeblock->addOperationNode(operation_node);
             operation_node->setParentNode(exeblock);

             exeblock->addControlOutputPort(operation_node);
             operation_node->addControlInputPort(exeblock);

             if (operation_node->getOperationType() == OperationNode::OperationType::MergeType) {

                exeblock->addMergeNode(static_cast<MergeNode*>(operation_node));
                static_cast<MergeNode*>(operation_node)->setParentNode(exeblock);
             }
          }
        }
      }

    });
}

void GraphGen::connectingBranch(mlir::func::FuncOp func){
  LLVM_DEBUG(llvm::dbgs() << "Connecting Branch.\n " ;);


  func.walk([&](dataflow::StateOp operation) {  
    if(operation.getInstanceName().str().find("if_then") != std::string::npos){
        this->state2if[operation] = dyn_cast<dataflow::IfOp>(operation->getNextNode());
    }
  });
  int id = 0;

  for(auto &op : this->state2if){
    auto if_cmp = this->map_value_node[op.getFirst().getCond()];
    static_cast<StateBranchNode*>(this->map_op_node[op.getFirst()])->addDataInputPort(if_cmp);
    if_cmp->addDataOutputPort(static_cast<StateBranchNode*>(this->map_op_node[op.getFirst()]));

   

    if(!op.getSecond().getElseRegion().empty()){
      auto stateNode = static_cast<StateBranchNode*>(this->map_op_node[op.getFirst()]);
      stateNode->addTrueBranch(this->map_op_node[&*op.getSecond().getThenRegion().front().begin()]);
      stateNode->addFalseBranch(this->map_op_node[&*op.getSecond().getElseRegion().front().begin()]);

      this->map_op_node[&*op.getSecond().getElseRegion().front().begin()]->addControlInputPortIndex(stateNode, 0);
      this->map_op_node[&*op.getSecond().getThenRegion().front().begin()]->addControlInputPortIndex(stateNode, 0);
    }

  }


  // state_branch_24.io.PredOp(0) <> store_10.io.SuccOp(0)
  // llvm::DenseMap<dataflow::StoreOp, dataflow::ExecutionBlockOp> storeInblock;
  // func.walk([&](dataflow::StoreOp op) {  
  //   storeInblock[op] = dyn_cast<dataflow::ExecutionBlockOp>(op->getParentOp());
  // });
  // for(auto &opInblock: storeInblock){
  //   auto op_node = this->map_op_node[opInblock.getFirst().getOperation()];
  //   opInblock.getSecond().walk([&](dataflow::StateOp){

  //   });

  //   for(auto &op: opInblock.getSecond()){
  //     op.dump();
  //     if(dyn_cast<dataflow::StateOp>(op)){
  //       LLVM_DEBUG(llvm::dbgs() << "Sllanch.\n " ;);
  //       op_node->addControlOutputPort(this->map_op_node[&op]);
  //       this->map_op_node[&op]->addControlInputPort(op_node);
  //       break;
  //     }
  //   }
  // }
  llvm::SmallVector<dataflow::SelectOp, 8> selectops;
  func.walk([&](dataflow::SelectOp operation) {  
    if (auto selectAttr = operation->getAttrOfType<StringAttr>("Data")) {
      llvm::StringRef selectValue = selectAttr.getValue();
      if (selectValue.find("IF-THEN-ELSE") != std::string::npos) {
          selectops.push_back(operation);
      }
    }
  });

  for(auto &op: selectops){
    auto ifop = dyn_cast<dataflow::IfOp>(op.getTrueValue().getDefiningOp());
    auto to = this->map_op_node[op.getOperation()];
    //IF-THEN
    auto &if_then_exe = ifop.getThenRegion().front().front();
    auto if_then_exeield = if_then_exe.getRegion(0).front().getTerminator()->getOperand(0);
    auto from_then = this->map_value_node[if_then_exeield];
    LLVM_DEBUG(llvm::dbgs() << "if_then_exeield.\n " ;);
    if_then_exeield.getDefiningOp()->dump();
    if(from_then != nullptr){//for op
      
      from_then->addDataOutputPort(to);
      to->addDataInputPort(from_then);
    }
   

    //IF-ELSE
    auto &if_else_exe = ifop.getElseRegion().front().front();
    auto if_else_exeYield = if_else_exe.getRegion(0).front().getTerminator()->getOperand(0);
    if_else_exeYield.getDefiningOp()->dump();
    auto from_else = this->map_value_node[if_else_exeYield];
    LLVM_DEBUG(llvm::dbgs() << "if_from_exeield.\n " ;);
    from_else->addDataOutputPort(to);
    to->addDataInputPort(from_else);

  }
  // exit(0);
}

void Graph::connectingGraph(mlir::func::FuncOp func){
  LLVM_DEBUG(llvm::dbgs() << "Connecting Graph.\n " ;);
  //Connecting the full graph.
  for (auto& const_node : this->const_list) {
    for(auto iter = const_node->outputDataport_begin(); iter != const_node->outputDataport_end(); iter++){
      if (isa<ArgumentNode>(&*iter->first))
        continue;
        this->addEdge(
        Edge::EdgeType::DataTypeEdge,
          std::make_pair(&*const_node, const_node->returnDataOutputPortIndex(&(*iter->first))),
          std::make_pair(&(*iter->first),
                         iter->first->returnDataInputPortIndex(&*const_node))
      );
    }
  }


  for (auto& op_node : this->op_list) {
    for(auto iter = op_node->outputDataport_begin(); iter != op_node->outputDataport_end(); iter++){
      if (isa<ArgumentNode>(&*iter->first))
        continue;
      this->addEdge(
          Edge::EdgeType::DataTypeEdge,
          std::make_pair(&*op_node, op_node->returnDataOutputPortIndex(&(*iter->first))),

          std::make_pair(&(*iter->first), iter->first->returnDataInputPortIndex(&*op_node)));
    }
  }


  for (auto iter = this->getArgCall()->live_in_mems_begin(); iter != this->getArgCall()->live_in_mems_end(); iter++) {
    if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LiveIn)
      continue;
    for(auto iter_data = iter->get()->outputDataport_begin(); iter_data != iter->get()->outputDataport_end(); iter_data++){
      if (isa<ArgumentNode>(&*iter_data->first))
        continue;
      this->addEdge(
          Edge::EdgeType::DataTypeEdge,
          std::make_pair(&*iter->get(), iter->get()->returnDataOutputPortIndex(&*iter_data->first)),
          std::make_pair(&*iter_data->first, iter_data->first->returnDataInputPortIndex(&*iter->get())));
    }
  }


  for (auto iter = this->getArgCall()->live_in_vals_begin(); iter != this->getArgCall()->live_in_vals_end(); iter++) {
    if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LiveIn)
      continue;

    for(auto iter_data = iter->get()->outputDataport_begin(); iter_data != iter->get()->outputDataport_end(); iter_data++){
      if (isa<ArgumentNode>(&*iter_data->first))
        continue;
      this->addEdge(
          Edge::EdgeType::DataTypeEdge,
          std::make_pair(&*iter->get(), iter->get()->returnDataOutputPortIndex(&*iter_data->first)),
          std::make_pair(&*iter_data->first, iter_data->first->returnDataInputPortIndex(&*iter->get())));
    }
  }

  uint32_t cnt = 0;
  auto cache   = this->getMemoryUnit();
  for (auto load_mem = cache->read_req_begin(); load_mem != cache->read_req_end() ;load_mem++) {
    static_cast<LSNode*>(&*load_mem->first)->setRouteID(cnt);
    cnt++;
  }
  for (auto store_mem = cache->write_req_begin(); store_mem != cache->write_req_end() ;store_mem++) {
    static_cast<LSNode*>(&*store_mem->first)->setRouteID(cnt);
    cnt++;
  }
  
  

}