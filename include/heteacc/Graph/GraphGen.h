#ifndef GRAPH_INIT_H
#define GRAPH_INIT_H

#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/Graph.h"
#include "heteacc/Graph/Node.h"
#include "heteacc/InitAllDialects.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "heteacc/Misc/VecUtils.h"
#include <queue>
#include <set>
#include <string>
#include <numeric>
#include <unordered_map>

namespace mlir {
namespace heteacc {

// #define DEBUG_TYPE "graph"
struct LoopInfo {
  std::string loop_name;

  int32_t loop_level = 999;
  int32_t loop_band = 999;
  bool loop_structure = true;

  mlir::Operation* enable = nullptr;

  mlir::Operation* loop_back = nullptr;
  mlir::Operation* loop_finish = nullptr;
  
  
  llvm::SmallVector<mlir::Operation*, 8> selectVec;

  mlir::Operation* exe_block_op;
  // Live-in
  // Loop -> ins (output)
  llvm::DenseMap<Value, llvm::SmallSetVector<Operation*, 8>> live_in_out_ins;

  // Loop -> Loop 
  llvm::DenseMap<Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_in_out_loop;

  // Loop <> ins (input)
  llvm::SmallSetVector<Value, 8> live_in_in_ins;

  // Loop -> Loop (input) 
  llvm::DenseMap<Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_in_in_loop;

  // Live-out
  // Loop -> ins (output)
  llvm::DenseMap<Value, llvm::SmallSetVector<Operation*, 8>>
      live_out_out_ins;

  // Loop -> Loop (output)
  llvm::DenseMap<Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_out_out_loop;

  // Loop -> ins (input)
  llvm::SmallSetVector<Value, 8> live_out_in_ins;

  // Loop -> Loop (input)
  llvm::DenseMap<Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_out_in_loop;


  llvm::DenseMap<Value, llvm::SmallVector<mlir::Operation*, 8>> carry_dependencies;

  llvm::DenseMap<Value, llvm::SmallVector<mlir::Operation*, 8>> carry_dependencies_without_connection;

  llvm::DenseMap<mlir::Operation*, Value> carry_dependencies_map;
  
  LoopInfo() : loop_name("Null")  {}
  LoopInfo(std::string name) : loop_name(name) {}
}; 


class GraphGen : public GraphVisitorBase<GraphGen, void>, 
                 public Graph {

public:
  std::unique_ptr<Graph> dependency_graph;
  

  // Default value is standard out
  llvm::raw_ostream& code_out;

  llvm::DenseMap<mlir::Value , Node*> map_value_node;
  llvm::DenseMap<mlir::Operation* , Node*> map_op_node;
  llvm::DenseMap<mlir::Operation* , llvm::SmallSetVector<Node*, 8>> op_built_node;

  llvm::DenseMap<dataflow::ForOp, LoopInfo> for_op_info;
  llvm::DenseMap<dataflow::ForOp, LoopNode*> for_op_node;
  llvm::DenseMap<mlir::memref::AllocaOp, MemoryNode*> memory_buffer_map;

  llvm::DenseMap<mlir::Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_in_ins_loop_edge;

  llvm::DenseMap<mlir::Value, llvm::DenseSet<std::pair<dataflow::ForOp, dataflow::ForOp>>>
      live_in_loop_loop_edge;

  llvm::DenseMap<dataflow::ForOp, llvm::DenseSet<std::pair<mlir::Value, mlir::Operation*>>>
      live_in_loop_ins_edge;

  llvm::DenseMap<std::pair<Value, mlir::Operation * >, ArgumentNode*> loop_edge_map;
  llvm::DenseMap<std::pair<Value, mlir::Operation * >, int> edge_direction_map; // 1: input, 2: output, 3. in/out

  llvm::DenseMap<Value, llvm::SmallVector<Operation*, 8>> 
      blacklist_loop_live_in_data_edge;

  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<mlir::Operation*, 8>>
      blacklist_control_edge;

  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation*, 8>>
      blacklist_carry_dependency_data_edge;
      
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation*, 8>>
      blacklist_loop_live_out_data_edge;

  llvm::DenseMap<mlir::Value, llvm::SmallSetVector<dataflow::ForOp, 8>> live_out_ins_loop_edge;

  llvm::DenseMap<mlir::Value, llvm::SmallVector<std::pair<dataflow::ForOp, dataflow::ForOp>>>
      loop_loop_edge_lout_map;

  llvm::DenseMap<dataflow::ForOp, llvm::DenseSet<std::pair<Value, mlir::Operation*>>>
      live_out_loop_ins_edge;
  
  LoopInfo analyzeLoopNode(dataflow::ForOp op);

  llvm::DenseMap<dataflow::StateOp, dataflow::IfOp> state2if;

  std::map<Operation *, bool> op2traversal;
  GraphGen(): 
    dependency_graph(std::make_unique<Graph>(NodeInfo(0, "dummy"))),
    code_out(llvm::outs())
    {}

  GraphGen(NodeInfo _n_info, llvm::raw_ostream& out):
    dependency_graph(std::make_unique<Graph>(_n_info, out)),
    code_out(out)
    {}
//Visit
public:


  bool applyGraphInit(func::FuncOp func, bool topFunc);
  void buildLoopGraph(mlir::func::FuncOp func);
  void dependencyAnalyze(mlir::func::FuncOp func);
  void connectingBranch(mlir::func::FuncOp func);


  void buildEnhancedControlDataFlowGraph(func::FuncOp func);


  using GraphVisitorBase::visitOp;


  void visitOp(func::CallOp op) {
    // assert(!"wait...");
  }

  void visitOp(func::ReturnOp op) {
    
    if(op.getOperands().size() == 0){
      this->dependency_graph->hasreturnValue = false;
    }else {
      this->dependency_graph->hasreturnValue = true;
    }
    auto return_node = this->dependency_graph->insertReturnNode(op);
    this->map_op_node[op.getOperation()] = return_node;
    this->dependency_graph->setOutputNode(this->map_op_node[op.getOperation()]);
    this->dependency_graph->returnOp = op.getOperation();
    for(auto operand: op.getOperands()){
        // auto return_node = this->dependency_graph->insertReturnNode(op);
        // this->map_op_node[op.getOperation()] = return_node;
        this->dependency_graph->returnValue = operand;
        this->map_value_node[operand] = return_node;
    }
  }

  void visitOp(arith::ShRUIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto shr_node = this->dependency_graph->insertShrUINode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      shr_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = shr_node;
      this->map_op_node[op.getOperation()] = shr_node;
    } else {
      auto shr_node = this->dependency_graph->insertShrUINode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = shr_node;
      this->map_op_node[op.getOperation()] = shr_node;
    }
  }

  void visitOp(arith::ShRSIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto shr_node = this->dependency_graph->insertShrNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      shr_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = shr_node;
      this->map_op_node[op.getOperation()] = shr_node;
    } else {
      auto shr_node = this->dependency_graph->insertShrNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = shr_node;
      this->map_op_node[op.getOperation()] = shr_node;
    }
  }

  void visitOp(arith::ShLIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto shl_node = this->dependency_graph->insertShlNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      shl_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = shl_node;
      this->map_op_node[op.getOperation()] = shl_node;
    } else {
      auto shl_node = this->dependency_graph->insertShlNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = shl_node;
      this->map_op_node[op.getOperation()] = shl_node;
    }
  }

  void visitOp(arith::AddIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto add_node = this->dependency_graph->insertAddNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      add_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = add_node;
      this->map_op_node[op.getOperation()] = add_node;
    } else {
      auto add_node = this->dependency_graph->insertAddNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = add_node;
      this->map_op_node[op.getOperation()] = add_node;
    }
  }

  void visitOp(arith::AddFOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto add_node = this->dependency_graph->insertAddNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      add_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = add_node;
      this->map_op_node[op.getOperation()] = add_node;
    } else {
      auto add_node = this->dependency_graph->insertAddNode(op.getResult(), DataType::FloatType);
      this->map_value_node[op.getResult()] = add_node;
      this->map_op_node[op.getOperation()] = add_node;
    } 
  }
  
  void visitOp(arith::SubIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto sub_node = this->dependency_graph->insertSubNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      sub_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = sub_node;
      this->map_op_node[op.getOperation()] = sub_node;
    } else {
      auto sub_node = this->dependency_graph->insertSubNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = sub_node;
      this->map_op_node[op.getOperation()] = sub_node;
    }  
  }

  void visitOp(arith::SubFOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto sub_node = this->dependency_graph->insertSubNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      sub_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = sub_node;
      this->map_op_node[op.getOperation()] = sub_node;
    } else {
      auto sub_node = this->dependency_graph->insertSubNode(op.getResult(), DataType::FloatType);
      this->map_value_node[op.getResult()] = sub_node;
      this->map_op_node[op.getOperation()] = sub_node;
    }
  }

  void visitOp(arith::AndIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto and_node = this->dependency_graph->insertAndiNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      and_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = and_node;
      this->map_op_node[op.getOperation()] = and_node;
    } else {
      auto and_node = this->dependency_graph->insertAndiNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = and_node;
      this->map_op_node[op.getOperation()] = and_node;
    }
  }

  void visitOp(arith::OrIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto or_node = this->dependency_graph->insertOriNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      or_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = or_node;
      this->map_op_node[op.getOperation()] = or_node;
    } else {
      auto or_node = this->dependency_graph->insertOriNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = or_node;
      this->map_op_node[op.getOperation()] = or_node;
    }
  }

  void visitOp(arith::DivSIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto div_node = this->dependency_graph->insertDivsiNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      div_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = div_node;
      this->map_op_node[op.getOperation()] = div_node;
    } else {
      auto div_node = this->dependency_graph->insertDivsiNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = div_node;
      this->map_op_node[op.getOperation()] = div_node;
    }
  }


  void visitOp(arith::MulIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto mul_node = this->dependency_graph->insertMulNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      mul_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = mul_node;
      this->map_op_node[op.getOperation()] = mul_node;
    } else {
      auto mul_node = this->dependency_graph->insertMulNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = mul_node;
      this->map_op_node[op.getOperation()] = mul_node;
    }
  }

  void visitOp(arith::MulFOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto mul_node = this->dependency_graph->insertMulNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      mul_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = mul_node;
      this->map_op_node[op.getOperation()] = mul_node;
    } else {
      auto mul_node = this->dependency_graph->insertMulNode(op.getResult(), DataType::FloatType);
      this->map_value_node[op.getResult()] = mul_node;
      this->map_op_node[op.getOperation()] = mul_node;
    }
  }


  void visitOp(arith::CmpIOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto cmp_node = this->dependency_graph->insertCmpNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      cmp_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = cmp_node;
      this->map_op_node[op.getOperation()] = cmp_node;
    } else {
      auto cmp_node = this->dependency_graph->insertCmpNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = cmp_node;
      this->map_op_node[op.getOperation()] = cmp_node;
    }
  }

  void visitOp(arith::CmpFOp op) {
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      auto cmp_node = this->dependency_graph->insertCmpNode(op.getResult(), DataType::VectorType);
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      cmp_node->setLaneNums(laneSize);
      this->map_value_node[op.getResult()] = cmp_node;
      this->map_op_node[op.getOperation()] = cmp_node;
    } else {
      auto cmp_node = this->dependency_graph->insertCmpNode(op.getResult(), DataType::FloatType);
      this->map_value_node[op.getResult()] = cmp_node;
      this->map_op_node[op.getOperation()] = cmp_node;
    }
  }

  void visitOp(arith::IndexCastOp op) {
    auto dummy_node = this->dependency_graph->insertBitCastNode(op);
    this->map_op_node[op.getOperation()] = dummy_node;
    this->map_value_node[op.getResult()] = dummy_node;
  }
  
  void visitOp(arith::TruncIOp op) {
    auto dummy_node = this->dependency_graph->insertBitCastNode(op);
    this->map_op_node[op.getOperation()] = dummy_node;
    this->map_value_node[op.getResult()] = dummy_node;
  }

  void visitOp(dataflow::MergeOp op) {
    auto merge_node = this->dependency_graph->insertMergeNode(op);
    this->map_op_node[op.getOperation()] = merge_node;
    this->map_value_node[op.getResult()] = merge_node;
  }

  void visitOp(memref::AllocaOp op) {
    MemRefType memRefType = op.getResult().getType().dyn_cast<MemRefType>();
    Type elementType = memRefType.getElementType();
    ArrayRef<int64_t> shape = op.getType().cast<MemRefType>().getShape();
    int64_t totalElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    int64_t elementByte = elementType.getIntOrFloatBitWidth() / 8;
    
    if(elementType.isa<mlir::IntegerType>()){

      auto alloca_node = this->dependency_graph->insertAllocaNode(op.getResult(), OperationNode::OperationType::AllocaType,
      totalElements, elementByte
      );
      this->map_value_node[op.getResult()] = alloca_node;
      this->map_op_node[op.getOperation()] = alloca_node;
      this->memory_buffer_map[op] = 
        this->dependency_graph->createBufferMemory(alloca_node, totalElements, elementByte);

    } else if(elementType.isa<mlir::FloatType>()){
      auto alloca_node = this->dependency_graph->insertAllocaNode(op.getResult(), OperationNode::OperationType::AllocaType,
      totalElements, elementByte
      );
      this->map_op_node[op.getOperation()] = alloca_node;
      this->map_value_node[op.getResult()] =  alloca_node;
      this->memory_buffer_map[op] = 
        this->dependency_graph->createBufferMemory(alloca_node, totalElements, elementByte);
    }
    else {
      op.dump();
      assert(!"Don't support for this alloca");
    }

  }

  void visitOp(dataflow::LoadOp op) {
    auto valueType = op.getResult().getType();

    if(valueType.isa<mlir::IntegerType>()){
      auto load_node = this->dependency_graph->insertLoadNode(op.getResult(), DataType::IntegerType);
      this->map_value_node[op.getResult()] = load_node;
      this->map_op_node[op.getOperation()] = load_node;
    } else if(valueType.isa<mlir::FloatType>()){
      auto load_node = this->dependency_graph->insertLoadNode(op.getResult(), DataType::FloatType);
      this->map_value_node[op.getResult()] = load_node;
      this->map_op_node[op.getOperation()] = load_node;
    } else if(valueType.isa<mlir::VectorType>()){
      auto load_node = this->dependency_graph->insertLoadNode(op.getResult(), DataType::VectorType);
      this->map_value_node[op.getResult()] = load_node;
      this->map_op_node[op.getOperation()] = load_node;
    }
    else {
      op.dump();
      assert(!"Don't support for this ");
    }

  }

  void visitOp(dataflow::StoreOp op) {
    auto valueType = op.getValue().getType();

    if(valueType.isa<mlir::IntegerType>()){
      auto store_node = this->dependency_graph->insertStoreNode(op.getAddress(), DataType::IntegerType, op.getOperation());
      // this->map_value_node[op.getMemRef()] = store_node;
      this->map_op_node[op.getOperation()] = store_node;
    } else if(valueType.isa<mlir::FloatType>()){
      auto store_node = this->dependency_graph->insertStoreNode(op.getAddress(), DataType::FloatType, op.getOperation());
      // this->map_value_node[op.getMemRef()] = store_node;
      this->map_op_node[op.getOperation()] = store_node;
    } else if(valueType.isa<mlir::VectorType>()){
      auto store_node = this->dependency_graph->insertStoreNode(op.getAddress(), DataType::VectorType, op.getOperation());
      // this->map_value_node[op.getMemRef()] = store_node;
      this->map_op_node[op.getOperation()] = store_node;
    }
    else {
      op.dump();
      assert(!"Don't support for this ");
    }
  }

  void visitOp(scf::ForOp op) {
    // uint32_t id = this->dependency_graph->getLoopNodeNums();
    // this->dependency_graph->analyzeLoopNode(op);
  }

  void visitOp(dataflow::ForOp op) {
    uint32_t id = this->dependency_graph->getLoopNodeNums();
    // this->dependency_graph->analyzeLoopNode(op);
  
    //Anzlyze Loop Node.
    this->for_op_info.insert(std::make_pair(op, analyzeLoopNode(op)));
    auto loop_node = this->dependency_graph->insertLoopNode(op);
    this->for_op_node[op] = loop_node;
  }

  void visitOp(dataflow::ExecutionBlockOp op) {
    this->map_op_node[op.getOperation()] = this->dependency_graph->insertExecutionBlockNode(op);
  }

  void visitOp(dataflow::StateOp op) {
    this->map_op_node[op.getOperation()] = this->dependency_graph->insertStateBranchNode(op);
  }

  void visitOp(dataflow::SelectOp op) {
    auto select_node = this->dependency_graph->insertSelectNode(op);
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      select_node->setLaneNums(laneSize);
    }
    this->map_op_node[op.getOperation()] = select_node;
    this->map_value_node[op.getResult()] = select_node;
  }
  void visitOp(arith::SelectOp op) {
    auto select_node = this->dependency_graph->insertSelectNode(op);
    auto valueType = op.getResult().getType();
    if(valueType.isa<mlir::VectorType>()){
      unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
      select_node->setLaneNums(laneSize);
    }
    this->map_op_node[op.getOperation()] = select_node;
    this->map_value_node[op.getResult()] = select_node;
  }
  void visitOp(dataflow::EnableOp op) {
    auto enable_node = this->dependency_graph->insertStateBranchNode(op);
    this->map_op_node[op.getOperation()] = enable_node;
    this->map_value_node[op.getResult()] = enable_node;
  }
  void visitOp(dataflow::AddressOp op) {
    auto address_node = this->dependency_graph->insertAddressGenNode(op);
    if (auto lane = op->getAttr("laneNums")) {
        address_node->setLaneNums(lane.cast<IntegerAttr>().getInt());
    }
    this->map_op_node[op.getOperation()] = address_node;
    this->map_value_node[op.getResult()] = address_node;
  }

  void visitOp(vector::ReductionOp op) {
    auto valueType = op.getVector().getType();
    auto reduction_op = this->dependency_graph->insertReductionNode(op.getResult(), DataType::VectorType);
    unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
    reduction_op->setLaneNums(laneSize);
    this->map_value_node[op.getResult()] = reduction_op;
    this->map_op_node[op.getOperation()] = reduction_op;
  
  }

  void visitOp(vector::BroadcastOp op) {
    auto valueType = op.getVector().getType();
    auto broadcast_op = this->dependency_graph->insertBitCastNode(op);
    unsigned laneSize = getVectorLaneSize(dyn_cast<mlir::VectorType>(valueType));
    broadcast_op->setLaneNums(laneSize);
    this->map_value_node[op.getResult()] = broadcast_op;
    this->map_op_node[op.getOperation()] = broadcast_op;
  
  }

};



}
}



#endif // GRAPH_INIT_H