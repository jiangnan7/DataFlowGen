
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "heteacc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/GraphGen.h"
#include <fstream>
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Casting.h>

using namespace mlir;
using namespace heteacc;


ConstNode* Graph::insertConstNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    bool isInt = type == DataType::IntegerType ? true : false;
    int num = this->const_list.size();
    std::string name = data + "_const_" + std::to_string(this->const_list.size());
    this->const_list.push_back(
        std::make_unique<ConstNode>(NodeInfo(this->const_list.size(),name), result.getDefiningOp(), isInt, this->const_list.size())
        
    );
    auto ff = std::find_if(const_list.begin(), const_list.end(), [&result, &num](auto& arg) -> bool {
        return (static_cast<ConstNode*>(arg.get())->getConstID() ==(num));
    });

    return static_cast<ConstNode*>(ff->get());
}   

CmpNode* Graph::insertCmpNode(Value result, DataType type){
// getPredicate
    OpCode code;
    if(auto op = dyn_cast<arith::CmpIOp>(result.getDefiningOp())){
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::sgt:
            code = OpCode::sgt;
            break;
        case arith::CmpIPredicate::eq:
            code = OpCode::eq;
            break;
        case arith::CmpIPredicate::ne:
            code = OpCode::ne;
            break;
        case arith::CmpIPredicate::slt:
            code = OpCode::slt;
            break;
        case arith::CmpIPredicate::sle:
            code = OpCode::sle;
            break;
        case arith::CmpIPredicate::sge:
            code = OpCode::sge;
            break;
        case arith::CmpIPredicate::ult:
            code = OpCode::ult;
            break;
        case arith::CmpIPredicate::ule:
            code = OpCode::ule;
            break;
        case arith::CmpIPredicate::ugt:
            code = OpCode::ugt;
            break;
        case arith::CmpIPredicate::uge:
            code = OpCode::uge;
            break;
        }
    } else {
        switch (dyn_cast<arith::CmpFOp>(result.getDefiningOp()).getPredicate()) {
        case arith::CmpFPredicate::OEQ:
            code = OpCode::eq;
            break;
        case arith::CmpFPredicate::OGT:
            code = OpCode::sgt;
            break;
        case arith::CmpFPredicate::OGE:
            code = OpCode::sge;
            break;
        case arith::CmpFPredicate::OLT:
            code = OpCode::slt;
            break;
        case arith::CmpFPredicate::OLE:
            code = OpCode::sle;
            break;
        case arith::CmpFPredicate::ONE:
            code = OpCode::ne;
            break;
        case arith::CmpFPredicate::UEQ:
            code = OpCode::eq;
            break;
        case arith::CmpFPredicate::UGT:
            code = OpCode::ugt;
            break;
        case arith::CmpFPredicate::UGE:
            code = OpCode::uge;
            break;
        case arith::CmpFPredicate::ULT:
            code = OpCode::ult;
            break;
        case arith::CmpFPredicate::ULE:
            code = OpCode::ule;
            break;
        case arith::CmpFPredicate::UNE:
            code = OpCode::ne;
            break;
        }
    }
    

    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_cmp_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<CmpNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::CmpType, type, result.getDefiningOp(), code

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<CmpNode*>(ff->get());
}
ComputeOperationNode* Graph::insertShrUINode(Value result, DataType type){
        
    std::string name = "int_lshr_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::lshr

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   
ComputeOperationNode* Graph::insertShrNode(Value result, DataType type){
        
    std::string name = "int_shr_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::shr

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   
ComputeOperationNode* Graph::insertShlNode(Value result, DataType type){
        
    std::string name = "int_shl_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::shl

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   
ComputeOperationNode* Graph::insertAddNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_add_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::add

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   

ComputeOperationNode* Graph::insertSubNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_sub_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::sub

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   

ComputeOperationNode* Graph::insertAndiNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_andi_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::andi

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}

ComputeOperationNode* Graph::insertOriNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_ori_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::ori

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}

ComputeOperationNode* Graph::insertDivsiNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_divsi" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::div

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   

ComputeOperationNode* Graph::insertMulNode(Value result, DataType type){
        
    std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = data + "_mul_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ComputeOperationNode>(NodeInfo(op_list.size(), name), 
        OperationNode::OperationType::BinaryType, type, result.getDefiningOp(), OpCode::mul

    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });

    return static_cast<ComputeOperationNode*>(ff->get());
}   

BitCastNode* Graph::insertBitCastNode(arith::IndexCastOp op){
        
    // std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = "cast_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<BitCastNode>(NodeInfo(op_list.size(), name), op.getOperation()
    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&op](auto& arg) -> bool {
        return static_cast<BitCastNode*>(arg.get())->getParentOp() == op.getOperation();
    });

    return static_cast<BitCastNode*>(ff->get());
}   
BitCastNode* Graph::insertBitCastNode(arith::TruncIOp op){
        
    // std::string data = type == DataType::IntegerType ? "int" : "float";
    std::string name = "trunc_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<BitCastNode>(NodeInfo(op_list.size(), name), op.getOperation()
    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&op](auto& arg) -> bool {
        return static_cast<BitCastNode*>(arg.get())->getParentOp() == op.getOperation();
    });

    return static_cast<BitCastNode*>(ff->get());
}   
MergeNode* Graph::insertMergeNode(dataflow::MergeOp op){

    std::string name = "merge_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<MergeNode>(NodeInfo(op_list.size(), name), op.getOperation(),
        op->getParentOp()
    ));

    auto ff = std::find_if(op_list.begin(), op_list.end(), [&op](auto& arg) -> bool {
        return static_cast<MergeNode*>(arg.get())->getParentOp() == op.getOperation();
    });

    return static_cast<MergeNode*>(ff->get());
}

LoopNode* Graph::insertLoopNode(dataflow::ForOp op){

    uint32_t id = this->getLoopNodeNums();
    auto new_loop = std::make_unique<LoopNode>(NodeInfo(id, "loop_" + std::to_string(id)));
    LoopNode* loop_ptr = new_loop.get();
    this->loop_nodes.push_back(std::move(new_loop));
    return loop_ptr;
}

ExecutionBlockNode* Graph::insertExecutionBlockNode(dataflow::ExecutionBlockOp op){

    std::string name = "exe_block_" + std::to_string(this->exe_block_list.size());
    //TODO such funcop...
    bool mask = false;
    if(dyn_cast<dataflow::ForOp>(op->getParentOp())){
        mask = true;
    }   
    this->exe_block_list.push_back(
            std::make_unique<ExecutionBlockNode>(NodeInfo(exe_block_list.size(), name), op, mask));
    auto ff = std::find_if(
        this->exe_block_list.begin(), this->exe_block_list.end(), [op](auto& arg) -> bool {
        return static_cast<ExecutionBlockNode*>(arg.get())->getRelatedOp() == op;
    });
    return static_cast<ExecutionBlockNode*>(ff->get());

}

StateBranchNode* Graph::insertStateBranchNode(dataflow::StateOp op){

    std::string name = "state_branch_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<StateBranchNode>(NodeInfo(this->op_list.size(), name), op));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [op](auto& arg) -> bool {
        return static_cast<StateBranchNode*>(arg.get())->getRelatedOp() == op;
    });
    return static_cast<StateBranchNode*>(ff->get());
}

StateBranchNode* Graph::insertStateBranchNode(dataflow::EnableOp op){

    std::string name = "enable_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<StateBranchNode>(NodeInfo(this->op_list.size(), name), op));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [op](auto& arg) -> bool {
        return static_cast<StateBranchNode*>(arg.get())->getControlOp() == op;
    });
    return static_cast<StateBranchNode*>(ff->get());
}
AddressGenNode* Graph::insertAddressGenNode(dataflow::AddressOp op){
        
    std::string name = "address_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<AddressGenNode>(NodeInfo(this->op_list.size(), name), op));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [op](auto& arg) -> bool {
        return static_cast<AddressGenNode*>(arg.get())->getRelatedOp() == op;
    });
    return static_cast<AddressGenNode*>(ff->get());
}  

SelectNode* Graph::insertSelectNode(dataflow::SelectOp op){

    std::string name = "select_" + std::to_string(this->op_list.size());
    bool dataSelect = false;
    if(auto attr = op->getAttr("Select")){
        if(attr.cast<StringAttr>().str().find("Data") != std::string::npos){
            dataSelect = true;
        }
    }
    this->op_list.push_back(
        std::make_unique<SelectNode>(NodeInfo(this->op_list.size(), name), op, dataSelect));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [op](auto& arg) -> bool {
        return static_cast<SelectNode*>(arg.get())->getRelatedOp() == op;
    });
    
    return static_cast<SelectNode*>(ff->get());
}

SelectNode* Graph::insertSelectNode(arith::SelectOp op){

    std::string name = "select_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<SelectNode>(NodeInfo(this->op_list.size(), name), op));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [op](auto& arg) -> bool {
        return static_cast<SelectNode*>(arg.get())->getArithRelatedOp() == op;
    });
    return static_cast<SelectNode*>(ff->get());
}

LSNode* Graph::insertLoadNode(Value result, DataType type){
    std::string name = "load_" + std::to_string(this->op_list.size());
    if(type == DataType::IntegerType){
        this->op_list.push_back(
        std::make_unique<LSNode>(NodeInfo(this->op_list.size(), name), OperationNode::OperationType::LoadType,
                                   DataType::IntegerType,
                                   LSNode::opmemType::load, result.getDefiningOp(),
                                   this->getMemoryUnit()));
    } else if(type == DataType::FloatType){
        this->op_list.push_back(
        std::make_unique<LSNode>(NodeInfo(this->op_list.size(), name), OperationNode::OperationType::LoadType,
                                   DataType::FloatType,
                                   LSNode::opmemType::load, result.getDefiningOp(),
                                   this->getMemoryUnit()));
    //isArrayTy
    } else{

    }
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });
    //TODO load/store节点里面隐含adder节点，后续增加。
    return static_cast<LSNode*>(ff->get());
}

LSNode* Graph::insertStoreNode(Value result, DataType type, mlir::Operation* op){
    std::string name = "store_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<LSNode>(NodeInfo(this->op_list.size(), name), OperationNode::OperationType::StoreType,
                                   LSNode::opmemType::store, op,
                                   this->getMemoryUnit()));

    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [&op](auto& arg) -> bool {
        return arg.get()->getOperation() == op;
    });
    return static_cast<LSNode*>(ff->get());
}


AllocaNode* Graph::insertAllocaNode(Value result, OperationNode::OperationType optype, uint32_t size, uint32_t num_byte){
    std::string name = "alloca_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<AllocaNode>(NodeInfo(this->op_list.size(), name), 
        OperationNode::OperationType::AllocaType, DataType::MemrefType, result.getDefiningOp(),
        size, num_byte
    ));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [&result](auto& arg) -> bool {
        return arg.get()->getOperation() == result.getDefiningOp();
    });
    return static_cast<AllocaNode*>(ff->get());
}

MemoryNode* Graph::createBufferMemory(AllocaNode* alloca, uint32_t size, uint32_t num_byte){

    this->scratchpad_memories.push_back(std::make_unique<MemoryNode>(
      NodeInfo(scratchpad_memories.size(),
               "buffer_memories_" + std::to_string(scratchpad_memories.size())),
      MemoryNode::memType::scratchpad, alloca, size, num_byte));
    return static_cast<MemoryNode*>(scratchpad_memories.end()->get());
 
    
}


ReturnNode* Graph::insertReturnNode(mlir::func::ReturnOp returnop){
    std::string name = "return_" + std::to_string(this->op_list.size());
    this->op_list.push_back(
        std::make_unique<ReturnNode>(NodeInfo(this->op_list.size(), name), 
        OperationNode::OperationType::ReturnType, returnop.getOperation()
    ));
    auto ff = std::find_if(this->op_list.begin(), this->op_list.end(), [&returnop](auto& arg) -> bool {
        return arg.get()->getOperation() == returnop.getOperation();
    });
    return static_cast<ReturnNode*>(ff->get());
}

/**
 * Insert a new Edge
 */
Edge* Graph::addEdge(Edge::EdgeType _typ, Port _node_src, Port _node_dst) {
  // TODO fix the indexing
  this->edge_list.push_back(std::make_unique<Edge>(_typ, _node_src, _node_dst));
  auto ff =
      std::find_if(edge_list.begin(),
                   edge_list.end(),
                   [_node_src, _node_dst](std::unique_ptr<Edge>& e) -> bool {
                     return (e->getSrc() == _node_src) && (e->getTar() == _node_dst);
                   });

  return ff->get();
}

void Graph::dumpECDFG(){

  std::ofstream _out_file("./output/" + this->graph_info.Name + ".json");
  _out_file << "The number of Nodes: "  << op_list.size()+arg_list.size()+loop_nodes.size()+exe_block_list.size()<< "\n";
  _out_file << "The number of Edges: "  << edge_list.size() << "\n";
  _out_file.close();
}