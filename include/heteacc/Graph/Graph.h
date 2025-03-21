#ifndef GRAPH_STRUCTURE_H
#define GRAPH_STRUCTURE_H

#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Node.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/InitAllDialects.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir {
namespace heteacc {

class Node;
class ArgumentNode;
class ComputeOperationNode;
class OperationNode;
class MemoryNode;
class CmpNode;
class Edge;

using ArgumentList    = std::list<std::unique_ptr<ArgumentNode>>;
using DummyList       = std::list<std::unique_ptr<Node>>;
using OperationList   = std::list<std::unique_ptr<OperationNode>>;
using ScratchpadList  = std::list<std::unique_ptr<MemoryNode>>;
using LoopNodeList    = std::list<std::unique_ptr<LoopNode>>;
using ExecutionBlockList       = std::list<std::unique_ptr<ExecutionBlockNode>>;
using BuiltInList     = std::list<std::unique_ptr<OperationNode>>;
using ConstList       = std::list<std::unique_ptr<ConstNode>>;
using EdgeList        = std::list<std::unique_ptr<Edge>>;
using Port            = std::pair<Node*, PortID>;

class Edge {
   public:
    using Port = std::pair<Node *, PortID>;

    enum EdgeType {
        DataTypeEdge = 0,
        ControlTypeEdge,
        MaskTypeEdge,
        MemoryReadTypeEdge,
        MemoryWriteTypeEdge,
        UknownType
    };

   private:
    EdgeType edge_type;
    Port src;
    Port tar;

   public:
    explicit Edge(EdgeType _ty) : edge_type(_ty) {}
    explicit Edge(Port _src, Port _tar)
        : edge_type(UknownType), src(_src), tar(_tar) {}
    explicit Edge(EdgeType _ty = UknownType, Port _src = {nullptr, 0},
                  Port _tar = {nullptr, 0})
        : edge_type(_ty), src(_src), tar(_tar) {}

    uint32_t getType() const { return edge_type; }

    Port getSrc() const { return src; }
    Port getTar() const { return tar; }

    bool operator == (const Edge &rhs) const {
        return this->getSrc() == rhs.getSrc();
    }
};

class Graph{

private:
  OperationList op_list;  
  DummyList dummy_list;  
  ArgumentList  arg_list;
 
  // Local memories
  ScratchpadList scratchpad_memories;
  // Loop nodes
  LoopNodeList loop_nodes;
  // Block nodes
  ExecutionBlockList exe_block_list;

  // List of the edges between nodes inside the graph
  EdgeList edge_list;

  ConstList const_list;
  
  BuiltInList built_node_list;
  // Splitcall for the function
  std::unique_ptr<FineArgCallNode> fine_arg_call;

  std::list<Value> countValue;  
  llvm::raw_ostream& outputHardware;

  Node* out_node;
public:
    NodeInfo graph_info;

    
    llvm::SmallVector<mlir::Value, 8> funArgValue;
    //Output
    bool hasreturnValue = false;
    mlir::Operation* returnOp;
    Value returnValue;
    // Memory units inside each graph
    std::unique_ptr<MemoryNode> memory_unit;

    explicit Graph()
    : graph_info(NodeInfo(0, "dummy")),
      outputHardware(llvm::outs())
    {}

    explicit Graph(NodeInfo _n_info)
    : graph_info(_n_info),
      fine_arg_call(std::make_unique<FineArgCallNode>(NodeInfo(0, "FineGrainedArgCall"))),
      memory_unit(std::make_unique<MemoryNode>(NodeInfo(0, "mem_ctrl_cache"), MemoryNode::memType::memory)),
      outputHardware(llvm::outs())
    {}

    explicit Graph(NodeInfo _n_info, llvm::raw_ostream& _output)
    : graph_info(_n_info),
      fine_arg_call(std::make_unique<FineArgCallNode>(NodeInfo(0, "FineGrainedArgCall"))),
      memory_unit(std::make_unique<MemoryNode>(NodeInfo(0, "mem_ctrl_cache"), MemoryNode::memType::memory)),
      outputHardware(_output)
    {}

    func::FuncOp top_function;
    std::map<uint32_t, Node*> id2node; 
    // explicit Graph()
    //    :   
    

public:

    auto
    getMemoryUnit() const {
        return this->memory_unit.get();
    }

    std::unordered_map<int , MemoryNode*> memID2Node;

    ArgumentNode* storeArg(Value arg){
        this->countValue.push_back(arg);
        this->arg_list.push_back(std::make_unique<ArgumentNode>(NodeInfo(this->arg_list.size(), "function_argument_" + std::to_string(this->countValue.size())),
        ArgumentNode::ArgumentType::LiveIn, isDataType(arg), arg
        ));
        return arg_list.back().get();
    };
    void dumpGraph(PrintType, std::string config_path = "");

    void dumpECDFG();

    ConstNode* insertConstNode(Value result, DataType type);

    CmpNode* insertCmpNode(Value result, DataType type);

    ComputeOperationNode* insertShrUINode(Value result, DataType type);
    ComputeOperationNode* insertShrNode(Value result, DataType type);
    
    ComputeOperationNode* insertAddNode(Value result, DataType type);
    ComputeOperationNode* insertMulNode(Value result, DataType type);
    ComputeOperationNode* insertDivsiNode(Value result, DataType type);
    ComputeOperationNode* insertSubNode(Value result, DataType type);
    ComputeOperationNode* insertAndiNode(Value result, DataType type);
    ComputeOperationNode* insertOriNode(Value result, DataType type);
    ComputeOperationNode* insertShlNode(Value result, DataType type);
    
    ReductionNode* insertReductionNode(Value result, DataType type);

    BitCastNode* insertBitCastNode(arith::IndexCastOp op);
    BitCastNode* insertBitCastNode(arith::TruncIOp op);
    BitCastNode* insertBitCastNode(vector::BroadcastOp op);
    MergeNode* insertMergeNode(dataflow::MergeOp op);

    LSNode* insertLoadNode(Value result, DataType type);
    LSNode* insertStoreNode(Value result, DataType type, mlir::Operation* op);

    LoopNode* insertLoopNode(dataflow::ForOp op);
    
    ExecutionBlockNode* insertExecutionBlockNode(dataflow::ExecutionBlockOp op);
    
    StateBranchNode* insertStateBranchNode(dataflow::StateOp op);
    StateBranchNode* insertStateBranchNode(dataflow::EnableOp op);

    AddressGenNode* insertAddressGenNode(dataflow::AddressOp op);

    SelectNode* insertSelectNode(dataflow::SelectOp op);
    SelectNode* insertSelectNode(arith::SelectOp op);
    
    ReturnNode* insertReturnNode(mlir::func::ReturnOp returnop);

    AllocaNode* insertAllocaNode(Value result, OperationNode::OperationType optype,  uint32_t size, uint32_t num_byte);

    MemoryNode* createBufferMemory(AllocaNode* alloca, uint32_t size, uint32_t num_byte);
    MemoryNode* createBufferMemory(uint32_t id, uint32_t size, uint32_t num_byte = 32);

    Edge* addEdge(Edge::EdgeType, Port src, Port dst);
    
    Node* getOutNode(){
      return this->out_node;
    }
    uint32_t getOpListSize() {
      return this->op_list.size();
    }
    uint32_t getExecutionBlockListSize() {
      return this->exe_block_list.size();
    }
    uint32_t getBuiltListSize() {
      return this->built_node_list.size();
    }
    uint32_t getLoopNodeNums() {
      return this->loop_nodes.size();
    }
    void setOutputNode(Node* node) {
      this->out_node = node;
    }
    FineArgCallNode* getArgCall() const {
      return fine_arg_call.get();
    }
    const ScratchpadList& getScratchpadMemories() const {
        return this->scratchpad_memories;
    }
    void connectingGraph(mlir::func::FuncOp func);

protected:
  // Scala specific functions
  void printScalaHeader();

  void printOperations(PrintType _pt);

  void printControlEdge(PrintType _pt);   
  void printConnection(PrintType _pt) ;

  void printLoopConnection(PrintType _pt);

  void printScalaEnd(PrintType _pt);
};


}
}



#endif // GRAPH_STRUCTURE_H