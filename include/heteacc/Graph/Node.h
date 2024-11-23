#ifndef NODE_STRUCTURE_H
#define NODE_STRUCTURE_H

#include <stdint.h>
#include <list>
#include <map>
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Attributes.h"
#include "heteacc/Graph/Utils.h"
#include "heteacc/Graph/Visitor.h"
#include "heteacc/Graph/Graph.h"
#include "heteacc/InitAllDialects.h"
#include "llvm/ADT/TypeSwitch.h"
#include <unordered_map>
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
namespace mlir {
namespace heteacc {


class Graph;
class Node;
class MemoryNode;
class LoopNode;
class LSNode;
class ReturnNode;
class AllocaNode;
class ComputeOperationNode;
class CmpNode;
class Node;
class ExecutionBlockNode;
class OperationNode;
class ContainerNode;
class MergeNode; 

enum DataType { IntegerType = 0, FloatType, MemrefType, UknownType };

enum OpCode {
    add = 0,
    mul,
    sub,
    eq,
    ne,
    slt,
    sle,
    sgt,
    sge,
    ult,
    ule,
    ugt,
    uge,
    shr,
    shl,
    ori,
    lshr,
    div,
    andi,
    other
  };

std::string opCodeToString(OpCode code);

DataType isDataType(Value arg);

enum PrintType { Scala = 0, Dot, Json };
struct PortID {
  uint32_t ID;

  // Default value for ID is equalt to zero
  PortID() : ID(0) {}
  PortID(uint32_t _id) : ID(_id) {}

  uint32_t
  getID() {
    return ID;
  }
  void
  setID(uint32_t _id) {
    ID = _id;
  }

  bool
  operator==(const PortID& rhs) const {
    return this->ID == rhs.ID;
  }
};

using PortEntry = std::pair<Node*, PortID>;

struct DataPort {
  std::list<PortEntry> data_input_port;
  std::list<PortEntry> data_output_port;
};

struct ControlPort {
  std::list<PortEntry> control_input_port;
  std::list<PortEntry> control_output_port;
};

struct MemoryPort {
  std::list<PortEntry> memory_req_port;
  std::list<PortEntry> memory_resp_port;
};



struct NodeInfo {
  uint32_t ID;
  std::string Name;

  NodeInfo(uint32_t _id, std::string _n) : ID(_id), Name(_n){};
  NodeInfo(std::string _n, uint32_t _id) : ID(_id), Name(_n){};
};


class Node{

public:

  enum NodeType {
    FunctionArgTy = 0,
    Operation,
    ComputeNodeTy,
    MemoryUnitTy,
    LoadOpTy,
    StoreOpTy,
    ConstTy,

    ExecutionBlockNodeTy,
    SuperNodeTy,
    LoopNodeTy,
    OperationNodeTy,
    
    GlobalValueTy,
    
    ConstFPTy,
    
    StackUnitTy,
    FloatingPointTy,
    ContainerTy,
    UnkonwTy

  };


private:
  // Type of the Node
  NodeType node_type;

  // Node information
  NodeInfo info;

  // List of data ports
  DataPort port_data;

  // List of Control ports
  ControlPort port_control;

  // Memory ports
  MemoryPort read_port_data;
  MemoryPort write_port_data;

public:
  Node(NodeType _nt, NodeInfo _ni) : node_type(_nt), info(_ni) {}

  NodeInfo
  getInfo() {
    return this->info;
  }
  uint32_t //op_list.size()
  getID() {
    return this->info.ID;
  }
  std::string //int_add_...
  getName() {
    return this->info.Name;
  }

  NodeType getType() const {
    return this->node_type;
  }
  
  std::vector<uint32_t> conflict_output_index;
  std::vector<uint32_t> conflict_input_index;

  PortID returnDataInputPortIndex(Node*);
  PortID returnControlInputPortIndex(Node*);
  PortID returnMemoryReadInputPortIndex(Node*);
  PortID returnMemoryWriteInputPortIndex(Node*);

  PortID returnDataOutputPortIndex(Node*);
  PortID returnControlOutputPortIndex(Node*);
  PortID returnMemoryReadOutputPortIndex(Node*);
  PortID returnMemoryWriteOutputPortIndex(Node*);

  // Node *returnDataOutputPortNode(uint32_t index);
  Node* returnControlOutputPortNode(uint32_t index);
  
  PortID addReadMemoryReqPort(Node*);
  PortID addReadMemoryRespPort(Node*);

  PortID addWriteMemoryReqPort(Node*);
  PortID addWriteMemoryRespPort(Node*);

  virtual std::string printDefinition(PrintType) {
    return this->info.Name + std::string(" Definition is Not defined!");
  }
  virtual std::string
  printInputEnable(PrintType, uint32_t) {
    return this->info.Name + std::string(" Input EnableInput with ID Not defined!");
  }
  virtual std::string
  printInputEnable(PrintType, std::pair<Node*, PortID>) {
    return this->info.Name + std::string(" Input1 EnableInput with ID Not defined!");
  }
  virtual std::string
  printInputEnable(PrintType) {
    return this->info.Name + std::string(" EnableInput Not defined!");
  }
  virtual std::string
  printOutputEnable(PrintType, std::pair<Node*, PortID>) {
    return this->info.Name + std::string(" OutpuEnable with ID Not defined!");
  }
  virtual std::string
  printOutputEnable(PrintType) {
    return this->info.Name + std::string(" EnableOutput Not defined!");
  }
  virtual std::string
  printOutputEnable(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> EnableOutput with ID Not defined!");
  }
  virtual std::string
  printInputData(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> DataInput with ID Not defined!");
  }
  virtual std::string
  printOutputData(PrintType) {
    return this->info.Name + std::string(" -> DataOutput Not defined!");
  }
  virtual std::string
  printOutputData(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> DataOutput with ID Not defined!");
  }
  virtual std::string
  printMemReadInput(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> MemInput with ID Not defined!");
  }

  virtual std::string
  printMemReadOutput(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> MemOutput with ID Not defined!");
  }

  virtual std::string
  printMemWriteInput(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> MemInput with ID Not defined!");
  }

  virtual std::string
  printMemWriteOutput(PrintType, uint32_t) {
    return this->info.Name + std::string(" -> MemOutput with ID Not defined!");
  }
  PortID addControlInputPort(Node* node);
  PortID addControlOutputPort(Node* node);

  PortID addControlInputPort(Node* node, uint32_t id);
  PortID addControlOutputPort(Node* node, uint32_t id);

  PortID addDataInputPort(Node* node);
  PortID addDataOutputPort(Node* node);

  PortID addDataInputPort(Node* node, uint32_t id);
  PortID addDataOutputPort(Node* node, uint32_t id);

  std::list<PortEntry> findControlInputNodeList(Node*);
  std::list<PortEntry> findControlOutputNodeList(Node*);
  std::list<PortEntry>::iterator findControlOutputNode(Node*);
  uint32_t
  numDataInputPort() {
    return port_data.data_input_port.size();
  }
  uint32_t
  numDataOutputPort() {
    return port_data.data_output_port.size();
  }
  uint32_t
  numControlInputPort() {
    return port_control.control_input_port.size();
  }
  uint32_t
  numControlOutputPort() {
    return port_control.control_output_port.size();
  }
  uint32_t
  numReadMemReqPort() {
    return read_port_data.memory_req_port.size();
  }
  uint32_t
  numReadMemRespPort() {
    return read_port_data.memory_resp_port.size();
  }
  uint32_t
  numWriteMemReqPort() {
    return write_port_data.memory_req_port.size();
  }
  uint32_t
  numWriteMemRespPort() {
    return write_port_data.memory_resp_port.size();
  }
  uint32_t
  numReadDataInputPort() {
    return read_port_data.memory_req_port.size();
  }
  uint32_t
  numReadDataOutputPort() {
    return read_port_data.memory_resp_port.size();
  }
  uint32_t
  numWriteDataInputPort() {
    return write_port_data.memory_req_port.size();
  }
  uint32_t
  numWriteDataOutputPort() {
    return write_port_data.memory_resp_port.size();
  }

  //input
  auto
  inputDataport_begin() {
    return this->port_data.data_input_port.begin();
  }
  auto
  inputDataport_end() {
    return this->port_data.data_input_port.end();
  }

  //output
  auto
  outputDataport_begin() {
    return this->port_data.data_output_port.begin();
  }
  auto
  outputDataport_end() {
    return this->port_data.data_output_port.end();
  }

  //mem
  auto
  read_req_begin() {
    return this->read_port_data.memory_req_port.begin();
  }
  auto
  read_req_end() {
    return this->read_port_data.memory_req_port.end();
  }
  auto
  write_req_begin() {
    return this->write_port_data.memory_req_port.begin();
  }

  auto
  write_req_end() {
    return this->write_port_data.memory_req_port.end();
  }
  auto
  read_resp_begin() {
    return this->read_port_data.memory_resp_port.begin();
  }
  auto
  read_resp_end() {
    return this->read_port_data.memory_resp_port.end();
  }
  auto
  write_resp_begin() {
    return this->write_port_data.memory_resp_port.begin();
  }
  auto
  write_resp_end() {
    return this->write_port_data.memory_resp_port.end();
  }
  auto
  inputControl_begin() {
    return this->port_control.control_input_port.begin();
  }
  auto
  inputControl_end() {
    return this->port_control.control_input_port.end();
  }
  auto
  outputControl_begin() {
    return this->port_control.control_output_port.begin();
  }
  auto
  outputControl_end() {
    return this->port_control.control_output_port.end();
  }
  /**
   * Adding a node to a specific index of control input port
   */
  void
  addControlInputPortIndex(Node* _n, uint32_t _id) {
    port_control.control_input_port.push_back(std::make_pair(_n, _id));
  }

  /**
   * Adding a node to a specific index of control output port
   */
  void
  addControlOutputPortIndex(Node* _n, uint32_t _id) {
    port_control.control_output_port.push_back(std::make_pair(_n, _id));
  }

  bool checkOutputConfict(uint32_t _port_id) {
    return std::find(conflict_output_index.begin(), conflict_output_index.end(), _port_id) != conflict_output_index.end();
  }
  bool checkInputConfict(uint32_t _port_id) {
    return std::find(conflict_input_index.begin(), conflict_input_index.end(), _port_id) != conflict_input_index.end();
  }
};


class OperationNode : public Node {

public:
  enum OperationType {
    BinaryType = 0,
    AllocaType,
    ReturnType,
    LoadType,
    LSType,
    StateBranchType,
    BitCastType,
    MergeType,
    SelectType,
    StoreType,
    CmpType,
    AddressGenType,
    InductionVarType
    
  };

private:
  OperationType op_type;
  DataType data_type;
  mlir::Operation* parent_op;
  OpCode op_code;
  ExecutionBlockNode* parentNode;

public:

  explicit OperationNode(NodeInfo _ni, OperationType _optype, DataType datatype, mlir::Operation* operation, OpCode opcode)
    : Node(Node::ComputeNodeTy, _ni), 
      op_type(_optype),
      data_type(datatype),
      parent_op(operation),
      op_code(opcode)
    {}

  explicit OperationNode(NodeInfo _ni, OperationType _optype,  mlir::Operation* operation)
    : Node(Node::ComputeNodeTy, _ni), 
      op_type(_optype),
      parent_op(operation),
      op_code(OpCode::other)
    {}

  explicit OperationNode(NodeInfo _ni, OperationType _optype, DataType datatype, mlir::Operation* operation)
    : Node(Node::ComputeNodeTy, _ni), 
      op_type(_optype),
      data_type(datatype),
      parent_op(operation),
      op_code(OpCode::other)
    {}

  mlir::Operation* getOperation();

  
  DataType getDataType() const {
    return this->data_type;
  }
  OperationType getOperationType() const {
    return this->op_type;
  }
  OpCode
  geOpCodeEnum() const {
    return this->op_code;
  }
  std::string getOpcodeName()  {
    return opCodeToString(this->geOpCodeEnum());
  }
  mlir::Operation* getOp() const {
    return this->parent_op;
  }
  bool isMemrefType() const {
    return this->data_type == DataType::MemrefType;
  }
  bool isIntegerType() const {
    return this->data_type == DataType::IntegerType;
  }
  bool isFloatType() const {
    return this->data_type == DataType::FloatType;
  }

  static bool classof(const Node* T) {
    return T->getType() == NodeType::OperationNodeTy;
  }

  void setParentNode(ExecutionBlockNode* node) {
    this->parentNode = node;
  }

  virtual std::string printDefinition(PrintType) override {
    return std::string("Not defined operations\n");
  }
};

class ConstNode : public Node {
private:
  mlir::Operation* parent_op;

  int value =0 ;
  double value_f =0.0;
  int id;
  bool isInt = true;
public:

  ConstNode(NodeInfo _ni, mlir::Operation* op, bool isint, int _id)
    : Node(Node::ConstTy, _ni), parent_op(op) ,isInt(isint) ,id(_id){
    if (isInt)
      value = dyn_cast<arith::ConstantOp>(op).getValue().cast<IntegerAttr>().getInt();
    else
      value_f = dyn_cast<arith::ConstantOp>(op).getValue().cast<FloatAttr>().getValueAsDouble();
  }

  int getConstID(){
    return this->id;
  }
  static bool
  classof(const Node* T) {
    return T->getType() == Node::ConstTy;
  }

  int getValue() {
    return value;
  }

  mlir::Operation* getConstantOp(){
    return this->parent_op;
  }
  virtual std::string printDefinition(PrintType) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printInputEnable(PrintType) override;
};


class BitCastNode : public OperationNode {
private:
  mlir::Operation* parent_op;


public:

  BitCastNode(NodeInfo _ni, mlir::Operation* op)
    : OperationNode(_ni, OperationType::BitCastType, op), parent_op(op)
    {}

 
  static bool
  classof(const Node* T) {
    return T->getType() == OperationType::BitCastType;
  }


  mlir::Operation* getParentOp(){
    return this->parent_op;
  }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
};

class ExecutionBlockNode : public Node {

public: 
  bool hasPhi = false;
  using ConstNodeList = std::list<ConstNode*>;
  using MergeNodeList = std::list<MergeNode*>;
  
  
private:

  std::list<OperationNode* > operation_node_list;
  dataflow::ExecutionBlockOp relatedOp;
  bool mask;
  mlir::Operation* control_op;
  ConstNodeList const_list;
  MergeNodeList merge_list;
  
public:
  explicit ExecutionBlockNode(NodeInfo nf, dataflow::ExecutionBlockOp op, bool ismask)
    : Node(Node::ExecutionBlockNodeTy, nf),
      relatedOp(op),
      mask(ismask) {}

  dataflow::ExecutionBlockOp getRelatedOp(){
    return relatedOp;
  }
  void setControlOp(mlir::Operation* op){
    control_op = op;
  }
  bool getNodeType() {
    return this->mask;
  }
  uint32_t getNumMerge() const {
    return this->merge_list.size();
  }
  void addconstNode(ConstNode* node);
  void addOperationNode(OperationNode* node);
  void addMergeNode(MergeNode* node);

  auto  const_begin() const {
    return this->const_list.begin();
  }
  auto  const_end() const {
    return this->const_list.end();
  }

  auto  operation_node_begin() const {
    return this->operation_node_list.begin();
  }
  auto  operation_node_end() const {
    return this->operation_node_list.end();
  }
  void
  setNodeType(bool _t) {
    this->mask = _t;
  }
  auto merge_begin() const {
    return this->merge_list.begin();
  }
  auto merge_end() const {
    return this->merge_list.end();
  }
  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printInputEnable(PrintType, std::pair<Node*, PortID>) override;
  virtual std::string printOutputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputEnable(PrintType, std::pair<Node*, PortID>) override;
  virtual std::string printMaskOutput(PrintType pt, uint32_t _id);
};


class MergeNode : public OperationNode {
private:
  Node*  mask_block;
  mlir::Operation* parent_op;
  mlir::Operation* block_op;
public:
  MergeNode(NodeInfo _ni, mlir::Operation* op = nullptr, mlir::Operation* block = nullptr)
    : OperationNode(_ni, OperationType::MergeType, op), parent_op(op), block_op(block) {}


  dataflow::ExecutionBlockOp getMaskOp() const {
    return dyn_cast<dataflow::ExecutionBlockOp>(block_op);
  }
    
  mlir::Operation* getParentOp(){
    return this->parent_op;
  }
  static bool
  classof(const OperationNode* T) {
    return T->getOperationType() == OperationType::MergeType;
  }


  void setParentNode(Node* _parent) {
    this->mask_block = _parent;
  }
  Node* getParentNode(){
    return this->mask_block;
  }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printMaskInput(PrintType);
};


class ArgumentNode : public Node {

public:
  enum ArgumentType { LiveIn = 0, LiveOut, LoopLiveIn, LoopLiveOut, CarryDependency};

protected:
 

private:
  ArgumentType argType;
  DataType dataType;
  Node* parentNode;
  Value argumentValue;
  ContainerNode* parent_call_node;

public:
  explicit ArgumentNode(NodeInfo ni,
                        ArgumentType arg_type,
                        DataType d_type,
                        Value arg      = nullptr,
                        ContainerNode* call_node = nullptr,
                        Node* arg_node = nullptr)
    : Node(Node::FunctionArgTy, ni),
      argType(arg_type),
      dataType(d_type),
      argumentValue(arg),
      parent_call_node(call_node),
      parentNode(arg_node) {}


  const Value getArgumentValue(){
    return this->argumentValue;
  }

  void
  setParentNode(Node* node) {
    this->parentNode = node;
  }

  Node* getParentNode() {
    return this->parentNode;
  }

  static bool
  classof(const Node* T) {
    return T->getType() == NodeType::FunctionArgTy;
  }
  // Define classof function so that we can use dyn_cast function
  // static bool
  // classof(const Node* T) {
  //   return T->getType() == Node::FunctionArgTy;
  // }
  // static bool
  // classof(const Node* T) {
  //   return isa<ArgumentNode>(T) && classof(cast<ArgumentNode>(T));
  // }


  ArgumentType getArgType() {
    return this->argType;
  }
  DataType getDataArgType() {
    return this->dataType;
  }

  
  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
};


class ContainerNode : public Node {

public:
  enum ContainType { LoopNodeTy = 0, FuncCallTy };
  using RegisterList = std::list<std::shared_ptr<ArgumentNode>>;

protected:
  ContainType con_type;
  
  // ptrs and vals are fore split call
  RegisterList live_in_mems;
  RegisterList live_in_vals;

  

  // live_in is for loop nodes
  RegisterList live_in_sets;

  RegisterList live_out_sets;
  RegisterList carry_depen_sets;



public:
  explicit ContainerNode(NodeInfo _nf)
    : Node(Node::ContainerTy, _nf), con_type(ContainType::LoopNodeTy) {}

  explicit ContainerNode(NodeInfo _nf, ContainType cn_type)
    : Node(Node::ContainerTy, _nf), con_type(cn_type) {}


  uint32_t getContainerType() const {
    return this->con_type;
  }

 
  ArgumentNode* insertLiveInArgument(Value Value, ArgumentNode::ArgumentType Type);
  ArgumentNode* insertLiveOutArgument(Value Value,
                                      ArgumentNode::ArgumentType Type);
  ArgumentNode* insertCarryDepenArgument(Value Value,
                                         ArgumentNode::ArgumentType argtype);

  Node* findLiveInNode(Value val);
  Node* findLiveOutNode(Value val);
  Node* findCarryDepenNode(Value val);

  uint32_t findLiveInArgumentIndex(ArgumentNode*);
  uint32_t findLiveOutArgumentIndex(ArgumentNode*);
  uint32_t findCarryDepenArgumentIndex(ArgumentNode*);


  uint32_t numLiveInArgList(ArgumentNode::ArgumentType type,
                            DataType dtype);
  uint32_t numLiveOutArgList(ArgumentNode::ArgumentType type);
  uint32_t numCarryDepenArgList(ArgumentNode::ArgumentType type);

  auto   live_in_sets_begin() {
    return this->live_in_sets.begin();
  }
  auto  live_in_sets_end() {
    return this->live_in_sets.end();
  }
  auto   live_out_sets_begin() {
    return this->live_out_sets.begin();
  }
  auto  live_out_sets_end() {
    return this->live_out_sets.end();
  }
  auto  live_in_mems_begin() {
    return this->live_in_mems.begin();
  }
  auto  live_in_mems_end() {
    return this->live_in_mems.end();
  }

  auto  live_in_vals_begin() {
    return this->live_in_vals.begin();
  }
  auto  live_in_vals_end() {
    return this->live_in_vals.end();
  }
  auto  carry_depen_sets_begin() {
    return this->carry_depen_sets.begin();
  }
  auto  carry_depen_sets_end() {
    return this->carry_depen_sets.end();
  }
  RegisterList getLiveInSets() {
    return this->live_in_sets;
  }
  RegisterList getLiveOutSets() {
    return this->live_out_sets;
  }
  RegisterList getCarryDepenSets() {
    return this->carry_depen_sets;
  }
};


/**
 * Implementing fine-grained argument calls.
 */
class FineArgCallNode : public ContainerNode {
public:
  enum ArgType { Ptrs = 0, Vals };

private:
  std::map<ArgumentNode*, ArgType> arg_types;

public:
  explicit FineArgCallNode(NodeInfo _nf) : ContainerNode(_nf, ContainerNode::ContainType::FuncCallTy) {}



  // uint32_t findLiveOutIndex(ArgumentNode*)          = delete;

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printOutputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
};



class ComputeOperationNode : public OperationNode {

public:
  

  explicit ComputeOperationNode(NodeInfo _ni, OperationType optype, DataType datatype, mlir::Operation* operation, OpCode opcode) 
    : OperationNode(_ni, OperationType::BinaryType, datatype, operation, opcode)
    {}

  // Overloading isa<>, dyn_cast from llvm
  static bool
  classof(const OperationNode* I) {
    return I->getOperationType() == OperationType::BinaryType;
  }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }
  
  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
};


class CmpNode : public OperationNode {

public:
  

  explicit CmpNode(NodeInfo _ni, OperationType optype, DataType datatype, mlir::Operation* operation, OpCode opcode) 
    : OperationNode(_ni, OperationType::CmpType, datatype, operation, opcode)
    {}

  // Overloading isa<>, dyn_cast from llvm
  static bool
  classof(const OperationNode* I) {
    return I->getOperationType() == OperationType::CmpType;
  }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }
  
  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
};





class AllocaNode : public OperationNode {


private:
  
  uint32_t size;
  uint32_t num_byte;

public:
  

  AllocaNode(NodeInfo _ni, OperationType type, mlir::Operation* operation) 
    : OperationNode(_ni, type, operation),
      size(1), 
      num_byte(0)
    {}

  AllocaNode(NodeInfo _ni, OperationType type, DataType data_type, mlir::Operation* operation,
             uint32_t _size, uint32_t _num_byte = 1) 
    : OperationNode(_ni, type, data_type, operation),
      size(_size),
      num_byte(_num_byte)
    {}

  uint32_t getNumByte() {
    return this->num_byte;
  }
  void setNumByte(uint32_t _n) {
    this->num_byte = _n;
  }

  uint32_t getSize() {
    return this->size;
  }
  void setSize(uint32_t _n) {
    this->size = _n;
  }

  static bool
  classof(const OperationNode* I) {
    return I->getOperationType() == OperationType::AllocaType;//TODO
  }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }

  virtual std::string printDefinition(PrintType) override;

};

class ReturnNode : public OperationNode {

public:
  ReturnNode(NodeInfo _ni, OperationType type, mlir::Operation* operation)
    : OperationNode(_ni, type, operation) 
    {}

  static bool
  classof(const OperationNode* I) {
    return I->getOperationType() == OperationType::ReturnType;//TODO
  }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
};




class MemoryNode : public Node {


public:
  enum memType  {
    memory = 0,
    scratchpad
  };

  AllocaNode* alloca_node;
  int64_t size;
  uint32_t num_byte;

private:

  memType mem_type;

public:
  explicit MemoryNode(NodeInfo _nf, memType memtype) 
  : Node(Node::MemoryUnitTy, _nf), mem_type(memtype)
  {}

  explicit MemoryNode(NodeInfo _nf, memType memtype,
                          AllocaNode* alloca,
                          uint32_t mem_size,
                          uint32_t mem_byte)
    : Node(Node::MemoryUnitTy, _nf),
      mem_type(memtype),
      alloca_node(alloca),
      size(mem_size),
      num_byte(mem_byte) {}


  AllocaNode* getAllocaNode() {
    return this->alloca_node;
  }

  uint32_t getMemSize() {
    return this->size;
  }
  void setMemSize(int64_t num) {
    this->size = num;
  }
  uint32_t getMemByte() {
    return this->num_byte;
  }

  memType getUnitType() {
    return this->mem_type;
  }

  void setType(memType type){
    this->mem_type = type;
  }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printMemReadInput(PrintType, uint32_t) override;
  virtual std::string printMemReadOutput(PrintType, uint32_t) override;
  virtual std::string printMemWriteInput(PrintType, uint32_t) override;
  virtual std::string printMemWriteOutput(PrintType, uint32_t) override;
};



/**
 * LoopNode contains all the instructions and useful information about the loops
 */
class LoopNode : public ContainerNode {
public:
  enum PortType {
    Active_Loop_Start = 0,
    Active_Loop_Back,
    Enable,
    LoopBack,
    LoopFinish,
    LoopExit
  };

  using PortList = std::list<std::pair<Node*, PortType>>;

private:
  std::list<std::pair<Node*, PortType>> port_type;
  LoopNode* parent_loop;
  std::list<OperationNode*> instruction_list;

  OperationNode* induction_variable;
  // std::list<SuperNode*> basic_block_list;
  std::list<OperationNode*> ending_instructions;

  // SuperNode* head_node;
  // SuperNode* latch_node;
  // std::list<SuperNode*> exit_node;
  // OperationNode* induction_variable;
  
  // Set auxiliary information for ouput nodes
  PortEntry activate_loop_start;
  PortEntry activate_loop_back;
  std::vector<PortEntry> loop_exits;

  bool outer_loop;
  bool setStart = false;
  // // Restrict the access to these two functions
  // using Node::addControlInputPort;
  // using Node::addControlOutputPort;

public:
  explicit LoopNode(NodeInfo _nf)
    : ContainerNode(_nf, ContainerNode::LoopNodeTy),
      parent_loop(nullptr),
      // head_node(nullptr),
      // latch_node(nullptr),
      // exit_node(std::list<SuperNode*>()),
      outer_loop(false) {
    // Set the size of control input prot to at least two
    // resizeControlInputPort(LOOPCONTROL);
    // resizeControlOutputPort(LOOPCONTROL);
  }

  int activate_num = 0;
  /**
   * Print functions
   */
  virtual std::string printDefinition(PrintType) override;
  // virtual std::string printOutputEnable(PrintType) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputEnable(PrintType, PortEntry) override;

  void setIndeuctionVar(OperationNode* var) {
    this->induction_variable = var;
  }
  auto getInductionVar() {
    return this->induction_variable;
  }

  void setControlLoop(Node* _n) {
    addControlInputPort(_n);
    port_type.push_back(std::make_pair(_n, PortType::Enable));
  }
  void setActiveOutputLoopSignal(Node* _n) {
    auto _port_info = PortID(this->numControlOutputPort());
    addControlOutputPort(_n);
    port_type.push_back(std::make_pair(_n, PortType::Active_Loop_Start));

    // Seting activate_loop_start
    this->activate_loop_start = std::make_pair(_n, _port_info);
  }
  void setActiveBackSignal(Node* _n) {
    auto _port_info = PortID(this->numControlOutputPort());
    addControlOutputPort(_n);
    port_type.push_back(std::make_pair(_n, PortType::Active_Loop_Back));


    this->activate_loop_back = std::make_pair(_n, _port_info);
  }
  void setActiveExitSignal(Node* _n) {
    auto _port_info = PortID(this->numControlOutputPort());
    addControlOutputPort(_n);
    port_type.push_back(std::make_pair(_n, PortType::LoopExit));

    // Seting activate_loop_start
    this->loop_exits.push_back(std::make_pair(_n, _port_info));
  }
  void setSignal(){
    this->setStart = true;
  }
  bool getSignal(){
    return this->setStart;
  }
  /**
   * Make sure that loop end enable signal is always fix to index 1
   */
  void
  setLoopEndEnable(Node* _n) {
    addControlOutputPort(_n);
    port_type.push_back(std::make_pair(_n, PortType::LoopFinish));
  }
};




class LSNode : public OperationNode {

public:
  enum opmemType {
    load = 0,
    store 
  }; 

  bool isStore = false;
private:
  
  MemoryNode* mem_node;
  uint32_t route_id;
  bool ground;
  opmemType op_type;

public:
  

  explicit LSNode(NodeInfo _ni, OperationType optype, opmemType memtype, mlir::Operation* operation,
    MemoryNode* _node = nullptr, uint32_t _id = 0) 
    : OperationNode(_ni, optype, operation),
      op_type(memtype),
      mem_node(_node),
      route_id(_id){
      if(memtype == opmemType::store){
        isStore = true;
      }
    }

  explicit LSNode(NodeInfo _ni, OperationType optype, DataType _type, opmemType memtype, mlir::Operation* operation,
    MemoryNode* _node = nullptr, uint32_t _id = 0) 
    : OperationNode(_ni, optype, _type, operation),
      op_type(memtype),
      mem_node(_node),
      route_id(_id){
      if(memtype == opmemType::store){
        isStore = true;
      }
    }
  
  void
  setMemoryUnit(MemoryNode* _node) {
    this->mem_node = _node;
  }
  void
  setRouteID(uint32_t _id) {
    this->route_id = _id;
  }
  bool
  isMemStore() {
    return this->isStore;
  }
  auto
  getRouteID() {
    return this->route_id;
  }
  
  static bool
  classof(const OperationNode* I) {
    return I->getOperationType() == OperationType::LSType;//TODO
  }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;

  virtual std::string printMemReadInput(PrintType, uint32_t) override;
  virtual std::string printMemReadOutput(PrintType, uint32_t) override;

  virtual std::string printOutputEnable(PrintType, uint32_t) override;
  virtual std::string printOutputEnable(PrintType) override;

  virtual std::string printMemWriteInput(PrintType, uint32_t) override;
  virtual std::string printMemWriteOutput(PrintType, uint32_t) override;

  std::string printGround(PrintType);
};

class StateBranchNode : public OperationNode {
private:
  bool ending_loop;

  dataflow::StateOp relatedOp;

  dataflow::EnableOp controlOp;
  
public:
  enum PredicateResult { True = 0, False };
  using PrintedNode = std::pair<Node*, PredicateResult>;

  std::list<std::pair<Node*, PredicateResult>> output_predicate;
  std::map<PrintedNode, uint32_t> printed_predicate;
  // StateBranchNode(NodeInfo ni, mlir::Operation* op)
  //   : OperationNode(ni, OperationType::StateBranchType, nullptr), 
  //     relatedOp(op),
  //     ending_loop(false) {}

  StateBranchNode(NodeInfo ni, dataflow::StateOp op)
    : OperationNode(ni, OperationType::StateBranchType, op.getOperation()), 
      relatedOp(op) {}
  
  StateBranchNode(NodeInfo ni, dataflow::EnableOp op)
    : OperationNode(ni, OperationType::StateBranchType, op.getOperation()), 
      controlOp(op) {}
  // StateBranchNode(NodeInfo _ni, bool _loop, llvm::BranchInst* _ins = nullptr)
  //   : OperationNode(_ni, InstType::BranchInstructionTy, _ins), ending_loop(_loop) {}

  
  static bool
  classof(const Node* T) {
    return T->getType() == OperationType::StateBranchType;
  }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }
  dataflow::StateOp getRelatedOp(){
    return relatedOp;
  }
  dataflow::EnableOp getControlOp(){
    return controlOp;
  }

  void addTrueBranch(Node* n) {
    this->output_predicate.push_back(std::make_pair(n, PredicateResult::True));
    this->addControlOutputPort(n);
  }
  void addFalseBranch(Node* n) {
    this->output_predicate.push_back(std::make_pair(n, PredicateResult::False));
    this->addControlOutputPort(n);
  }

  void
  setEndingLoopBranch() {
    ending_loop = true;
  }
  bool
  getEndingLoopBranch() {
    return ending_loop;
  }
  // /// replace two nodes form the control output container
  // virtual void replaceControlOutputNode(Node* src, Node* tar) override;

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printOutputEnable(PrintType, uint32_t) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
};

class SelectNode : public OperationNode {
private:
  // SuperNode* mask_node;
  dataflow::SelectOp relatedOp;
  arith::SelectOp relatedArithOp;

  bool dataSelect = false;
public:
  SelectNode(NodeInfo ni, dataflow::SelectOp op)
    : OperationNode(ni, OperationType::SelectType, op.getOperation()), 
      relatedOp(op) {}
  SelectNode(NodeInfo ni, dataflow::SelectOp op, bool data = false)
    : OperationNode(ni, OperationType::SelectType, op.getOperation()), 
      relatedOp(op),
      dataSelect(data) {}
  SelectNode(NodeInfo ni, arith::SelectOp op)
    : OperationNode(ni, OperationType::SelectType, op.getOperation()), 
      relatedArithOp(op) {}

  // SuperNode*
  // getMaskNode() const {
  //   return mask_node;
  // }
  dataflow::SelectOp getRelatedOp(){
    return relatedOp;
  }
  arith::SelectOp getArithRelatedOp(){
    return relatedArithOp;
  }
  void setDataSelect(bool value){
    this->dataSelect = value;
  }
  // static bool
  // classof(const OperationNode* T) {
  //   return T->getOpCode() == OperationNode::SelectInstructionTy;
  // }
  // static bool
  // classof(const Node* T) {
  //   return isa<OperationNode>(T) && classof(cast<OperationNode>(T));
  // }

  // void
  // setParentNode(SuperNode* _parent) {
  //   this->mask_node = _parent;
  // }

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
};

class AddressGenNode : public OperationNode {
private:
  // GepInfo gep_info;
  dataflow::AddressOp relatedOp;
public:
  AddressGenNode(NodeInfo _ni, dataflow::AddressOp op = nullptr)
    : OperationNode(_ni, OperationType::AddressGenType, op.getOperation()), 
      relatedOp(op) {}
    

  // AddressGenNode(NodeInfo _ni, dataflow::AddressOp op = nullptr, )
  //   : OperationNode(_ni, OperationType::AddressGenType, op.getOperation()),
    
  //    {}
  dataflow::AddressOp getRelatedOp(){
    return relatedOp;
  }
  static bool
  classof(const Node* T) {
    return T->getType() == OperationNode::OperationType::AddressGenType;
  }
 

  virtual std::string printDefinition(PrintType) override;
  virtual std::string printInputEnable(PrintType) override;
  virtual std::string printInputEnable(PrintType, uint32_t) override;
  virtual std::string printInputData(PrintType, uint32_t) override;
  virtual std::string printOutputData(PrintType, uint32_t) override;
};

}//heteacc
}//mlir



#endif // GRAPH_STRUCTURE_H