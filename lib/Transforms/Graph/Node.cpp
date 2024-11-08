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


using namespace mlir;
using namespace heteacc;


DataType heteacc::isDataType(Value arg){

    Type type = arg.getType();
    if(type.isa<mlir::IntegerType>()){
      return DataType::IntegerType;
//
    } else if(type.isa<mlir::MemRefType>()){
      return  DataType::MemrefType;

    } else if(type.isa<mlir::FloatType>()){
      return DataType::FloatType;
      
    } else if(type.isa<mlir::VectorType>()){
       assert(!"Unsuppported Type.");
       return DataType::UknownType;
    } else{
       assert(!"Unsuppported Type.");
       return DataType::UknownType;
    }
}

std::string heteacc::opCodeToString(heteacc::OpCode code) {
    static const std::unordered_map<heteacc::OpCode, std::string> opCodeMap = {
        {heteacc::OpCode::add, "Add"},
        {heteacc::OpCode::mul, "Mul"},
        {heteacc::OpCode::sub, "Sub"},
        {heteacc::OpCode::eq, "eq"},
        {heteacc::OpCode::ne, "ne"},
        {heteacc::OpCode::slt, "slt"},
        // {heteacc::OpCode::sle, "Pass"},
        {heteacc::OpCode::sgt, "sgt"},
        {heteacc::OpCode::sge, "gte"},
        {heteacc::OpCode::ult, "ult"},
        // {heteacc::OpCode::ule, "Pass"},
        {heteacc::OpCode::ugt, "ugt"},
        // {heteacc::OpCode::uge, "Pass"},
        {heteacc::OpCode::shr, "ashr"},
        {heteacc::OpCode::shl, "shl"},
        {heteacc::OpCode::lshr, "lshr"},
        {heteacc::OpCode::div, "udiv"},
        {heteacc::OpCode::andi, "and"},
        {heteacc::OpCode::ori, "or"},
        {heteacc::OpCode::other, "other"}
    };

    auto it = opCodeMap.find(code);
    if (it != opCodeMap.end()) {
        return (it->second);
    } else {
        return "unknown";
    }
}

std::list<PortEntry> Node::findControlInputNodeList(Node* _node) {
  std::list<PortEntry> result;
  copy_if(this->port_control.control_input_port.begin(),
          this->port_control.control_input_port.end(),
          std::back_inserter(result),
          [_node](auto& arg) -> bool { return arg.first == _node; });
  return result;
}

std::list<PortEntry> Node::findControlOutputNodeList(Node* _node) {
  std::list<PortEntry> result;
  copy_if(this->port_control.control_output_port.begin(),
          this->port_control.control_output_port.end(),
          std::back_inserter(result),
          [_node](auto& arg) -> bool { return arg.first == _node; });

  return result;
}
std::list<PortEntry>::iterator
Node::findControlOutputNode(Node* _node) {
  return find_if(this->port_control.control_output_port.begin(),
                 this->port_control.control_output_port.end(),
                 [_node](auto& arg) -> bool { return arg.first == _node; });
}

//===----------------------------------------------------------------------===//
//                            ContainerNode Class
//===----------------------------------------------------------------------===//

ArgumentNode* ContainerNode::insertLiveInArgument(Value val, ArgumentNode::ArgumentType type){

    auto valType = DataType::IntegerType;
    if(val.getType().isa<mlir::MemRefType>()){
      valType = DataType::MemrefType;
    } else if (val.getType().isa<mlir::FloatType>()) {
      valType = DataType::FloatType;
    } else if (val.getType().isa<mlir::IntegerType>()) {
      valType = DataType::IntegerType;
    } else if (val.getType().isa<mlir::IndexType>()){
      valType = DataType::IntegerType;
    } else {
      assert(!"Unsuppported Type.");  
    }
    switch (con_type){
        case ContainerNode::ContainType::LoopNodeTy:{
            auto ff = std::find_if(this->live_in_sets_begin(), this->live_in_sets_end(), [&val](auto& arg) -> bool {
              return arg.get()->getArgumentValue() == val;
            });
            if (ff == this->live_in_sets_end()) {
              this->live_in_sets.push_back(std::make_unique<ArgumentNode>(
                  NodeInfo(this->live_in_sets.size(), "live_in_" + std::to_string(this->live_in_sets.size())),
                  type,
                  valType,
                  val,
                  this));

              ff = std::find_if(this->live_in_sets_begin(), this->live_in_sets_end(), [&val](auto& arg) -> bool {
                return arg.get()->getArgumentValue() == val;
              });
            }
            return ff->get();
        }
        case ContainerNode::ContainType::FuncCallTy:{
            if(val.getType().isa<mlir::MemRefType>()){
                auto find = std::find_if(
                    this->live_in_mems_begin(), this->live_in_mems_end(), [&val](auto& arg) -> bool {
                      return arg.get()->getArgumentValue() == val;
                    });
                if(find == this->live_in_mems_end()){
                    this->live_in_mems.push_back(std::make_unique<ArgumentNode>(NodeInfo(this->live_in_mems.size(), "arg_mem" + std::to_string(this->live_in_mems.size())),
                    type, DataType::MemrefType, val, this                    
                    ));
                }
                find = std::find_if(
                    this->live_in_mems_begin(), this->live_in_mems_end(), [&val](auto& arg) -> bool {
                        return arg.get()->getArgumentValue() == val;
                    });
                return find->get();
            } else if(val.getType().isa<mlir::FloatType>() || val.getType().isa<mlir::IntegerType>() || val.getType().isa<mlir::IndexType>()){
                auto ff = std::find_if(
                this->live_in_vals_begin(), this->live_in_vals_end(), [&val](auto& arg) -> bool {
                  return arg.get()->getArgumentValue() == val;
                });
                if (ff == this->live_in_vals_end()) {
                  auto tempType = val.getType().isa<mlir::FloatType>() ? DataType::FloatType : DataType::IntegerType;
                  this->live_in_vals.push_back(std::make_unique<ArgumentNode>(NodeInfo(this->live_in_vals.size(), "arg_val"), 
                      type, tempType, val, this  ));
                  ff = std::find_if(
                      this->live_in_vals_begin(), this->live_in_vals_end(), [&val](auto& arg) -> bool {
                        return arg.get()->getArgumentValue() == val;
                      });
                }
                return ff->get();

            } else {
                assert(!"Unsuppported Type.");
            }
        }
        default: assert(!"Container type is unkonw!");
    }
}


ArgumentNode*
ContainerNode::insertLiveOutArgument(Value val,
                                     ArgumentNode::ArgumentType type) {
  auto valType = DataType::IntegerType;
    if(val.getType().isa<mlir::MemRefType>()){
      valType = DataType::MemrefType;
    } else if (val.getType().isa<mlir::FloatType>()) {
      valType = DataType::FloatType;
    } else if (val.getType().isa<mlir::IntegerType>()) {
      valType = DataType::IntegerType;
    } else if (val.getType().isa<mlir::IndexType>()){
      valType = DataType::IntegerType;
    } else {
      assert(!"Unsuppported Type.");  
    }

  auto ff = std::find_if(this->live_out_sets.begin(), this->live_out_sets.end(), [&val](auto& arg) -> bool {
    return arg.get()->getArgumentValue() == val;
  });
  if (ff == this->live_out_sets.end()) {
    this->live_out_sets.push_back(std::make_unique<ArgumentNode>(
        NodeInfo(this->live_out_sets.size(), "live_in_" + std::to_string(this->live_out_sets.size())),
        type, valType, val, this));

    ff = std::find_if(this->live_out_sets.begin(), this->live_out_sets.end(), [&val](auto& arg) -> bool {
      return arg.get()->getArgumentValue() == val;
    });
  }

  return ff->get();
}

ArgumentNode*
ContainerNode::insertCarryDepenArgument(Value val,
                                        ArgumentNode::ArgumentType argtype) {
  auto valType = DataType::IntegerType;
    if(val.getType().isa<mlir::MemRefType>()){
      valType = DataType::MemrefType;
    } else if (val.getType().isa<mlir::FloatType>()) {
      valType = DataType::FloatType;
    } else if (val.getType().isa<mlir::IntegerType>()) {
      valType = DataType::IntegerType;
    } else if (val.getType().isa<mlir::IndexType>()){
      valType = DataType::IntegerType;
    } else {
      assert(!"Unsuppported Type.");  
    }
  auto ff =
      std::find_if(this->carry_depen_sets.begin(), this->carry_depen_sets.end(), [&val](auto& arg) -> bool {
        return arg.get()->getArgumentValue() == val;
      });
  if (ff == this->carry_depen_sets.end()) {
    this->carry_depen_sets.push_back(std::make_unique<ArgumentNode>(
        NodeInfo(this->carry_depen_sets.size(), "carry_" + std::to_string(this->carry_depen_sets.size())),
        argtype,
        valType,
        val, this));

    ff = std::find_if(this->carry_depen_sets.begin(), this->carry_depen_sets.end(), [&val](auto& arg) -> bool {
      return arg.get()->getArgumentValue() == val;
    });
  }

  return ff->get();
}

Node*
ContainerNode::findCarryDepenNode(Value val) {
  auto ff =
      std::find_if(this->carry_depen_sets.begin(), this->carry_depen_sets.end(), [&val](auto& arg) -> bool {
        return arg.get()->getArgumentValue() == val;
      });
  if (ff == this->carry_depen_sets.end()) {
    return nullptr;
    // assert(!"Couldn't find the live-in");
  }

  return ff->get();
}

uint32_t
ContainerNode::findLiveInArgumentIndex(ArgumentNode* _arg_node) {
  auto _arg_type      = _arg_node->getArgType();
  auto _arg_data_type = _arg_node->getDataArgType();

  RegisterList _local_list;

  auto find_function = [_arg_type](auto& node) {
    return (node->getArgType() == _arg_type);
  };

  switch (con_type) {
    case ContainType::FuncCallTy: {
      if (_arg_data_type == DataType::MemrefType) {
        std::copy_if(live_in_mems.begin(),
                     live_in_mems.end(),
                     std::back_inserter(_local_list),
                     find_function);
      } else if (_arg_data_type == DataType::IntegerType || _arg_data_type == DataType::FloatType ) {
        std::copy_if(live_in_vals.begin(),
                     live_in_vals.end(),
                     std::back_inserter(_local_list),
                     find_function);
      }
    }
    case ContainType::LoopNodeTy: {
      std::copy_if(
          live_in_sets.begin(), live_in_sets.end(), std::back_inserter(_local_list), find_function);
    }
  }

  auto arg_find = std::find_if(
      _local_list.begin(), _local_list.end(), [_arg_node](auto& arg) -> bool {
        return arg.get() == _arg_node;
      });

  ptrdiff_t pos = std::distance(_local_list.begin(), arg_find);
  return pos;
}

uint32_t
ContainerNode::findLiveOutArgumentIndex(ArgumentNode* _arg_node) {
  auto _arg_type = _arg_node->getArgType();
  RegisterList _local_list;

  auto find_function = [_arg_type](auto& node) {
    return (node->getArgType() == _arg_type);
  };

  std::copy_if(
      live_out_sets.begin(), live_out_sets.end(), std::back_inserter(_local_list), find_function);

  auto arg_find = std::find_if(
      _local_list.begin(), _local_list.end(), [_arg_node](auto& arg) -> bool {
        return arg.get() == _arg_node;
      });

  ptrdiff_t pos = std::distance(_local_list.begin(), arg_find);
  return pos;
}

uint32_t ContainerNode::findCarryDepenArgumentIndex(ArgumentNode* _arg_node) {
  auto _arg_type = _arg_node->getArgType();
  RegisterList _local_list;

  auto find_function = [_arg_type](auto& node) {
    return (node->getArgType() == _arg_type);
  };

  std::copy_if(this->carry_depen_sets.begin(),
               this->carry_depen_sets.end(),
               std::back_inserter(_local_list),
               find_function);

  auto arg_find = std::find_if(
      _local_list.begin(), _local_list.end(), [_arg_node](auto& arg) -> bool {
        return arg.get() == _arg_node;
      });

  ptrdiff_t pos = std::distance(_local_list.begin(), arg_find);
  return pos;
}

uint32_t ContainerNode::numLiveInArgList(ArgumentNode::ArgumentType type,
                                DataType dtype) {
  RegisterList _local_list;

  auto find_function = [type](auto& node) { return (node->getArgType() == type); };

  switch (con_type) {
    case ContainType::FuncCallTy: {
      if (dtype == DataType::MemrefType) {
        std::copy_if(live_in_mems.begin(),
                     live_in_mems.end(),
                     std::back_inserter(_local_list),
                     find_function);
      } else if (dtype == DataType::IntegerType || dtype == DataType::FloatType) {
        std::copy_if(live_in_vals.begin(),
                     live_in_vals.end(),
                     std::back_inserter(_local_list),
                     find_function);
      }
    }
    case ContainType::LoopNodeTy: {
      std::copy_if(
          live_in_sets.begin(), live_in_sets.end(), std::back_inserter(_local_list), find_function);
    }
  }

  return _local_list.size();
}

uint32_t ContainerNode::numLiveOutArgList(ArgumentNode::ArgumentType type) {
  RegisterList _local_list;

  auto find_function = [type](auto& node) { return (node->getArgType() == type); };

  std::copy_if(
      live_out_sets.begin(), live_out_sets.end(), std::back_inserter(_local_list), find_function);

  return _local_list.size();
}

uint32_t
ContainerNode::numCarryDepenArgList(ArgumentNode::ArgumentType type) {
  RegisterList _local_list;

  auto find_function = [type](auto& node) { return (node->getArgType() == type); };

  std::copy_if(this->carry_depen_sets.begin(),
               this->carry_depen_sets.end(),
               std::back_inserter(_local_list),
               find_function);

  return _local_list.size();
}

Node* ContainerNode::findLiveInNode(Value val) {
  Node* return_ptr = nullptr;

  switch (con_type) {
    case ContainerNode::ContainType::FuncCallTy: {
      if (val.getType().isa<mlir::MemRefType>()) {
        auto ff = std::find_if(
            this->live_in_mems.begin(), this->live_in_mems.end(), [&val](auto& arg) -> bool {
              return arg.get()->getArgumentValue() == val;
            });
        if (ff == this->live_in_mems.end()) {
          return nullptr;
        }

        return_ptr = ff->get();
      } else {
        auto ff = std::find_if(
            this->live_in_vals.begin(), this->live_in_vals.end(), [&val](auto& arg) -> bool {
              return arg.get()->getArgumentValue() == val;
            });
        if (ff == this->live_in_vals.end()) {
          return nullptr;
        }

        return_ptr = ff->get();
      }
    }
    case ContainerNode::ContainType::LoopNodeTy: {
      auto ff = std::find_if(this->live_in_sets_begin(), this->live_in_sets_end(), [&val](auto& arg) -> bool {
        return arg.get()->getArgumentValue() == val;
      });
      if (ff == this->live_in_sets_end()) {
        return nullptr;
      }
      return_ptr = (ff->get());
    }
  }

  return return_ptr;
}

Node* ContainerNode::findLiveOutNode(Value val) {
  auto ff = std::find_if(live_out_sets.begin(), live_out_sets.end(), [&val](auto& arg) -> bool {
    return arg.get()->getArgumentValue() == val;
  });
  if (ff == live_out_sets.end()) {
    return nullptr;
  }

  return ff->get();
}


mlir::Operation* OperationNode::getOperation(){
    return this->parent_op;
}

void ExecutionBlockNode::addconstNode(ConstNode* node) {
  this->const_list.push_back(node);
}
void ExecutionBlockNode::addOperationNode(OperationNode* node) {
  this->operation_node_list.push_back(node);
}
void ExecutionBlockNode::addMergeNode(MergeNode* node){
  this->merge_list.push_back(node);
}

PortID
Node::addReadMemoryReqPort(Node* const n) {
  auto _port_info = PortID(read_port_data.memory_req_port.size());
  read_port_data.memory_req_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID
Node::addReadMemoryRespPort(Node* const n) {
  auto _port_info = PortID(read_port_data.memory_resp_port.size());
  read_port_data.memory_resp_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID
Node::addWriteMemoryReqPort(Node* const n) {
  auto _port_info = PortID(write_port_data.memory_req_port.size());
  write_port_data.memory_req_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID
Node::addWriteMemoryRespPort(Node* const n) {
  auto _port_info = PortID(write_port_data.memory_resp_port.size());
  write_port_data.memory_resp_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

Node* Node::returnControlOutputPortNode(uint32_t index) {
  auto node = port_control.control_output_port.begin();
  std::advance(node, index);
  return node->first;
}

PortID Node::returnDataOutputPortIndex(Node* _node) {
  auto ff = std::find_if(this->port_data.data_output_port.begin(),
                         this->port_data.data_output_port.end(),
                         [&_node](auto& arg) -> bool { return arg.first == _node; });
  if (ff == this->port_data.data_output_port.end()){
    assert(!"Node doesn't exist\n");
  }

  return find_if(this->port_data.data_output_port.begin(),
                 this->port_data.data_output_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}

PortID Node::returnDataInputPortIndex(Node* _node) {
  auto ff = std::find_if(this->port_data.data_input_port.begin(),
                         this->port_data.data_input_port.end(),
                         [&_node](auto& arg) -> bool { return arg.first == _node; });

  if (ff == this->port_data.data_input_port.end())
    assert(!"Node doesn't exist\n");

  return find_if(this->port_data.data_input_port.begin(),
                 this->port_data.data_input_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}

PortID
Node::returnControlOutputPortIndex(Node* _node) {
  auto ff = std::find_if(this->port_control.control_output_port.begin(),
                         this->port_control.control_output_port.end(),
                         [&_node](auto& arg) -> bool { return arg.first == _node; });

  if (ff == this->port_control.control_output_port.end())
    assert(!"Node doesn't exist\n");

  return ff->second;
}

PortID
Node::returnControlInputPortIndex(Node* _node) {
  auto ff = find_if(this->port_control.control_input_port.begin(),
                    this->port_control.control_input_port.end(),
                    [&_node](auto& arg) -> bool { return arg.first == _node; });

  if (ff == this->port_control.control_input_port.end())
    assert(!"Input node doesn't exist\n");

  return ff->second;
}
// Return memory indexes
PortID
Node::returnMemoryReadInputPortIndex(Node* _node) {
  return find_if(this->read_port_data.memory_req_port.begin(),
                 this->read_port_data.memory_req_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}

PortID
Node::returnMemoryReadOutputPortIndex(Node* _node) {
  return find_if(this->read_port_data.memory_resp_port.begin(),
                 this->read_port_data.memory_resp_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}

PortID
Node::returnMemoryWriteInputPortIndex(Node* _node) {
  return find_if(this->write_port_data.memory_req_port.begin(),
                 this->write_port_data.memory_req_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}

PortID
Node::returnMemoryWriteOutputPortIndex(Node* _node) {
  return find_if(this->write_port_data.memory_resp_port.begin(),
                 this->write_port_data.memory_resp_port.end(),
                 [&_node](auto& arg) -> bool { return arg.first == _node; })
      ->second;
}
PortID Node::addDataInputPort(Node* n) {
  auto _port_info = PortID(this->port_data.data_input_port.size());
  this->port_data.data_input_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID Node::addDataInputPort(Node* n, uint32_t _id) {
  auto _port_info = PortID(_id);
  this->port_data.data_input_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID Node::addDataOutputPort(Node* n) {
  auto _port_info = PortID(this->port_data.data_output_port.size());
  this->port_data.data_output_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID Node::addDataOutputPort(Node* n, uint32_t _id) {
  auto _port_info = PortID(_id);
  this->port_data.data_output_port.emplace_back(std::make_pair(n, _port_info));
  return _port_info;
}

PortID Node::addControlInputPort(Node* n) {
  auto port_info = PortID(this->port_control.control_input_port.size());
  this->port_control.control_input_port.emplace_back(std::make_pair(n, port_info));
  return port_info;
}

PortID Node::addControlInputPort(Node* n, uint32_t _id) {
  auto port_info = PortID(_id);
  this->port_control.control_input_port.emplace_back(std::make_pair(n, port_info));
  return port_info;
}

PortID Node::addControlOutputPort(Node* n) {
  auto port_info = PortID(this->port_control.control_output_port.size());
  this->port_control.control_output_port.emplace_back(std::make_pair(n, port_info));
  return port_info;
}

PortID Node::addControlOutputPort(Node* n, uint32_t _id) {
  auto port_info = PortID(_id);
  this->port_control.control_output_port.emplace_back(std::make_pair(n, port_info));
  return port_info;
}

