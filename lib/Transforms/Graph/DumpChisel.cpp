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

/**
 * Node
 */

std::string ConstNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "  val $name = Module(new $type(value = $val"
              ", ID = $id))\n\n";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$num_out",
      std::to_string(this->numDataOutputPort()));
      strReplace(_text, "$id", this->getID());
      strReplace(_text, "$type", "ConstFastNode");
      if(this->isInt){
          strReplace(_text, "$val", std::to_string(this->value));
      } else {
          strReplace(_text, "$val", std::to_string(this->value_f) + "L");
      }
      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string ConstNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.Out";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ConstNode::printInputEnable(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


//===----------------------------------------------------------------------===//
//                            BitCastNode Class
//===----------------------------------------------------------------------===//

std::string BitCastNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "  val $name = Module(new $type(NumOuts = "
              "$num_out, ID = $id))\n\n";
      strReplace(_text, "$id", std::to_string(this->getID()));
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$type", "BitCastNode");
      strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string BitCastNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string BitCastNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string BitCastNode::printInputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Input";
      strReplace(_text, "$name", _name.c_str());

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string ComputeOperationNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
        _text = "  val $name = Module(new $type(NumOuts = "
            "$num_out, ID = $id, opCode = \"$opcode\")(sign = false, Debug "
            "= false))\n\n";
  
        strReplace(_text, "$name", _name.c_str());
        strReplace(_text, "$id", this->getID());
        strReplace(_text, "$opcode", this->getOpcodeName());
        strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));
        if(this->isIntegerType()){
            // strReplace(_text, "$type", "ComputeNode");
            strReplace(_text, "$type", "ComputeNodeWithoutState");
        } else if (this->isFloatType()){
            strReplace(_text, "(sign = false, Debug "
            "= false)", "(t = FType.D)");
            strReplace(_text, "$type", "FPComputeNode");
        }
     break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ComputeOperationNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ComputeOperationNode::printOutputData(PrintType _pt, uint32_t _port_id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($_port_id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$_port_id", _port_id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ComputeOperationNode::printInputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      if (_idx == 0)
        _text = "$name.io.LeftIO";
      else
        _text = "$name.io.RightIO";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}



//===----------------------------------------------------------------------===//
//                            CmpNode Class
//===----------------------------------------------------------------------===//

std::string CmpNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "  val $name = Module(new $type(NumOuts = "
              "$num_out, ID = $id, opCode = \"$opcode\")(sign = $sign, Debug "
              "= false))\n\n";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));
      strReplace(_text, "$id", this->getID());
      // strReplace(_text, "$type", "ComputeNode");
      strReplace(_text, "$type", "ComputeNodeWithoutState");
      strReplace(
          _text,
          "$opcode",
          opCodeToString(this->geOpCodeEnum()));
      //TODO: SIGN && Folat Cmp.
      strReplace(_text, "$sign", "false");

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string CmpNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string CmpNode::printInputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      if (_idx == 0)
        _text = "$name.io.LeftIO";
      else
        _text = "$name.io.RightIO";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string
CmpNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}



//===----------------------------------------------------------------------===//
//                            StateBranchNode Class
//===----------------------------------------------------------------------===//

std::string StateBranchNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      if (this->numControlInputPort() > 1 && this->numControlOutputPort() == 1
          && this->numDataInputPort() == 0)
        _text = "  val $name = Module(new $type(NumPredOps=$npo, ID = "
                "$id))\n\n";
      else if (this->numControlInputPort() > 1 && this->numControlOutputPort() > 1
               && this->numDataInputPort() == 0)
        _text = "  val $name = Module(new $type(NumPredOps=$npo, "
                "NumOuts=$nout, ID = "
                "$id))\n\n";
      else if (this->numControlInputPort() == 1 && this->numControlOutputPort() > 1
               && this->numDataInputPort() == 0)
        _text = "  val $name = Module(new $type(NumOuts=$nout, ID = "
                "$id))\n\n";
      else
        _text = "  val $name = Module(new $type(ID = "
                "$id))\n\n";

      if (this->numDataInputPort() > 0) {
        _text = "  val $name = Module(new $type(NumTrue = $true, NumFalse "
                "= $false, NumPredecessor = $pred, ID = "
                "$id))\n\n";

        // Getting port index
        uint32_t p_true_index  = 0;
        uint32_t p_false_index = 0;
        for (auto _p : this->output_predicate) {
          if (_p.second == this->PredicateResult::False)
            p_false_index++;
          else if (_p.second == this->PredicateResult::True)
            p_true_index++;
        }

        strReplace(_text,
                      "$type",
                      "CBranchNodeVariable");
        strReplace(_text, "$false", p_false_index);
        strReplace(_text, "$true", p_true_index);
        strReplace(_text, "$pred", this->numControlInputPort() - 1);

      } else
        strReplace(_text, "$type", "UBranchNode");
      strReplace(_text, "$nout", this->numControlOutputPort());
      strReplace(_text, "$npo", this->numControlInputPort() - 1);
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());
      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string StateBranchNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string StateBranchNode::printInputEnable(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.PredOp($id)";
      strReplace(_text, "$name", _name.c_str());

      strReplace(_text, "$id", _id -1);
      break;
    default: break;
  }

  return _text;
}

std::string StateBranchNode::printOutputEnable(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      // The branch is UBranch and there is no true and false outptu
      if (this->numDataInputPort() == 0) {
        _text = "$name.io.Out($id)";
        strReplace(_text, "$name", _name.c_str());
        strReplace(_text, "$id", _id);
      } else {
        // The branch is CBranch and there is true and false outptut
        auto node            = this->returnControlOutputPortNode(_id);
        uint32_t false_index = 0;
        uint32_t true_index  = 0;
        for (auto pr : output_predicate) {
          if (pr.first == node) {
            auto result = printed_predicate.insert(std::make_pair(pr, 1));
            if (result.second == false) {
              if (pr.second == StateBranchNode::PredicateResult::False)
                false_index++;
              else
                true_index++;
              continue;
            } else {
              if (pr.second == StateBranchNode::PredicateResult::True) {
                _text = "$name.io.TrueOutput($id)";
                strReplace(_text, "$name", _name.c_str());
                strReplace(_text, "$id", true_index);
              } else if (pr.second == StateBranchNode::PredicateResult::False) {
                _text = "$name.io.FalseOutput($id)";
                strReplace(_text, "$name", _name.c_str());
                strReplace(_text, "$id", false_index);
              } else {
                _text = "$name.io.CONDITIONAL?";
                strReplace(_text, "$name", _name.c_str());
              }
              break;
            }
          }

          if (pr.second == StateBranchNode::PredicateResult::False)
            false_index++;
          else
            true_index++;
        }
      }
      break;
    default: break;
  }

  return _text;
}

std::string StateBranchNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.CmpIO";
      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}


std::string SelectNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "  val $name = Module(new $type("
              "NumOuts = $num_out, ID = $id))\n\n";
      // strReplace(_text, "$type", "SelectNode");
      strReplace(_text, "$type", "SelectNodeWithOutState");
      strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string SelectNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string SelectNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      if (_id == 0)
        _text = "$name.io.Select";
      else if (_id == 1)
        _text = "$name.io.InData1";
      else if (_id == 2)
        _text = "$name.io.InData2";
      else
        assert(!"Select nod can not have more than three inputs! (select, "
                "input1, input2)");

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string SelectNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string ExecutionBlockNode::printDefinition(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "  val $name = Module(new $type(NumInputs = $num_in, NumOuts = "
              "$num_out, BID = $bid))\n\n";

      if (!this->getNodeType()) {
          strReplace(
              _text,
              "$type",
              "ExecutionBlockNode");}
        else {
          _text = "  val $name = Module(new $type("
                  "NumInputs = $num_in, "
                  "NumOuts = "
                  "$num_out, NumPhi = $num_phi, BID = $bid))\n\n";
          strReplace(_text, "$type", "BasicBlockNode");
        }
  

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$num_in", this->numControlInputPort());
      strReplace(_text, "$num_out", this->numControlOutputPort());
      strReplace(_text, "$bid", this->getID());
      strReplace(_text, "$num_phi", this->getNumMerge());

    break;
    
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ExecutionBlockNode::printInputEnable(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.predicateIn($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);
      break;
    default: break;
  }

  return _text;
}

std::string ExecutionBlockNode::printInputEnable(PrintType pt, std::pair<Node*, PortID> _node) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.predicateIn($id)";

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _node.second.getID());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ExecutionBlockNode::printOutputEnable(PrintType pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ExecutionBlockNode::printOutputEnable(PrintType pt, std::pair<Node*, PortID> _node) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _node.second.getID());

      break;
    
    default: assert(!"Uknown print type!");
  }
  return _text;
}
std::string ExecutionBlockNode::printMaskOutput(PrintType pt, uint32_t _id) {
  std::string  _text;
  std::string  _name(this->getName());
  switch (pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.MaskBB($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

//===----------------------------------------------------------------------===//
//                            MergeNode Class
//===----------------------------------------------------------------------===//

std::string MergeNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "  val $name = Module(new $type(NumInputs = $num_in, "
              "NumOutputs = $num_out, ID = $id, Res = false))\n\n";

      strReplace(_text, "$type", "MergeNode");
      strReplace(_text, "$num_in", this->numDataInputPort());
      strReplace(_text, "$num_out",this->numDataOutputPort());

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string MergeNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string MergeNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.InData($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string MergeNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string MergeNode::printMaskInput(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Mask";
      strReplace(_text, "$name", _name.c_str());

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

//===----------------------------------------------------------------------===//
//                            LoopNode Class
//===----------------------------------------------------------------------===//


std::string LoopNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());

  auto make_argument_port = [](const auto& _list) {
    std::vector<uint32_t> _arg_count;
    for (auto& l : _list)
      _arg_count.push_back(l->numDataOutputPort());
    return _arg_count;
  };

  switch (_pt) {
    case PrintType::Scala: {
      _text = "  val $name = Module(new $type(NumIns = "
              "List($<input_vector>), "
              "NumOuts = List($<num_out>), "
              "NumCarry = List($<num_carry>), "
              "NumExits = $num_exit, ID = $id))\n\n";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());
      strReplace(_text, "$type", "LoopBlockNode");
      strReplace(_text, "$num_exit", static_cast<uint32_t>(this->loop_exits.size()));

      auto live_in_args = make_argument_port(this->getLiveInSets());
      strReplace(_text, "$<input_vector>", live_in_args, ", ");

      strReplace(_text, "$<num_out>", make_argument_port(this->getLiveOutSets()), ", ");

      strReplace(_text, "$<num_carry>", make_argument_port(this->getCarryDepenSets()), ", ");

      break;
    }
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string LoopNode::printOutputEnable(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::replace(_name.begin(), _name.end(), '.', '_');
  std::string _text;
  auto node   = this->returnControlOutputPortNode(_id);
  auto node_t = find_if(port_type.begin(), port_type.end(), [node](auto _nt) -> bool {
    return _nt.first == node;
  });

  switch (_pt) {
    case PrintType::Scala:
      if (node_t->second == PortType::LoopFinish)
        _text = "$name.io.loopfinish";
      else if (node_t->second == PortType::Active_Loop_Start)
        _text = "$name.io.activate_loop_start";
      else if (node_t->second == PortType::Active_Loop_Back)
        _text = "$name.io.activate_loop_back";
      else if (node_t->second == PortType::Enable)
        _text = "$name.io.???";

      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}

std::string LoopNode::printOutputEnable(PrintType _pt, PortEntry _port) {
  std::string _name(this->getName());
  std::replace(_name.begin(), _name.end(), '.', '_');
  std::string _text;

  auto port_equal = [](auto port_1, auto port_2) -> bool {
    return ((port_1.first == port_2.first)
            && (port_1.second.getID() == port_2.second.getID()));
  };

  switch (_pt) {
    case PrintType::Scala:
      if (port_equal(this->activate_loop_start, _port)){
        if(!this->getSignal())
          _text = "$name.io.activate_loop_start";
        else 
          _text = "$name.io.activate_loop_back";
        this->setSignal();
      }
      //   _text = "$name.io.activate_loop_start";
      // else if (port_equal(this->activate_loop_back, _port))
      //   _text = "$name.io.activate_loop_back";
      else {
        auto out_port = find_if(this->loop_exits.begin(),
                                this->loop_exits.end(),
                                std::bind(port_equal, std::placeholders::_1, _port));

        if (out_port != this->loop_exits.end()) {
          _text        = "$name.io.loopExit($id)";
          uint32_t pos = std::distance(this->loop_exits.begin(), out_port);
          strReplace(_text, "$id", pos);

        } else
          _text = "$name.io.XXX";
      }

      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}

std::string LoopNode::printInputEnable(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      if (_id == 0)
        _text = "$name.io.enable";
      else if (_id == 1)
        _text = "$name.io.loopBack(0)";
      else if (_id >= 2)
        _text = "$name.io.loopFinish($id)";
      else
        _text = "XXXXXX";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id - 2);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}
//===----------------------------------------------------------------------===//
//                            ReturnNode 
//===----------------------------------------------------------------------===//
//
std::string ReturnNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "  val $name = Module(new $type(retTypes = "
              "List($<input_list>), "
              "ID = $id))\n\n";
      strReplace(_text, "$type", "RetNode2");
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());
      strReplace(_text,
                    "$<input_list>",
                    std::vector<uint32_t>(this->numDataInputPort(), 32),
                    ", ");

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ReturnNode::printInputEnable(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.In.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string ReturnNode::printInputEnable(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.In.Succ($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _idx);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string ReturnNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.In.data(\"field$id\")";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ReturnNode::printOutputData(PrintType _pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Out";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ReturnNode::printOutputData(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Out";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string AllocaNode::printDefinition(PrintType _pt) {
    return "";
}



//===----------------------------------------------------------------------===//
//                            ArgumentNode Class
//===----------------------------------------------------------------------===//
std::string ArgumentNode::printDefinition(PrintType _pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "  val $name = Module(new $type(NumOuts = "
              "$num_out, ID = $id))\n\n";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));
      strReplace(_text, "$id", this->getID());
      strReplace(_text, "$type", "ArgumentNode");

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ArgumentNode::printInputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala: {
      switch (this->getArgType()) {
        case ArgumentNode::ArgumentType::LiveIn: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.In($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(_text, "$id", _idx);

          break;
        }
        case ArgumentNode::ArgumentType::LiveOut: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.InLiveOut($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(_text, "$id", _idx);
          break;
        }
        case ArgumentNode::ArgumentType::LoopLiveIn: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.InLiveIn($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(_text, "$id", _idx);

          break;
        }
        case ArgumentNode::ArgumentType::LoopLiveOut: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.InLiveOut($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(_text, "$id", _idx);
          break;
        }
        case ArgumentNode::ArgumentType::CarryDependency: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.CarryDepenIn($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(_text, "$id", _idx);
          break;
        }

        default: assert(!"Unrecognized argument node type!"); break;
      }

      break;
    }
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string ArgumentNode::printOutputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala: {
      switch (this->getArgType()) {
        case ArgumentNode::ArgumentType::LiveIn: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.$out.data.elements(\"field$num\")($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(
              _text, "$num", this->parent_call_node->findLiveInArgumentIndex(this));
          strReplace(_text, "$out", "Out");
          strReplace(_text, "$id", _idx);
          // if (this->getDataArgType() == DataType::MemrefType)
          //   strReplace(_text, "$data", "dataPtrs");
          // else if (this->getDataArgType() == DataType::IntegerType || this->getDataArgType() == DataType::FloatType)
          //   strReplace(_text, "$data", "dataVals");
          break;
        }
        case ArgumentNode::ArgumentType::LiveOut: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.$out($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(
              _text, "$num", this->parent_call_node->findLiveOutArgumentIndex(this));
          strReplace(_text, "$out", "Out");
          strReplace(_text, "$id", _idx);
          break;
        }
        case ArgumentNode::ArgumentType::LoopLiveIn: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.$out.elements(\"field$num\")($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(
              _text, "$num", this->parent_call_node->findLiveInArgumentIndex(this));
          if (this->parent_call_node->getContainerType() == ContainerNode::ContainType::LoopNodeTy)
            strReplace(_text, "$out", "OutLiveIn");
          else
            strReplace(_text, "$out", "Out");

          strReplace(_text, "$id", _idx);

          break;
        }
        case ArgumentNode::ArgumentType::LoopLiveOut: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.$out.elements(\"field$num\")($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(
              _text, "$num", this->parent_call_node->findLiveOutArgumentIndex(this));
          strReplace(_text, "$out", "OutLiveOut");

          strReplace(_text, "$id", _idx);
          break;
        }
        case ArgumentNode::ArgumentType::CarryDependency: {
          std::replace(_name.begin(), _name.end(), '.', '_');
          _text = "$call.io.$out.elements(\"field$num\")($id)";
          strReplace(_text, "$call", this->parent_call_node->getName());
          strReplace(
              _text, "$num", this->parent_call_node->findCarryDepenArgumentIndex(this));
          strReplace(_text, "$out", "CarryDepenOut");

          strReplace(_text, "$id", _idx);
          break;
        }
        default: assert(!"Unrecognized type of node\n"); break;
      }

      break;
    }
    default: assert(!"Uknown print type!");
  }
  return _text;
}


//===----------------------------------------------------------------------===//
//                            GetElementPtrStruct Class
//===----------------------------------------------------------------------===//
std::string AddressGenNode::printDefinition(PrintType _pt) {
  std::string _text("");
  std::string _name(this->getName());

  switch (_pt) {
    case PrintType::Scala: {
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "  val $name = Module(new $type(NumIns = $num_ins, "
              "NumOuts = $num_out, "
              "ID = $id)(ElementSize = $size, ArraySize = List()))\n\n";
      strReplace(_text, "$type", "GepNode");
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", std::to_string(this->getID()));
      strReplace(_text, "$num_out", std::to_string(this->numDataOutputPort()));

      // The first input is always baseaddress
      strReplace(_text, "$num_ins", std::to_string(this->numDataInputPort() - 1));

      std::stringstream _array;

      // strReplace(_text, "$size",
      //*std::prev(this->gep_info.element_size.end()));
      strReplace(_text, "$size", 32);

      // strReplace(_text, "$array", "List(" + _array.str() + ")");

      break;
    }
    default: assert(!"Don't support!");
  }
  return _text;
}

std::string AddressGenNode::printInputEnable(PrintType pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.enable($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    case PrintType::Dot: assert(!"Dot file format is not supported!");
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string AddressGenNode::printInputEnable(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string AddressGenNode::printOutputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:
      std::replace(_name.begin(), _name.end(), '.', '_');
      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _idx);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string AddressGenNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::replace(_name.begin(), _name.end(), '.', '_');
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      if (_id == 0)
        _text = "$name.io.baseAddress";
      // else if (_id == 1){
      //    if(this->numDataInputPort() == 2)
      //        _text = "$name.io.idx";
      //    else
      //        _text = "$name.io.idx1";
      //}
      else
        _text = "$name.io.idx($ix)";

      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$ix", _id - 1);
      break;
    default: break;
  }

  return _text;
}



//===----------------------------------------------------------------------===//
//                            LSNode Class
//===----------------------------------------------------------------------===//

std::string LSNode::printDefinition(PrintType _pt) {
  std::string _text("");
  std::string _name(this->getName());

  switch (_pt) {
    case PrintType::Scala: {
      
      if(this->isStore){
        // _text = "  val $name = Module(new UnTypStoreCache(NumPredOps = $npo, "
        //       "NumSuccOps = $nso, "
        //       "ID = $id, RouteID = $rid))\n\n";
        _text = "  val $name = Module(new Store(NumOuts = 1, "
              "ID = $id, RouteID = $rid))\n\n";
      } else  {
        // _text = "  val $name = Module(new UnTypLoadCache(NumPredOps = $npo, "
        //       "NumSuccOps = $nso, "
        //       "NumOuts = $num_out, ID = $id, RouteID = $rid))\n\n";
        _text = "  val $name = Module(new Load("
              "NumOuts = $num_out, ID = $id, RouteID = $rid))\n\n";
      }    
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", this->getID());
      strReplace(_text, "$rid", this->getRouteID());
      strReplace(_text, "$num_out", this->numDataOutputPort());
      strReplace(_text, "$npo", this->numControlInputPort() - 1);
      strReplace(_text, "$nso", this->numControlOutputPort());

      break;
    }
    default: assert(!"Don't support!");
  }
  return _text;
}

std::string LSNode::printInputEnable(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.enable";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}


std::string LSNode::printInputEnable(PrintType pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.PredOp($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id - 1);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string LSNode::printOutputData(PrintType _pt, uint32_t _idx) {
  std::string _text;
  std::string _name(this->getName());
  switch (_pt) {
    case PrintType::Scala:

      _text = "$name.io.Out($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _idx);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string LSNode::printInputData(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  // std::replace(_name.begin(), _name.end(), '.', '_');
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      if(this->isStore){
        if (_id == 0)
          _text = "$name.io.inData";
        else
          _text = "$name.io.GepAddr";
        strReplace(_text, "$name", _name.c_str());
        strReplace(_text, "$id", _id);
      } else {
        _text = "$name.io.GepAddr";

        strReplace(_text, "$name", _name.c_str());
        strReplace(_text, "$mem", this->mem_node->getName());
      }
      
      break;
    default: break;
  }

  return _text;
}

std::string LSNode::printMemReadInput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.MemResp";

      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}

std::string LSNode::printMemReadOutput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.MemReq";
      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}


std::string LSNode::printOutputEnable(PrintType pt, uint32_t _id) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.SuccOp($id)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$id", _id);

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string LSNode::printOutputEnable(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala:

      _text = "$name.io.SuccOp($id)";
      strReplace(_text, "$name", _name.c_str());

      break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}



std::string LSNode::printMemWriteInput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.MemResp";

      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}


std::string LSNode::printMemWriteOutput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.MemReq";
      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}

std::string LSNode::printGround(PrintType _pt) {
  std::string _name(this->getName());
  std::replace(_name.begin(), _name.end(), '.', '_');
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      _text = "$name.io.Out(0).ready := true.B";

      strReplace(_text, "$name", _name.c_str());
      break;
    default: break;
  }

  return _text;
}

std::string MemoryNode::printDefinition(PrintType pt) {
  std::string _text;
  std::string _name(this->getName());
  switch (pt) {
    case PrintType::Scala: {

      _text = "  //Cache\n"
              "  val $name = Module(new $module_type(ID = $id, NumRead = "
              "$num_rd, NumWrite = $num_wr))\n"
              "\n"
              "  io.MemReq <> $name.io.cache.MemReq\n"
              "  $name.io.cache.MemResp <> io.MemResp\n\n";
      ;
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$module_type", "CacheMemoryEngine");
      strReplace(_text, "$id", std::to_string(this->getID()));
      strReplace(_text, "$num_rd", this->numReadDataInputPort());
      strReplace(_text, "$num_wr", this->numWriteDataInputPort());

      // case PrintType::Scala: {
      // _text = "  val $name = Module(new $module_type(ID = $id, Size = 32, "
      //         "NReads = $num_rd, NWrites = $num_wr)\n"
      //         "(WControl = new WriteMemController(NumOps = $num_wr, BaseSize = 2, NumEntries = 2))\n"
      //         "(RControl = new ReadMemController(NumOps = $num_rd, BaseSize = 2, NumEntries = 2))\n"
      //         "(RWArbiter = new ReadWriteArbiter()))"
      // _text = "  //Cache\n"

      // strReplace(_text, "$name", _name.c_str());
      // strReplace(_text, "$module_type", "UnifiedController");
      // strReplace(_text, "$id", std::to_string(this->getID()));
      // strReplace(_text, "$num_rd", this->numReadDataInputPort());
      // strReplace(_text, "$num_wr", this->numWriteDataInputPort());

    // } break;

    } break;
    default: assert(!"Uknown print type!");
  }
  return _text;
}

std::string MemoryNode::printMemReadInput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      // _text = "$name.io.rd.mem($mid).MemReq";
      _text = "$name.io.load_address($mid)";
      // _text = "$name.io.ReadIn($mid)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$mid", _id);
      // TODO add mid
      break;
    default: break;
  }

  return _text;
}

std::string MemoryNode::printMemReadOutput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      // _text = "$name.io.rd.mem($mid).MemResp";
      _text = "$name.address_out";
      // _text = "$name.io.ReadOut($mid)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$mid", _id);
      // TODO add mid
      break;
    default: break;
  }

  return _text;
}

std::string MemoryNode::printMemWriteInput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      // _text = "$name.io.wr.mem($mid).MemReq";
      _text = "$name.io.store_address($mid)";
      // _text = "$name.io.WriteIn($mid)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$mid", _id);
      break;
    default: break;
  }

  return _text;
}

std::string MemoryNode::printMemWriteOutput(PrintType _pt, uint32_t _id) {
  std::string _name(this->getName());
  std::string _text;
  switch (_pt) {
    case PrintType::Scala:
      // _text = "$name.io.wr.mem($mid).MemResp";
      _text = "$name.io.store_data($mid)";
      // _text = "$name.io.WriteOut($mid)";
      strReplace(_text, "$name", _name.c_str());
      strReplace(_text, "$mid", _id);
      break;
    default: break;
  }

  return _text;
}

std::string FineArgCallNode::printDefinition(PrintType _pt) {
    std::string _text;
    std::string _name(this->getName());

    auto make_argument_port = [](const auto& _list) {
    std::vector<uint32_t> _arg_count;
    for (auto& l : _list)
      _arg_count.push_back(l->numDataOutputPort());
        // if (_arg_count.size() == 0)
        //_arg_count.push_back(0);
        return _arg_count;
    };
    
    switch (_pt) {
        case PrintType::Scala: {
       
            _text = "\n  val $name = Module(new $type(argTypes = "
                    "List($<ptrs_input_vector> $<vals_input_vector>)))\n"
                    "  $name.io.In <> io.in\n\n";

            strReplace(_text, "$name", _name);
            strReplace(_text, "$type", "SplitCallDCR");
            strReplace(_text, "$id", std::to_string(this->getID()));

            // TODO make a list of liveins first
            auto find_function = [](auto& node) {
                return (node->getArgType() == ArgumentNode::ArgumentType::LiveIn);
            };
            RegisterList _local_list_ptrs;
            RegisterList _local_list_vals;
            std::copy_if(this->live_in_mems.begin(),
                        this->live_in_mems.end(),
                        std::back_inserter(_local_list_ptrs),
                        find_function);
            std::copy_if(this->live_in_vals.begin(),
                        this->live_in_vals.end(),
                        std::back_inserter(_local_list_vals),
                        find_function);

            strReplace(
                _text, "$<ptrs_input_vector>", make_argument_port(_local_list_ptrs), ", ");
            strReplace(
                _text, "$<vals_input_vector>", make_argument_port(_local_list_vals), ", ");

            break;
        }
        default: assert(!"Don't support!");
    }
    return _text;
}

std::string FineArgCallNode::printOutputEnable(PrintType _pt, uint32_t id){
    std::string _name(this->getName());
    std::string _text;
    switch (_pt) {
        case PrintType::Scala:
        _text = "$name.io.Out.enable";
        strReplace(_text, "$name", _name.c_str());
        break;
        default: break;
    }

    return _text;
}
std::string FineArgCallNode::printOutputData(PrintType _pt, uint32_t id){
    std::string _text;
    std::string _name(this->getName());
    switch (_pt) {
        case PrintType::Scala:
  
        _text = "$name.io.Out.data(\"field$id\")";
        strReplace(_text, "$name", _name.c_str());
        strReplace(_text, "$id", id);

        break;
        
        default: assert(!"Uknown print type!");
    }
    return _text;
}





void Graph::dumpGraph(PrintType _pt, std::string json_path) {
    switch (_pt) {
        case PrintType::Scala:

        printScalaHeader();

        //TODO: Function

        llvm::outs() << "printOperations.\n";
        printOperations(PrintType::Scala);    
        llvm::outs() << "printControlEdge.\n";
        printControlEdge(PrintType::Scala);    
        llvm::outs() << "printLoopConnection.\n";
        printLoopConnection(PrintType::Scala);  
        llvm::outs() << "printConnection.\n";
        printConnection(PrintType::Scala);  

        //Output
        this->outputHardware << "  io.out <> " << this->out_node->printOutputData(PrintType::Scala)
                        << "\n\n";
        this->outputHardware << "}\n\n"; 
        llvm::outs() << "printScalaEnd.\n";
        // printScalaEnd(PrintType::Scala);

        break;

        default: assert(!"Uknown print type!");
    }
}

#define DATA_SIZE 32
/**
 * Print specific scala header files
 */
void Graph::printScalaHeader(){

    this->outputHardware << R"AAA(
//===------------------------------------------------------------*- Scala -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

package heteacc.generator

import chipsalliance.rocketchip.config._
import chisel3._
import chisel3.util._
import chisel3.Module._
import chisel3.testers._
import chisel3.iotesters._


import heteacc.config._
import heteacc.fpu._
import heteacc.interfaces._
import heteacc.junctions._
import heteacc.memory._
import heteacc.node._
import heteacc.loop._
import heteacc.execution._
import utility._
)AAA";

    std::string function_file = "\nabstract class $module_nameDFIO(implicit val p: Parameters) extends Module with HasAccelParams {\n"
      "\tval io = IO(new Bundle {\n"
      "\t  val in = Flipped(Decoupled(new Call(List($<input_vector_vals> $<input_vector_ptrs>))))\n"
      // "\t  val MemResp = Flipped(Valid(new MemResp))\n"
      // "\t  val MemReq = Decoupled(new MemReq)\n"
      "\t  val out = Decoupled(new Call(List($<output_vector>)))\n"
      "\t})\n}\n\nclass $module_nameDF(implicit p: Parameters) extends $module_nameDFIO()(p){\n";

    strReplace(function_file, "$module_name", this->graph_info.Name);
    auto num_in_args_ptrs = this->getArgCall()->numLiveInArgList(
        ArgumentNode::ArgumentType::LiveIn, DataType::MemrefType);
    auto num_in_args_vals = this->getArgCall()->numLiveInArgList(
        ArgumentNode::ArgumentType::LiveIn, DataType::IntegerType);

    uint32_t num_out_args = (this->hasreturnValue) ? 1 : 0;
    std::vector<uint32_t> _input_args_ptrs(num_in_args_ptrs, DATA_SIZE);
    std::vector<uint32_t> _input_args_vals(num_in_args_vals, DATA_SIZE);
    std::vector<uint32_t> _output_args(num_out_args, DATA_SIZE);

    strReplace(function_file, "$<input_vector_ptrs>", _input_args_ptrs, ", ");
    strReplace(function_file, "$<input_vector_vals>", _input_args_vals, ", ");
    strReplace(function_file, "$<output_vector>", _output_args, ", ");

    this->outputHardware << function_file;

      //FineArgCallNode
    this->outputHardware << this->getArgCall()->printDefinition(PrintType::Scala);

    // Memory.
    this->outputHardware << this->memory_unit->printDefinition(PrintType::Scala);
}

/**
 * Print the operation definition
 */
void Graph::printOperations(PrintType _pt) {
    switch (_pt) {
        case PrintType::Scala:

        this->outputHardware << helperScalaPrintHeader("Printing Const nodes.");
        for(auto &const_node : this->const_list){
            this->outputHardware << "  //";
            if (const_node->getConstantOp())
            const_node->getConstantOp()->print(this->outputHardware);
            else
                this->outputHardware << "NullPtr";
            this->outputHardware << "\n";
            this->outputHardware << const_node->printDefinition(PrintType::Scala);
        }
        this->outputHardware << helperScalaPrintHeader("Printing Execution Block nodes.");
        for (auto &exe : this->exe_block_list) {
            this->outputHardware << exe->printDefinition(PrintType::Scala);
        }

        this->outputHardware << helperScalaPrintHeader("Printing Operation nodes. " + std::to_string(op_list.size()));
        for (auto& ins_node : this->op_list) {
            // auto compute_op = dyn_cast<ComputeOperationNode>(&*ins_node);
            // if(compute_op){ 
            //     this->outputHardware << "  //";
            //     ins_node->getOperation()->print(this->outputHardware);
            //     this->outputHardware << "\n";
            //     this->outputHardware << ins_node->printDefinition(PrintType::Scala);
            // }

            this->outputHardware << "  //";
            ins_node->getOperation()->print(this->outputHardware);
            this->outputHardware << "\n";
            this->outputHardware << ins_node->printDefinition(PrintType::Scala);

            // continue;
            // auto call_ins = dyn_cast<ReturnNode>(&*ins_node);
            // if (ins_node->numDataOutputPort() == 0 && ins_node->numControlOutputPort() == 0
            //     && call_ins == nullptr)
            // continue;
            
            // break;
        }

        this->outputHardware << helperScalaPrintHeader("Printing Loop nodes.");
        for(auto &node : this->loop_nodes){
          this->outputHardware << node->printDefinition(PrintType::Scala);
        }
        break;
    default: assert(!"We don't support the other types right now");
  }

}
void Graph::printControlEdge(PrintType _pt) {
  switch (_pt) {
        case PrintType::Scala:
        this->outputHardware << helperScalaPrintHeader("Control Signal.");
        
        for(auto iter_output_control = this->fine_arg_call.get()->outputControl_begin(); iter_output_control != this->fine_arg_call.get()->outputControl_end(); iter_output_control++){
          auto node   = (iter_output_control->first);
          auto output_index = node->returnControlInputPortIndex(this->fine_arg_call.get());

          auto input_index = this->fine_arg_call.get()->returnControlOutputPortIndex(node);

          this->outputHardware << "  "
                        << this->fine_arg_call.get()->printOutputEnable(PrintType::Scala, output_index.getID())
                        << " <> "
                        << node->printInputEnable(PrintType::Scala)//,input_index.getID()
                        << "\n\n";
        }
        for(auto &loop: this->loop_nodes){
          for(auto &op : this->op_list){
            auto _list_input_enable   = op.get()->findControlInputNodeList(loop.get());
            auto _list_output_enable  = loop->findControlOutputNodeList(op.get());

            auto _input_it  = _list_input_enable.begin();
            auto _output_it = _list_output_enable.begin();

            for (uint32_t i = 0; i < _list_input_enable.size(); ++i) {
              std::advance(_input_it, i);
              std::advance(_output_it, i);
              this->outputHardware << "  "
                            << _input_it->first->printOutputEnable(
                                  PrintType::Scala, *_output_it)
                            << " <> "
                            << _output_it->first->printInputEnable(PrintType::Scala)//_input_it->second.getID()
                            << "\n\n";
            }

          }  
        }
        for(auto &exe: this->exe_block_list){
            std::set<Node*> unique_loop_nodes;
            for(auto iter_input_control = exe->inputControl_begin(); iter_input_control != exe->inputControl_end(); iter_input_control++){
              if(static_cast<LoopNode*>(iter_input_control->first)->getType() == ContainerNode::ContainType::LoopNodeTy){
                unique_loop_nodes.insert(static_cast<LoopNode*>(iter_input_control->first));
              } else {
                auto node   = static_cast<Node*>(iter_input_control->first);
                auto output_index = node->returnControlOutputPortIndex(exe.get());
                this->outputHardware << "  "
                        << exe->printInputEnable(PrintType::Scala, *iter_input_control)
                        << " <> "
                        << node->printOutputEnable(PrintType::Scala,
                                                          output_index.getID())
                        << "\n\n";
              }
            }
            for (auto loop : unique_loop_nodes) {
              auto _list_input_enable  = exe->findControlInputNodeList(loop);
              auto _list_output_enable = loop->findControlOutputNodeList(exe.get());

              assert(_list_input_enable.size() == _list_output_enable.size());

              auto _input_it  = _list_input_enable.begin();
              auto _output_it = _list_output_enable.begin();

              for (uint32_t i = 0; i < _list_input_enable.size(); ++i) {
                std::advance(_input_it, i);
                std::advance(_output_it, i);
                this->outputHardware << "  "
                              << _output_it->first->printInputEnable(
                                    PrintType::Scala, _input_it->second.getID())
                              << " <> "
                              << _input_it->first->printOutputEnable(PrintType::Scala,
                                                                    *_output_it)
                              << "\n\n";
              }
            }  
        }
        
        this->outputHardware << helperScalaPrintHeader("Loop Control Signal.");
        for (auto& loop : this->loop_nodes) {
          for(auto iter_input_control = loop.get()->inputControl_begin(); iter_input_control != loop.get()->inputControl_end(); iter_input_control++){
            auto _input_node = iter_input_control->first;
            auto _output_index = _input_node->returnControlOutputPortIndex(loop.get());
            this->outputHardware << "  "
                          << loop.get()->printInputEnable(PrintType::Scala,
                                                      iter_input_control->second.getID())
                          << " <> "
                          << _input_node->printOutputEnable(PrintType::Scala,
                                                            _output_index.getID())
                          << "\n\n";
          }
        } 

        //store control
        std::vector<LSNode*> store_nodes;
        for (auto& ins_node : this->op_list) {
          if(ins_node->getOperationType() == OperationNode::OperationType::StoreType){
            if (ins_node->numDataOutputPort() == 0)
              this->outputHardware << "  " << static_cast<LSNode*>(ins_node.get())->printGround(PrintType::Scala) << "\n\n";

              for(auto iter = ins_node->outputControl_begin(); iter != ins_node->outputControl_end(); iter++){
                if (auto branch_node = static_cast<StateBranchNode*>(iter->first)) {
                  this->outputHardware
                      << "  "
                      << branch_node->printInputEnable(
                            PrintType::Scala,
                            branch_node->returnControlInputPortIndex(ins_node.get()).getID())
                      << " <> "
                      << static_cast<LSNode*>(ins_node.get())->printOutputEnable(
                            PrintType::Scala,
                            ins_node.get()->findControlOutputNode(iter->first)->second.getID())
                      << "\n\n";
                }
              }

          }
        }
        break;
  }
}
/**
 * Print the Connection.
 */
void Graph::printConnection(PrintType _pt) {
  switch (_pt) {
        case PrintType::Scala:
        this->outputHardware << helperScalaPrintHeader("Printing Connection.");

        for(auto &exe: this->exe_block_list){
           int i=0;
           for (auto iter = exe.get()->merge_begin(); iter != exe.get()->merge_end(); iter++) {
            this->outputHardware << "  " << (*iter)->printMaskInput(PrintType::Scala) << " <> "
                      << exe->printMaskOutput(PrintType::Scala, i++)
                      << "\n\n";
            }
        }
        for (auto& _data_edge : this->edge_list) {
          if (_data_edge->getType() == Edge::DataTypeEdge) {
            this->outputHardware << "  "
                          << _data_edge->getTar().first->printInputData(
                                PrintType::Scala, _data_edge->getTar().second.getID())
                          << " <> "
                          << _data_edge->getSrc().first->printOutputData(
                                PrintType::Scala, _data_edge->getSrc().second.getID())
                          << "\n\n";
          }
        }
        auto cache = this->getMemoryUnit();
        for(auto iter = cache->read_req_begin(); iter != cache->read_req_end(); iter++){
          this->outputHardware << "  "
                      << cache->printMemReadInput(
                             PrintType::Scala,
                             cache->returnMemoryReadInputPortIndex(iter->first).getID())
                      << " <> "
                      << iter->first->printMemReadOutput(
                             PrintType::Scala,
                             iter->first->returnMemoryReadOutputPortIndex(cache).getID())
                      << "\n\n";

        this->outputHardware << "  "
                      << iter->first->printMemReadInput(
                             PrintType::Scala,
                             iter->first->returnMemoryReadInputPortIndex(cache).getID())
                      << " <> "
                      << cache->printMemReadOutput(
                             PrintType::Scala,
                             cache->returnMemoryReadOutputPortIndex(iter->first).getID())
                      << "\n\n";

        }
        for(auto iter = cache->write_req_begin(); iter != cache->write_req_end(); iter++){
        this->outputHardware << "  "
                      << cache->printMemWriteInput(
                             PrintType::Scala,
                             cache->returnMemoryWriteInputPortIndex(iter->first).getID())
                      << " <> "
                      << iter->first->printMemWriteOutput(
                             PrintType::Scala,
                             iter->first->returnMemoryWriteOutputPortIndex(cache).getID())
                      << "\n\n";

        this->outputHardware << "  "
                      << iter->first->printMemWriteInput(
                             PrintType::Scala,
                             iter->first->returnMemoryWriteInputPortIndex(cache).getID())
                      << " <> "
                      << cache->printMemWriteOutput(
                             PrintType::Scala,
                             cache->returnMemoryWriteOutputPortIndex(iter->first).getID())
                      << "\n\n";
      }
      //TODO: ALLOC OP. DMM.
      //Block -> enable
      this->outputHardware << helperScalaPrintHeader("Printing Execution Block Enable.");
        for (auto &exe : this->exe_block_list) {
          for(auto iter = exe.get()->const_begin(); iter != exe.get()->const_end(); iter++){
            this->outputHardware
              << "  " << (*iter)->printInputEnable(PrintType::Scala) << " <> "
              << exe->printOutputEnable(
                     PrintType::Scala,
                     exe->returnControlOutputPortIndex(*iter).getID())
              << "\n\n";
          }
          for(auto iter = exe.get()->operation_node_begin(); iter != exe.get()->operation_node_end(); iter++){
            llvm::outs() << (*iter)->getName();
            auto operation_node = static_cast<Node *>(*iter);
            auto ff = std::find_if(
                operation_node->inputControl_begin(),
                operation_node->inputControl_end(),
                [&exe](auto& arg) -> bool { return exe.get() == &*arg.first; });
            if (ff == operation_node->inputControl_end()) 
              assert(!"Couldn't find the control edge\n");
            this->outputHardware
              << "  " << operation_node->printInputEnable(PrintType::Scala) << " <> "
              << exe->printOutputEnable(
                     PrintType::Scala,
                     exe->returnControlOutputPortIndex(operation_node).getID())
              << "\n\n";
          }
        }
      break;

      // default: assert(!"We don't support the other types right now");
  }


}


/**
 * Print the Connection.
 */
void Graph::printLoopConnection(PrintType _pt) {
  switch (_pt) {
        case PrintType::Scala:
        this->outputHardware << helperScalaPrintHeader("Loop dependencies.");
        this->outputHardware << helperScalaPrintHeader("Input Data dependencies.");

        for (auto& loop_node : this->loop_nodes) {
          uint32_t c = 0;
          for(auto iter=loop_node->live_in_sets_begin(); iter != loop_node->live_in_sets_end(); iter++){
            if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LoopLiveIn)
              continue;
            for(auto iter_input_data = iter->get()->inputDataport_begin(); iter_input_data != iter->get()->inputDataport_end(); iter_input_data++){
              this->outputHardware
                  << "  " << iter->get()->printInputData(PrintType::Scala, c++) << " <> "
                  << iter_input_data->first->printOutputData(
                        PrintType::Scala,
                        iter_input_data->first->returnDataOutputPortIndex(iter->get()).getID())
                  << "\n\n";
            }
          }
        }

        this->outputHardware << helperScalaPrintHeader("Live-in dependencies.");
        for (auto& loop_node : this->loop_nodes) {
          for(auto iter=loop_node->live_in_sets_begin(); iter != loop_node->live_in_sets_end(); iter++){

            if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LoopLiveIn)
              continue;
            for(auto iter_data = iter->get()->outputDataport_begin(); iter_data != iter->get()->outputDataport_end(); iter_data++){

              if (isa<ArgumentNode>(iter_data->first))
                continue;
              this->outputHardware
                  << "  "
                  << iter_data->first->printInputData(
                        PrintType::Scala,
                        iter_data->first->returnDataInputPortIndex(iter->get()).getID())
                  << " <> "
                  << iter->get()->printOutputData(
                        PrintType::Scala,
                        iter->get()->returnDataOutputPortIndex(iter_data->first).getID())
                  << "\n\n";
            }
          }
        }

        this->outputHardware << helperScalaPrintHeader("Output Data dependencies.");

        for (auto& loop_node : this->loop_nodes) {
          uint32_t c = 0;
          for(auto iter=loop_node->live_out_sets_begin(); iter != loop_node->live_out_sets_end(); iter++){
            if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LoopLiveOut)
              continue;
            for(auto iter_input_data = iter->get()->inputDataport_begin(); iter_input_data != iter->get()->inputDataport_end(); iter_input_data++){
              this->outputHardware
                  << "  " << iter->get()->printInputData(PrintType::Scala, c++) << " <> "
                  << iter_input_data->first->printOutputData(
                        PrintType::Scala,
                        iter_input_data->first->returnDataOutputPortIndex(iter->get()).getID())
                  << "\n\n";
            }
          }
        }

        this->outputHardware << helperScalaPrintHeader("Live-out dependencies.");
        for (auto& loop_node : this->loop_nodes) {
          for(auto iter=loop_node->live_out_sets_begin(); iter != loop_node->live_out_sets_end(); iter++){

            if (iter->get()->getArgType() != ArgumentNode::ArgumentType::LoopLiveOut)
              continue;
            for(auto iter_data = iter->get()->outputDataport_begin(); iter_data != iter->get()->outputDataport_end(); iter_data++){

              if (isa<ArgumentNode>(iter_data->first))
                continue;
              this->outputHardware
                  << "  "
                  << iter_data->first->printInputData(
                        PrintType::Scala,
                        iter_data->first->returnDataInputPortIndex(iter->get()).getID())
                  << " <> "
                  << iter->get()->printOutputData(
                        PrintType::Scala,
                        iter->get()->returnDataOutputPortIndex(iter_data->first).getID())
                  << "\n\n";
            }
          }
        }

        this->outputHardware << helperScalaPrintHeader("Carry dependencies");
        for (auto& loop_node : this->loop_nodes) {
          uint32_t c = 0;
          for(auto carry_value=loop_node->carry_depen_sets_begin(); carry_value != loop_node->carry_depen_sets_end(); carry_value++){
          // for (auto& carry_value : loop_node->getCarryDepenSets()) {
            if (carry_value->get()->getArgType() != ArgumentNode::ArgumentType::CarryDependency)
              continue;
            for(auto iter = carry_value->get()->inputDataport_begin(); iter != carry_value->get()->inputDataport_end(); iter++){  
              this->outputHardware
                  << "  "
                  << carry_value->get()->printInputData(
                        PrintType::Scala, c++)
                  << " <> "
                  << iter->first->printOutputData(
                        PrintType::Scala,
                        iter->first->returnDataOutputPortIndex(carry_value->get()).getID())
                  << "\n\n";
            }
          // }
          
            for(auto iter = carry_value->get()->outputDataport_begin(); iter != carry_value->get()->outputDataport_end(); iter++){  
              if (isa<ArgumentNode>(iter->first))
                continue;
              this->outputHardware
                  << "  "
                  << iter->first->printInputData(
                        PrintType::Scala,
                        iter->first->returnDataInputPortIndex(carry_value->get()).getID())
                  << " <> "
                  << carry_value->get()->printOutputData(
                        PrintType::Scala,
                        carry_value->get()->returnDataOutputPortIndex(iter->first).getID())
                  << "\n\n";
            }
          }
        }
        break;
  }
}
void Graph::printScalaEnd(PrintType _pt){
  switch (_pt) {
      case PrintType::Scala:
      std::string function_file = "\n\nimport java.io.{File, FileWriter}\n\n"
                      "object $class_nameTop extends App {\n"
                      "  implicit val p = new WithAccelConfig ++ new WithTestConfig\n"
                      "  val verilogString = getVerilogString(new $module_name())\n"
                      "  val filePath = \"RTL/$module_name.v\"\n"
                      "  val writer = new PrintWriter(filePath)\n"
                      "  try { \n "
                      "     writer.write(verilogString)\n"
                      "  } finally {\n"
                      "    writer.close()\n"
                      "  }\n}";
    strReplace(function_file, "$class_name", this->graph_info.Name);
    strReplace(function_file, "$module_name", this->graph_info.Name);

    this->outputHardware << function_file;
      
    break;
    
  }
}