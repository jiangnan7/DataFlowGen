
#ifndef DIALECT_DATAFLOW_TD
#define DIALECT_DATAFLOW_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def DataFlow_Dialect : Dialect {
  let name = "dataflow";
  let summary = "An Vec out-of-tree MLIR dialect";
  let description = [{
    vec
  }];
  // let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::heteacc::dataflow";
  // let hasConstantMaterializer = 1;

}


def StageLikeInterface : OpInterface<"StageLikeInterface"> {
  let description = [{
    This interface is used to represent dataflow stage.
  }];

  let methods = [    
    InterfaceMethod<"Return body region of the stage",
      "mlir::Region &", "getBody", (ins), "return $_op.getBody();">,
    InterfaceMethod<"Check whether the stage has hierarchy",
      "bool", "hasHierarchy", (ins), [{
        return $_op.walk([&](StageLikeInterface stage) {
          if (stage != $_op)
            return WalkResult::interrupt();
          return WalkResult::advance();
        }).wasInterrupted();
      }]>,
    InterfaceMethod<"Return whether the value is a stage livein",
                    "bool", "isLivein", (ins "mlir::Value":$value)>,
    InterfaceMethod<"Return the liveins of the stage",
                    "llvm::SmallVector<mlir::Value>", "getLiveins">,
    InterfaceMethod<"Return the internal users of a stage livein",
                    "llvm::SmallVector<mlir::Operation *>", "getLiveinUsers",
                    (ins "mlir::Value":$livein)>,
  ];
}



class DataFlowOp<string mnemonic, list<Trait> traits = []> :
        Op<DataFlow_Dialect, mnemonic, traits>;

class DataFlowAttr<string name, list<Trait> traits = []> :
    AttrDef<DataFlow_Dialect, name, traits>;

class DataFlowType<string name, list<Trait> traits = []> :
    TypeDef<DataFlow_Dialect, name, traits>;


include "heteacc/Dialect/DataFlow/DataFlowOps.td"

#endif // DIALECT_DataFlow_TD