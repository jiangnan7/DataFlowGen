add_mlir_dialect(DataFlowOps dataflow)

set(LLVM_TARGET_DEFINITIONS DataFlow.td)
mlir_tablegen(DataFlowOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(DataFlowOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDataFlowOpsEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS DataFlow.td)
mlir_tablegen(DataFlowOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DataFlowOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRDataFlowOpsInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS DataFlow.td)
mlir_tablegen(DataFlowOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DataFlowOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRDataFlowOpsAttributesIncGen)


# set(LLVM_TARGET_DEFINITIONS Passes.td)
# mlir_tablegen(Passes.h.inc -gen-pass-decls)
# add_public_tablegen_target(MLIRDataFlowPassIncGen)

# set(LLVM_TARGET_DEFINITIONS DataFlowOps.td)
# add_mlir_dialect(DataFlowOps dataflow)
# add_mlir_doc(DataFlowDialect DataFlowDialect DataFlow/ -gen-dialect-doc)
# add_mlir_doc(DataFlowOps DataFlowOps DataFlow/ -gen-op-doc)
