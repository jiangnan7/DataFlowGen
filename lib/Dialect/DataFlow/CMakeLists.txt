add_mlir_dialect_library(MLIRDataFlow
		DataFlowDialect.cpp
		Utils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/heteacc/Dialect/DataFlow

        DEPENDS
        MLIRDataFlowOpsIncGen
		MLIRDataFlowOpsEnumsIncGen
		MLIRDataFlowOpsInterfacesIncGen
		MLIRDataFlowOpsAttributesIncGen
	LINK_LIBS PUBLIC
	MLIRIR
	)
