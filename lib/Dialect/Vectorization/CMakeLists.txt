add_mlir_dialect_library(MLIRVectorization
		VecDialect.cpp
		VecOps.cpp
		

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/heteacc/Dialect/Vectorization

        DEPENDS
        MLIRVectorizationOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
