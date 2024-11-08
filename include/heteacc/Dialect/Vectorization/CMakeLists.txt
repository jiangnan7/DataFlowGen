
add_mlir_dialect(VectorizationOps vectorization)
add_mlir_doc(VectorizationDialect VectorizationDialect Vectorization/ -gen-dialect-doc)
add_mlir_doc(VectorizationOps VectorizationOps Vectorization/ -gen-op-doc)

# set(LLVM_TARGET_DEFINITIONS Passes.td)
# mlir_tablegen(Passes.h.inc -gen-pass-decls)
# add_public_tablegen_target(MLIRVectorizationPassIncGen)
