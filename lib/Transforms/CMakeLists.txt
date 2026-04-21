add_mlir_dialect_library(MLIRHLSTransforms

  Dataflow/GenerateDataflow.cpp
  Dataflow/OptimizeDataflow.cpp
  Dataflow/EnhancedCDFG.cpp
  Dataflow/HybridBranchPrediction.cpp
  Graph/GraphGen.cpp
  Graph/Graph.cpp
  Graph/Node.cpp
  Graph/Utils.cpp
  Graph/DumpChisel.cpp
  Graph/DependencyAnalyze.cpp

  Loop/AffineLoopPermutation.cpp
  Loop/AffineLoopPerfection.cpp
  Vectorization/OperationFusion.cpp

  Memory/SimplifyVectorMemrefAccess.cpp
  Memory/AnalyzeMemrefAddress.cpp
  Memory/GenerateGEP.cpp
  Misc/Utils.cpp
  Misc/HybridAnalysis.cpp
  Misc/EvaluationModel.cpp

  DEPENDS
  MLIRVectorizationOpsIncGen
  MLIRHETEACCTransformsIncGen
)
