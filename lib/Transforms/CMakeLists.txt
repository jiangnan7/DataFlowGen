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
  Memory/MemoryScheduling.cpp

  Misc/Utils.cpp
  Misc/HybridAnalysis.cpp
  Misc/EvaluationModel.cpp
  Misc/VecUtils.cpp

  Vectorization/AutoVectorization.cpp
  Vectorization/BlockCostModel.cpp
  Vectorization/GraphConversion.cpp
  Vectorization/OperationFusion.cpp
  Vectorization/ScoreModel.cpp
  Vectorization/Seeding.cpp
  Vectorization/SLPGraph.cpp
  Vectorization/SLPGraphBuilder.cpp
  Vectorization/SLPVectorizationPatternMatch.cpp

  DEPENDS
  MLIRVectorizationOpsIncGen
  MLIRHETEACCTransformsIncGen
)
