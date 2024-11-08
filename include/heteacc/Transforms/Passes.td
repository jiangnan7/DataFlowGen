#ifndef TRANSORMS_PASSES_TD
#define TRANSORMS_PASSES_TD

include "mlir/Pass/PassBase.td"



def GenerateDataflow : Pass<"generate-dataflow", "func::FuncOp"> {
  let summary = "Generate dataflow hierarchy from affine loops";
  let constructor = "mlir::heteacc::createGenerateDataflowPass()";
}

def OptimizeDataflow : Pass<"optimize-dataflow", "func::FuncOp"> {
  let summary = "Simplify load and store access";
  let description = [{
    Analyze memory addresses to simplify access and reduce resource consumption.
  }];
  let constructor = "mlir::heteacc::createOptimizeDataflowPass()";
}

def EnhancedCDFG : Pass<"enhanced-cdfg", "func::FuncOp"> {
  let summary = " ";
  let constructor = "mlir::heteacc::createEnhancedCDFGPass()";
}

def HybridBranchPrediction : Pass<"hybird-branch-prediction", "func::FuncOp"> {
  let summary = " ";
  let constructor = "mlir::heteacc::createHybridBranchPredictionPass()";
}


def GraphInit : Pass<"graph-init", "func::FuncOp"> {
  let summary = "Preprocess the functions subsequent  optimizations";
  let constructor = "mlir::heteacc::createGraphInitPass()";

  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"\"main\"",
           "The top function for HLS synthesis">
  ];
}

//===----------------------------------------------------------------------===//
// Loop-related Passes
//===----------------------------------------------------------------------===//
def AffineLoopPermutation :
      Pass<"affine-loop-permutation", "func::FuncOp"> {
  let summary = "Optimize the order of affine loop nests";
  let description = [{
    This pass will optimize the order of perfect affine loop nests through
    polyhedral-based dependency analysis. The objective of this pass is to
    maximize the distance of loop-carried dependencies.
  }];
  let constructor = "mlir::heteacc::createAffineLoopPermutationPass()";
}

def AffineLoopPerfection :
      Pass<"affine-loop-perfection", "func::FuncOp"> {
  let summary = "Try to perfect a nested loop";
  let description = [{
    This pass will try to perfect all affine loops. Specifically, this pass will
    move operations under non-innermost loops into innermost loop and create if
    statements to ensure the correctness of the transformation.
  }];
  let constructor = "mlir::heteacc::createAffineLoopPerfectionPass()";
}


//===----------------------------------------------------------------------===//
// Vector-related Passes
//===----------------------------------------------------------------------===//
def EvaluationModel : Pass<"evaluation-model", "func::FuncOp"> {
  let summary = "Build an evaluation model to analyze the efficiency of vectorization.";
  let constructor = "mlir::heteacc::createEvaluationModelPass()";
  let options = [
      Option<"targetSpec", "target-spec", "std::string",
            /*default=*/"\"./config.json\"",
            "File path: target backend specifications and configurations">
    ];
}


//===----------------------------------------------------------------------===//
// Vector-related Passes
//===----------------------------------------------------------------------===//
def OperationFusion : Pass<"operation-fusion", "func::FuncOp"> {
  let summary = "Fusion of vector-related operations for greater efficiency";
  let constructor = "mlir::heteacc::createOperationFusionPass()";

}

def SimplifyVectorMemrefAccess :
      Pass<"simplify-vector-memref-access", "func::FuncOp"> {
  let summary = "Simplify load and store operations";
  let description = [{
    This simplify-memref-access pass will eliminate the known redundant load and
    store operations for simplifying the memref access.
  }];
  let constructor = "mlir::heteacc::createSimplifyVectorMemrefAccessPass()";
}


def AnalyzeMemrefAddress :
      Pass<"analyze-memref-address", "func::FuncOp"> {
  let summary = "Analyze address of affine access";
  let description = [{
    Extract affine access information and add to L/S as attributes.
  }];
  let constructor = "mlir::heteacc::createAnalyzeMemrefAddressPass()";
}

def GenerateGEP :
      Pass<"generate-GEP", "func::FuncOp"> {
  let summary = "generate GEP for affine access";
  let description = [{
    generate my GEP.
  }];
  let constructor = "mlir::heteacc::createGenerateGEPPass()";
}

#endif // TRANSORMS_PASSES_TD