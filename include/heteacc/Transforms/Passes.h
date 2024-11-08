#ifndef VECTORIZATION_PASSES_H
#define VECTORIZATION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "heteacc/InitAllDialects.h"
#include <memory>
namespace mlir {
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir::heteacc {


//Loop

std::unique_ptr<Pass> createAffineLoopPermutationPass();
std::unique_ptr<Pass> createAffineLoopPerfectionPass();

std::unique_ptr<Pass> createGenerateDataflowPass();
std::unique_ptr<Pass> createOptimizeDataflowPass();
std::unique_ptr<Pass> createEnhancedCDFGPass();
std::unique_ptr<Pass> createHybridBranchPredictionPass();
//Vector
std::unique_ptr<Pass> createOperationFusionPass();
std::unique_ptr<Pass> createSimplifyVectorMemrefAccessPass();

std::unique_ptr<Pass> createGenerateGEPPass();
std::unique_ptr<Pass> createAnalyzeMemrefAddressPass();

std::unique_ptr<Pass> createEvaluationModelPass(std::string TargetSpec = "./config.json");

std::unique_ptr<Pass> createGraphInitPass(std::string hlsTopFunc = "main");

/// Generate the code for registering passes.
// #define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
// #define GEN_PASS_DECL
#include "heteacc/Transforms/Passes.h.inc"

} // namespace 

#endif // VECTORIZATION_PASSES_H

