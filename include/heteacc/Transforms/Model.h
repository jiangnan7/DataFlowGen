#ifndef MODEL_H
#define MODEL_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/JSON.h"


namespace mlir{
namespace heteacc{


// Get the operator name to latency/DSP usage mapping.
void getLatencyMap(llvm::json::Object *config,
                   llvm::StringMap<int64_t> &latencyMap);
// void getDspUsageMap(llvm::json::Object *config,
//                     llvm::StringMap<int64_t> &dspUsageMap);

// class VectorModel
//     : public 
}   
}
#endif //MODEL_H