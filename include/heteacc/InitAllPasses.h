//===----------------------------------------------------------------------===//
//
// Jiangnan Li
//
//===----------------------------------------------------------------------===//

#ifndef HLS_INITALLPASSES_H
#define HLS_INITALLPASSES_H

#include "mlir/InitAllPasses.h"

#include "heteacc/Transforms/Passes.h"

namespace mlir {
namespace heteacc {
   
namespace {
#define GEN_PASS_REGISTRATION
#include "heteacc/Transforms/Passes.h.inc"
} // namespace

// Add all the passes.
inline void registerAllPasses() {

    mlir::registerAllPasses();

    registerPasses();

}

} // namespace hls
} // namespace mlir

#endif // HLS_INITALLPASSES_H
