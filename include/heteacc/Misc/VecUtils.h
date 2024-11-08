
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <assert.h>


namespace mlir{
namespace heteacc{


// Return the number of lanes along the vectorized dimension for the vector
// type. For a multidimensional vector, return the innermost dimension size
inline unsigned getVectorLaneSize(VectorType type) {
  assert(type.getRank() > 0 && "Cannot handle rank-0 vectors");
  auto dimSize = type.getDimSize(type.getRank() - 1);
  assert(dimSize >= 0 && "Vector dimension cannot be negative");
  return std::max(1u, (unsigned)dimSize);
}


 bool applyOpFusion(func::FuncOp func);




}
}