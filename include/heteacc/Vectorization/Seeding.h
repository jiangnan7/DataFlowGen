#ifndef MLIR_SEEDING_H
#define MLIR_SEEDING_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include "heteacc/Vectorization/BlockCostModel.h"
#include "heteacc/Vectorization/SLPGraph.h"

namespace mlir {
namespace heteacc {

/// A seed analysis is responsible for finding seed instructions for SLP
/// vectorization in a program. Every analysis computes a set of available
/// operations once and then continuously updates the set from vectorization
/// iteration to vectorization iteration.
class SeedAnalysis {

public:
  explicit SeedAnalysis(Operation *rootOp, unsigned width)
      : rootOp{rootOp}, width{width} {}
  virtual ~SeedAnalysis() = default;
  /// Get the next seed.
  virtual SmallVector<Value, 4> next();
  /// Notify the analysis that an SLP graph consisting of the provided
  /// superwords has successfully been converted. The operations contained in
  /// the superwords should not appear in subsequent seeds.
  void update(ArrayRef<Superword *> convertedSuperwords);

protected:
  virtual void computeAvailableOps() = 0;
  virtual SmallVector<Value, 4> nextSeed() const = 0;
  Operation *const rootOp;
  unsigned const width;
  // SetVector to make seeding deterministic from run to run.
  llvm::SmallSetVector<Operation *, 32> availableOps;

private:
  bool availableComputed = false;
};

class TopDownAnalysis : public SeedAnalysis {

public:
  TopDownAnalysis(Operation *rootOp, unsigned width);

protected:
  void computeAvailableOps() override;
  SmallVector<Value, 4> nextSeed() const override;
};

} // namespace heteacc
} // namespace mlir

#endif // MLIR_SEEDING_H
