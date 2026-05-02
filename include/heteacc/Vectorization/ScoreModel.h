#ifndef MLIR_SCORE_MODEL_H
#define MLIR_SCORE_MODEL_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include "heteacc/Vectorization/BlockCostModel.h"
#include "heteacc/Vectorization/SLPGraph.h"
#include "heteacc/Vectorization/VectorizationPatternMacth.h"

namespace mlir {
namespace heteacc {

/// A ScoreModel is responsible for assigning scores to candidate operations
/// during the operation reordering step in the SLP graph building phase. Every
/// such model must implement a single method that retrieves a best candidate
/// from a set of candidate operations, depending on the lookahead.
class ScoreModel {
public:
  explicit ScoreModel(unsigned lookAhead);
  virtual ~ScoreModel() = default;
  /// Returns the candidate for the given value that best matches it based on
  /// the model's lookahead.
  virtual Value getBest(Value value, ArrayRef<Value> candidates) = 0;

protected:
  unsigned lookAhead;
};

/// A score model based on Porpodas's Look-Ahead SLP
/// (https://dl.acm.org/doi/10.1145/3168807). Some slight adaptions were done to
/// the score computation such that gather loads and broadcast loads are also
/// taken into account (consecutive score > gather score > broadcast score).
class PorpodasModel : public ScoreModel {
  using ScoreModel::ScoreModel;

public:
  Value getBest(Value value, ArrayRef<Value> candidates) override;

private:
  unsigned getLookAheadScore(Value last, Value candidate,
                             unsigned maxLevel) const;
};

} // namespace heteacc
} // namespace mlir

#endif // MLIR_SCORE_MODEL_H
