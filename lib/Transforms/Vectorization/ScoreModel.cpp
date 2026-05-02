#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IntegerSet.h"

#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
#include "heteacc/Vectorization/ScoreModel.h"

using namespace mlir;
using namespace heteacc;

// === ScoreModel === //

ScoreModel::ScoreModel(unsigned lookAhead) : lookAhead{lookAhead} {}

// === PorpodasModel === //

Value PorpodasModel::getBest(Value value, ArrayRef<Value> candidates) {
  Value best = nullptr;
  // Look-ahead on various levels
  // TODO: when the level is increased, we recompute everything from the level
  // before. that could be changed...
  for (size_t level = 1; level <= lookAhead; ++level) {
    // Best is the candidate with max score
    unsigned bestScore = 0;
    llvm::SmallSet<unsigned, 4> scores;
    for (auto candidate : candidates) {
      // Get the look-ahead score
      unsigned score = getLookAheadScore(value, candidate, level);
      if (scores.empty() || score > bestScore) {
        best = candidate;
        bestScore = score;
        scores.insert(score);
      }
    }
    // If found best at level don't go deeper
    if (best != nullptr && scores.size() > 1) {
      break;
    }
  }
  assert(best && "no best value found");
  return best;
}

unsigned PorpodasModel::getLookAheadScore(Value last, Value candidate,
                                          unsigned maxLevel) const {
  auto *lastOp = last.getDefiningOp();
  auto *candidateOp = candidate.getDefiningOp();
  if (!lastOp || !candidateOp) {
    return last == candidate;
  }
  if (lastOp->getName() != candidateOp->getName()) {
    return 0;
  }
  //   if (auto lhsLoad = dyn_cast<SPNBatchRead>(lastOp)) {
  //     // We know both operations share the same opcode.
  //     auto rhsLoad = cast<SPNBatchRead>(candidateOp);
  //     if (lhsLoad.batchMem() == rhsLoad.batchMem() && lhsLoad.dynamicIndex()
  //     == rhsLoad.dynamicIndex()) {
  //       if (lhsLoad.staticIndex() + 1 == rhsLoad.staticIndex()) {
  //         // Returning 3 prefers consecutive loads to gather loads and
  //         broadcast loads. return 3;
  //       }
  //       // Returning 2 prefers gather loads to broadcast loads.
  //       if (lhsLoad.staticIndex() != rhsLoad.staticIndex()) {
  //         return 2;
  //       }
  //       // Broadcast load.
  //       return 1;
  //     } else {
  //       return 0;
  //     }
  //   }
  if (maxLevel == 0) {
    return 1;
  }
  unsigned scoreSum = 0;
  for (auto &lastOperand : getOperands(last)) {
    for (auto &candidateOperand : getOperands(candidate)) {
      scoreSum +=
          getLookAheadScore(lastOperand, candidateOperand, maxLevel - 1);
    }
  }
  return scoreSum;
}
