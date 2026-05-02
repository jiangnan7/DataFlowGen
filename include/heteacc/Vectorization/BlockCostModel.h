// Reference to
// https://github.com/esa-tu-darmstadt/spn-compiler/blob/master/mlir/include/Conversion/LoSPNtoCPU/Vectorization/SLP/CostModel.h
#ifndef MLIR_VECTORIZATION_BLOCK_COST_MODEL_H
#define MLIR_VECTORIZATION_BLOCK_COST_MODEL_H

#include "mlir/IR/PatternMatch.h"

#include "heteacc/Vectorization/GraphConversion.h"
#include "heteacc/Vectorization/PatternVisitor.h"
#include "heteacc/Vectorization/SLPGraph.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
#include "heteacc/Vectorization/VectorizationPatternMacth.h"
namespace mlir {
namespace heteacc {

class SLPPatternApplicator;
class ConversionState;
/// The cost model calculates the profitability of the vectorization. It is also
/// used to determine the most profitable pattern for each superword, depending
/// on the current vectorization state. It extends the PatternVisitor class to
/// make pattern cost computations easier.
class CostModel : public PatternVisitor {
public:
  /// Computes the cost of a scalar value recursively, i.e.
  /// \code cost(value) = cost(value op) + sum(cost(operand i)) \endcode for
  /// each operand \em i. Depending on the vectorization state, cost may be zero
  /// if the value has already been marked as computed.
  double getScalarCost(Value value);
  /// Computes the cost of a superword if the provided pattern were to be
  /// applied. The computation is done in a similar fashion to
  /// CostModel::getScalarCost(Value) i.e. \code cost(superword, pattern) =
  /// cost(pattern) + sum(cost(superword i, superword i pattern)) \endcode for
  /// each operand superword i.
  double getSuperwordCost(Superword *superword,
                          SLPVectorizationPattern *pattern);
  /// Determines if an extraction for the provided value would be more
  /// profitable than computing it in a scalar fashion.
  bool isExtractionProfitable(Value value);
  /// Set the conversion state which keeps track of values that have been marked
  /// as computed. This is required for accurate, conversion state-aware cost
  /// computations.
  void setConversionState(std::shared_ptr<ConversionState> newConversionState);
  /// Computes the cost of an entire block. The set of dead ops is necessary to
  /// only consider operations that are actually relevant.
  double getBlockCost(Block *block,
                      SmallPtrSetImpl<Operation *> const &deadOps) const;

  double computeScalarCost(Value value) const;

  void visitDefault(SLPVectorizationPattern const *pattern,
                    Superword const *superword) override;
  void visit(BroadcastSuperword const *pattern,
             Superword const *superword) override;
  void visit(VectorizeConstant const *pattern,
             Superword const *superword) override;

protected:
  double cost;
  /// For computation of required scalar values.
  LeafPatternVisitor leafVisitor;
  std::shared_ptr<ConversionState> conversionState;

private:
  void updateCost(Value value, double newCost, bool updateUses);
  double getExtractionCost(Value value) const;
  static constexpr double MAX_COST = std::numeric_limits<double>::max();
  DenseMap<Value, double> cachedScalarCost;
  SLPPatternApplicator const &patternApplicator;

public:
  /// The base constructor for every cost model. Requires a pattern applicator
  /// because the cost model needs to know which patterns are applicable.
  explicit CostModel(SLPPatternApplicator const &applicator)
      : cost(0), patternApplicator(applicator) {}
};

} // namespace heteacc
} // namespace mlir

#endif // MLIR_VECTORIZATION_BLOCK_COST_MODEL_H
