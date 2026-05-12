// Reference to
// https://github.com/esa-tu-darmstadt/spn-compiler/blob/master/mlir/include/Conversion/LoSPNtoCPU/Vectorization/SLP/CostModel.h
#ifndef MLIR_PATTERN_VISITOR_H
#define MLIR_PATTERN_VISITOR_H

#include "mlir/IR/PatternMatch.h"

#include "heteacc/Vectorization/SLPGraph.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
#include "heteacc/Vectorization/VectorizationPatternMacth.h"

namespace mlir {
namespace heteacc {

/// A PatternVisitor can be used to compute information for a superword,
/// depending on the pattern to be applied (e.g. any scalar values a superword
/// requires for every pattern). Every visitor should store computed data
/// internally as no values are returned, i.e. some method should be provided
/// for accessing the information.
class PatternVisitor {
public:
  /// Visit a superword under the BroadcastSuperword pattern.
  virtual void visit(BroadcastSuperword const *pattern,
                     Superword const *superword);

  virtual void visit(VectorizeReduction const *pattern,
                     Superword const *superword);

  /// Visit a superword under the VectorizeConstant pattern.
  virtual void visit(VectorizeConstant const *pattern,
                     Superword const *superword);
  /// Visit a superword under the CreateConsecutiveLoad pattern.
  virtual void visit(CreateConsecutiveLoad const *pattern,
                     Superword const *superword);
  //  Visit a superword under the CreateConsecutiveStore pattern.
  virtual void visit(CreateConsecutiveStore const *pattern,
                     Superword const *superword);
  /// Visit a superword under the CreateGatherLoad pattern.
  virtual void visit(CreateGatherLoad const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeAdd pattern.
  virtual void visit(VectorizeAddI const *pattern, Superword const *superword);
  virtual void visit(VectorizeAddF const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeMul pattern.
  virtual void visit(VectorizeMulI const *pattern, Superword const *superword);
  virtual void visit(VectorizeMulF const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeIndexCast pattern.
  virtual void visit(VectorizeIndexCast const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeIndirectLoad pattern.
  virtual void visit(VectorizeIndirectLoad const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeIndirectStore pattern.
  virtual void visit(VectorizeIndirectStore const *pattern,
                     Superword const *superword);
  /// Visit a superword under the VectorizeCmpI pattern.
  virtual void visit(VectorizeCmpI const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeSubI pattern.
  virtual void visit(VectorizeSubI const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeSubF pattern.
  virtual void visit(VectorizeSubF const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeCmpF pattern.
  virtual void visit(VectorizeCmpF const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeAndI pattern.
  virtual void visit(VectorizeAndI const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeOrI pattern.
  virtual void visit(VectorizeOrI const *pattern, Superword const *superword);
  /// Visit a superword under the VectorizeXOrI pattern.
  virtual void visit(VectorizeXOrI const *pattern, Superword const *superword);

protected:
  virtual ~PatternVisitor() = default;
  /// Default visiting method so that we don't have to override every single
  /// visit method in each visitor.
  virtual void visitDefault(SLPVectorizationPattern const *pattern,
                            Superword const *superword) = 0;
};

/// A LeafPatternVisitor visits leaf superwords of an SLP graph and computes
/// which of their elements need to be computed/available in a scalar fashion
/// beforehand.
class LeafPatternVisitor : public PatternVisitor {
public:
  /// Returns the scalar values whose scalar computation is required for the
  /// superword if the provided pattern were to be applied.
  ArrayRef<Value>
  getRequiredScalarValues(SLPVectorizationPattern const *pattern,
                          Superword const *superword);
  void visit(BroadcastSuperword const *pattern,
             Superword const *superword) override;

protected:
  void visitDefault(SLPVectorizationPattern const *pattern,
                    Superword const *superword) override;

private:
  SmallVector<Value, 4> scalarValues;
};
} // namespace heteacc
} // namespace mlir

#endif // MLIR_PATTERN_VISITOR_H
