// Reference to
// https://github.com/esa-tu-darmstadt/spn-compiler/blob/master/mlir/include/Conversion/LoSPNtoCPU/Vectorization/SLP/VectorizationPatternMacth.h

#ifndef MLIR_SLP_PATTERN_MATCH_H
#define MLIR_SLP_PATTERN_MATCH_H

#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Vectorization/BlockCostModel.h"
#include "heteacc/Vectorization/GraphConversion.h"
#include "heteacc/Vectorization/SLPGraph.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
namespace mlir {
namespace heteacc {

class ConversionManager;
class PatternVisitor;

/// An SLP vectorization pattern's purpose is describing when it is applicable
/// to a given SLP vector and rewriting such an SLP vector by creating
/// corresponding SIMD instructions.
class SLPVectorizationPattern {
public:
  /// The conversion manager is required for setting up appropriate insertion
  /// points and for keeping track of vector -> SIMD operation mappings.
  explicit SLPVectorizationPattern(ConversionManager &conversionManager);
  virtual ~SLPVectorizationPattern() = default;
  /// Create SIMD instructions for the given superword. Uses the provided
  /// rewriter for operation creation. Also automatically sets up appropriate
  /// insertion points.
  void rewriteSuperword(Superword *superword, RewriterBase &rewriter);
  /// Returns true if the pattern is applicable to the provided superword,
  /// otherwise false.
  virtual LogicalResult match(Superword *superword) = 0;
  /// Accepts the provided visitor and calls the visitor's method that
  /// corresponds to the pattern class.
  virtual void accept(PatternVisitor &visitor,
                      Superword const *superword) const = 0;

  virtual std::string getName() const { return name; }

protected:
  std::string name;
  virtual Value rewrite(Superword *superword, RewriterBase &rewriter) = 0;
  ConversionManager &conversionManager;
};

/// Vectorization patterns that are applicable to specific operations only
/// (additions, ..). An optional pack of compatible op types can be supplied to
/// create vectors with more than one op code. A simple use case would be
/// matching vectors containing both MLIR constants and LoSPN constants, as
/// these are lowered to constants anyways.
template <typename SourceOp, typename... CompatibleOps>
class OpSpecificVectorizationPattern : public SLPVectorizationPattern {
  using SLPVectorizationPattern::SLPVectorizationPattern;

public:
  LogicalResult match(Superword *superword) override {
    for (auto value : *superword) {
      if (auto op =
              compatibleOperationOrNull<SourceOp, CompatibleOps...>(value)) {
        if (superword->numOperands() == op->getNumOperands()) {
          continue;
        }
      }
      return failure();
    }
    return success();
  }

  template <typename LastOp> Operation *compatibleOperationOrNull(Value value) {
    if (auto op = value.getDefiningOp<LastOp>()) {
      return op;
    }
    return nullptr;
  }

  template <typename FirstOp, typename SecondOp, typename... Ops>
  Operation *compatibleOperationOrNull(Value value) {
    if (auto op = compatibleOperationOrNull<FirstOp>(value)) {
      return op;
    }
    return compatibleOperationOrNull<SecondOp, Ops...>(value);
  }
};

/// Vectorization patterns for computation in normal space.
template <typename SourceOp, typename... CompatibleOps>
class NormalSpaceVectorizationPattern
    : public OpSpecificVectorizationPattern<SourceOp, CompatibleOps...> {
  using OpSpecificVectorizationPattern<
      SourceOp, CompatibleOps...>::OpSpecificVectorizationPattern;

public:
  LogicalResult match(Superword *superword) override {
    if (failed(
            OpSpecificVectorizationPattern<SourceOp, CompatibleOps...>::match(
                superword))) {
      return failure();
    }
    return success();
  }
};

// === Op-agnostic patterns === //

struct BroadcastSuperword : public SLPVectorizationPattern {
  using SLPVectorizationPattern::SLPVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

struct VectorizeReduction : public SLPVectorizationPattern {
  using SLPVectorizationPattern::SLPVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

// === Op-specific patterns === //

/// Vectorization pattern for creating SIMD constants.
struct VectorizeConstant
    : public NormalSpaceVectorizationPattern<arith::ConstantOp> {
  using NormalSpaceVectorizationPattern<
      arith::ConstantOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating vector load operations.
struct CreateConsecutiveLoad
    : public OpSpecificVectorizationPattern<memref::LoadOp> {
  using OpSpecificVectorizationPattern<
      memref::LoadOp>::OpSpecificVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating gather load operations.
struct CreateGatherLoad
    : public OpSpecificVectorizationPattern<memref::LoadOp> {
  using OpSpecificVectorizationPattern<
      memref::LoadOp>::OpSpecificVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating vector store operations.
struct CreateConsecutiveStore
    : public OpSpecificVectorizationPattern<memref::StoreOp> {
  using OpSpecificVectorizationPattern<
      memref::StoreOp>::OpSpecificVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, Value vector, RewriterBase &rewriter);
  Value rewrite(Superword *superword, RewriterBase &rewriter) override {
    return Value{};
  }
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

// === Op-specific normal space patterns === //

/// Vectorization pattern for creating normal space vector additions.
struct VectorizeAddI : public NormalSpaceVectorizationPattern<arith::AddIOp> {
  using NormalSpaceVectorizationPattern<
      arith::AddIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating normal space vector additions.
struct VectorizeAddF : public NormalSpaceVectorizationPattern<arith::AddFOp> {
  using NormalSpaceVectorizationPattern<
      arith::AddFOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating normal space vector multiplications.
struct VectorizeMulI : public NormalSpaceVectorizationPattern<arith::MulIOp> {
  using NormalSpaceVectorizationPattern<
      arith::MulIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for creating normal space vector multiplications.
struct VectorizeMulF : public NormalSpaceVectorizationPattern<arith::MulFOp> {
  using NormalSpaceVectorizationPattern<
      arith::MulFOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for vector comparisons.
struct VectorizeCmpI : public OpSpecificVectorizationPattern<arith::CmpIOp> {
  using OpSpecificVectorizationPattern<
      arith::CmpIOp>::OpSpecificVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for integer subtraction.
struct VectorizeSubI : public NormalSpaceVectorizationPattern<arith::SubIOp> {
  using NormalSpaceVectorizationPattern<
      arith::SubIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for float subtraction.
struct VectorizeSubF : public NormalSpaceVectorizationPattern<arith::SubFOp> {
  using NormalSpaceVectorizationPattern<
      arith::SubFOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for float comparisons.
struct VectorizeCmpF : public OpSpecificVectorizationPattern<arith::CmpFOp> {
  using OpSpecificVectorizationPattern<
      arith::CmpFOp>::OpSpecificVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for bitwise AND.
struct VectorizeAndI : public NormalSpaceVectorizationPattern<arith::AndIOp> {
  using NormalSpaceVectorizationPattern<
      arith::AndIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for bitwise OR.
struct VectorizeOrI : public NormalSpaceVectorizationPattern<arith::OrIOp> {
  using NormalSpaceVectorizationPattern<
      arith::OrIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for bitwise XOR.
struct VectorizeXOrI : public NormalSpaceVectorizationPattern<arith::XOrIOp> {
  using NormalSpaceVectorizationPattern<
      arith::XOrIOp>::NormalSpaceVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for index_cast operations.
struct VectorizeIndexCast
    : public OpSpecificVectorizationPattern<arith::IndexCastOp> {
  using OpSpecificVectorizationPattern<
      arith::IndexCastOp>::OpSpecificVectorizationPattern;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for indirect (non-linear) loads via
/// dataflow.vector.load.
struct VectorizeIndirectLoad
    : public OpSpecificVectorizationPattern<memref::LoadOp> {
  using OpSpecificVectorizationPattern<
      memref::LoadOp>::OpSpecificVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, RewriterBase &rewriter) override;
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

/// Vectorization pattern for indirect (non-linear) stores via
/// dataflow.vector.store.
struct VectorizeIndirectStore
    : public OpSpecificVectorizationPattern<memref::StoreOp> {
  using OpSpecificVectorizationPattern<
      memref::StoreOp>::OpSpecificVectorizationPattern;
  LogicalResult match(Superword *superword) override;
  Value rewrite(Superword *superword, Value vector, RewriterBase &rewriter);
  Value rewrite(Superword *superword, RewriterBase &rewriter) override {
    return Value{};
  }
  void accept(PatternVisitor &visitor,
              Superword const *superword) const override;
};

} // namespace heteacc
} // namespace mlir

#endif // MLIR_SLP_PATTERN_MATCH_H
