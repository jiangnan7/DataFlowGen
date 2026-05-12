#ifndef MISC_VECTORIZATION_UTILS_H
#define MISC_VECTORIZATION_UTILS_H

#include "heteacc/Dialect/DataFlow/DataFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include <assert.h>
namespace mlir {
namespace heteacc {

/// A description of a (parallelizable) reduction in an affine loop.
struct LoopReduction {
  /// Reduction kind.
  arith::AtomicRMWKind kind;

  /// Position of the iteration argument that acts as accumulator.
  unsigned iterArgPosition;

  /// The value being reduced.
  Value value;
};

using ReductionLoopMap = DenseMap<Operation *, SmallVector<LoopReduction, 2>>;

// Return the number of lanes along the vectorized dimension for the vector
// type. For a multidimensional vector, return the innermost dimension size
inline unsigned getVectorLaneSize(mlir::VectorType type) {
  assert(type.getRank() > 0 && "Cannot handle rank-0 vectors");
  auto dimSize = type.getDimSize(type.getRank() - 1);
  assert(dimSize >= 0 && "Vector dimension cannot be negative");
  return std::max(1u, (unsigned)dimSize);
}

bool applyOpFusion(func::FuncOp func);

bool ofVectorizableType(Value value);

/// Returns true if every value in [begin, end) is vectorizable according to
/// MLIR's vector type rules.
template <typename ValueIterator>
bool ofVectorizableType(ValueIterator begin, ValueIterator end) {
  return std::all_of(begin, end, [&](auto const &value) {
    return ofVectorizableType(value) && !value.template isa<BlockArgument>();
    // return ofVectorizableType(value);
  });
}

/// Returns true if the operation is vectorizable according to both the LoSPN
/// dialect's vectorizable op interface and MLIR's vector type rules.
bool vectorizable(Operation *op);

/// Returns true if the value is vectorizable according to both the LoSPN
/// dialect's vectorizable op interface and MLIR's vector type rules.
bool vectorizable(Value value);

/// Returns true if every value in [begin, end) is vectorizable.
template <typename ValueIterator>
bool vectorizable(ValueIterator begin, ValueIterator end) {
  if (begin->template isa<BlockArgument>() || std::next(begin) == end) {
    return false;
  }
  const auto &name = begin->getDefiningOp()->getName();
  ++begin;
  if (begin->getDefiningOp() == nullptr)
    return false;
  while (begin != end) {
    if (!vectorizable(*begin) || (begin->getDefiningOp()->getName() != name)) {
      return false;
    }
    ++begin;
  }
  return true;
}

bool commutative(Value value);

/// Returns true if every value in [begin, end) represents a commutative
/// operation.
template <typename ValueIterator>
bool commutative(ValueIterator begin, ValueIterator end) {
  while (begin != end) {
    if (!commutative(*begin)) {
      return false;
    }
    ++begin;
  }
  return true;
}

/// Returns true if both values are loads and the second value is consecutive to
/// the first.
bool consecutiveLoads(Value lhs, Value rhs);

bool consecutiveStores(Value lhs, Value rhs);

SmallVector<memref::StoreOp, 8> getAllStoreUsers(Value value);
/// Return the operands of the value as a C++ vector.
SmallVector<Value, 2> getOperands(Value value);

/// Returns true if all values in [begin, end) are loads and consecutive to
/// their predecessor.
template <typename ValueIterator>
bool consecutiveLoads(ValueIterator begin, ValueIterator end) {
  Value previous = *begin;
  if (++begin == end || previous.isa<BlockArgument>() ||
      !dyn_cast<memref::LoadOp>(previous.getDefiningOp())) {
    return false;
  }
  while (begin != end) {
    Value current = *begin;
    if (!consecutiveLoads(previous, current)) {
      return false;
    }
    previous = current;
    ++begin;
  }
  return true;
}

/// Returns true if all values in [begin, end) are stores and consecutive to
/// their predecessor.
template <typename ValueIterator>
bool consecutiveStores(ValueIterator begin, ValueIterator end) {
  Value previous = *begin;
  if (++begin == end || previous.isa<BlockArgument>()) {
    return false;
  }

  while (begin != end) {
    Value current = *begin;
    if (!consecutiveStores(previous, current)) {
      return false;
    }
    previous = current;
    ++begin;
  }
  return true;
}

/// Returns true if all values in [begin, end) are leaf nodes.
/// Note: The original dialect-specific check (LeafNodeInterface) was removed.
/// This function now always returns true as all values are treated as leaves.
template <typename ValueIterator>
bool allLeaf(ValueIterator begin, ValueIterator end) {
  return true;
}

/// Sort all provided values in place by their opcode (i.e. in lexicographical
/// order). An optional smallest opcode can be provided to force certain values
/// to the front.
void sortByOpcode(SmallVectorImpl<Value> &values,
                  Optional<OperationName> smallestOpcode = std::nullopt);

SmallPtrSet<Operation *, 32> computeDeadOps(Block *block);
void computeDepths(ArrayRef<Value> seed, DenseMap<Value, unsigned> &depths);

} // namespace heteacc
} // namespace mlir

#endif // MISC_VECTORIZATION_UTILS_H
