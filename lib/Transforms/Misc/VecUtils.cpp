#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include "heteacc/Misc/VecUtils.h"
using namespace mlir;

#define DEBUG_TYPE "auto-vectorization"

bool heteacc::ofVectorizableType(Value value) {
  return VectorType::isValidElementType(value.getType());
}

bool heteacc::vectorizable(Value value) {
  if (auto *definingOp = value.getDefiningOp()) {
    if (!vectorizable(definingOp)) {
      return false;
    }
  }
  return ofVectorizableType(value);
}

bool heteacc::commutative(Value value) {
  return value.getDefiningOp() &&
         value.getDefiningOp()->hasTrait<OpTrait::IsCommutative>();
}

bool heteacc::vectorizable(Operation *op) {

  if (!op->hasTrait<mlir::OpTrait::OneResult>())
    return false;

  if (!ofVectorizableType(op->getResult(0)))
    return false;

  // Constants are always vectorizable.
  if (op->hasTrait<mlir::OpTrait::ConstantLike>())
    return true;

  // IndexCastOp is vectorizable (used in indirect access patterns).
  if (isa<arith::IndexCastOp>(op))
    return true;

  // Arithmetic operations are vectorizable if their result type is valid.
  if (isa<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
          arith::SubIOp, arith::SubFOp, arith::DivSIOp, arith::DivUIOp,
          arith::DivFOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
          arith::ShRUIOp, arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
          arith::MaxUIOp, arith::MinFOp, arith::MaxFOp>(op)) {
    // Exclude index-typed operations (used for address computation).
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<mlir::IndexType>())
        return false;
    }
    return true;
  }

  // Memory operations: require affine metadata or slp.group for stride info.
  if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    if (op->hasAttr("affineCoeff") || op->hasAttr("slp.group"))
      return true;
    // Allow loads/stores with simple index patterns (single linear index).
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      if (loadOp.getIndices().size() == 1)
        return true;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      if (storeOp.getIndices().size() == 1)
        return true;
    }
    return false;
  }

  // Ops explicitly marked with slp.group are vectorizable.
  if (op->hasAttr("slp.group"))
    return true;

  return false;
}

bool heteacc::consecutiveLoads(Value lhs, Value rhs) {
  if (lhs == rhs || !lhs.getDefiningOp() || !rhs.getDefiningOp()) {
    return false;
  }
  auto lhsLoad = dyn_cast<memref::LoadOp>(lhs.getDefiningOp());
  auto rhsLoad = dyn_cast<memref::LoadOp>(rhs.getDefiningOp());
  if (!lhsLoad || !rhsLoad) {
    return false;
  }
  if (lhsLoad.getMemref() != rhsLoad.getMemref()) {
    return false;
  }
  auto lhsIndices = lhsLoad.getIndices();
  auto rhsIndices = rhsLoad.getIndices();
  LLVM_DEBUG(llvm::dbgs() << "lhsIndices.size() = " << lhsIndices.size()
                          << ", rhsIndices.size() = " << rhsIndices.size()
                          << "\n");

  if (lhsIndices.size() != rhsIndices.size()) {
    return false;
  }

  Attribute lhsaffineCoeff = lhsLoad->getAttr("affineCoeff");
  Attribute lhsaffineOffset = lhsLoad->getAttr("affineOffset");
  Attribute lhsaffineMap = lhsLoad->getAttr("map");

  Attribute rhsaffineCoeff = rhsLoad->getAttr("affineCoeff");
  Attribute rhsaffineOffset = rhsLoad->getAttr("affineOffset");
  Attribute rhsaffineMap = rhsLoad->getAttr("map");

  // Both must have affine metadata to be considered consecutive.
  if (!lhsaffineCoeff || !rhsaffineCoeff)
    return false;

  if ((lhsaffineCoeff == rhsaffineCoeff) &&
      (lhsaffineOffset == rhsaffineOffset)) {
    return true;
  }

  // identity mapping
  if (lhsaffineMap && rhsaffineMap && lhsaffineMap == rhsaffineMap) {
    return true;
  }

  return false;
}

bool heteacc::consecutiveStores(Value lhs, Value rhs) {
  if (getAllStoreUsers(lhs).empty() || getAllStoreUsers(rhs).empty()) {
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "consecutiveStores: checking stores\n");
  auto lhsStore = getAllStoreUsers(lhs).front();
  auto rhsStore = getAllStoreUsers(rhs).front();
  if (!lhsStore || !rhsStore) {
    return false;
  }
  if (lhsStore.getMemref() != rhsStore.getMemref()) {
    return false;
  }
  auto lhsIndices = lhsStore.getIndices();
  auto rhsIndices = rhsStore.getIndices();
  LLVM_DEBUG(llvm::dbgs() << "lhsIndices.size() = " << lhsIndices.size()
                          << ", rhsIndices.size() = " << rhsIndices.size()
                          << "\n");

  if (lhsIndices.size() != rhsIndices.size()) {
    return false;
  }

  Attribute lhsaffineCoeff = lhsStore->getAttr("affineCoeff");
  Attribute lhsaffineOffset = lhsStore->getAttr("affineOffset");
  Attribute lhsaffineMap = lhsStore->getAttr("map");

  Attribute rhsaffineCoeff = rhsStore->getAttr("affineCoeff");
  Attribute rhsaffineOffset = rhsStore->getAttr("affineOffset");
  Attribute rhsaffineMap = rhsStore->getAttr("map");

  // Both must have affine metadata to be considered consecutive.
  if (!lhsaffineCoeff || !rhsaffineCoeff)
    return false;

  if ((lhsaffineCoeff == rhsaffineCoeff) &&
      (lhsaffineOffset == rhsaffineOffset)) {
    return true;
  }

  // identity mapping
  if (lhsaffineMap == rhsaffineMap) {
    return true;
  }

  return false;
}

/// Helper function to check if a value has a store operation as one of its
/// users
SmallVector<memref::StoreOp, 8> heteacc::getAllStoreUsers(Value value) {
  SmallVector<memref::StoreOp> storeUsers;
  for (auto *user : value.getUsers()) {
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Found store operation user for value.\n";
        storeOp.dump();
      });
      storeUsers.push_back(storeOp);
    }
  }
  if (storeUsers.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No store operation users found for value.\n");
  }
  return storeUsers;
}

SmallVector<Value, 2> heteacc::getOperands(Value value) {
  SmallVector<Value, 2> operands;
  assert(value.getDefiningOp() &&
         "operations without defining op do not have operands");
  operands.reserve(value.getDefiningOp()->getNumOperands());
  for (auto operand : value.getDefiningOp()->getOperands()) {
    operands.emplace_back(operand);
  }
  return operands;
}

SmallPtrSet<Operation *, 32> heteacc::computeDeadOps(Block *block) {
  LLVM_DEBUG(llvm::dbgs() << "computeDeadOps\n");
  SmallPtrSet<Operation *, 32> deadOps;
  llvm::SmallSetVector<Operation *, 32> worklist;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isOpTriviallyDead(op)) {
      LLVM_DEBUG(op->dump());
      worklist.insert(op);
      deadOps.insert(op);
    }
  });
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    for (auto const &operand : op->getOperands()) {
      if (auto *operandOp = operand.getDefiningOp()) {
        auto users = operandOp->getUsers();
        if (std::all_of(
                std::begin(users), std::end(users),
                [&](Operation *user) { return deadOps.contains(user); })) {
          worklist.insert(operandOp);
          deadOps.insert(operandOp);
        }
      }
    }
  }
  return deadOps;
}

/// Compute depths of all operations 'above' the seed. The seed is assumed to
/// have depth zero.
void heteacc::computeDepths(ArrayRef<Value> seed,
                            DenseMap<Value, unsigned> &depths) {
  llvm::SmallSetVector<Value, 32> worklist;
  for (auto value : seed) {
    depths[value] = 0;
    worklist.insert(value);
  }
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    if (auto *definingOp = value.getDefiningOp()) {
      for (auto operand : definingOp->getOperands()) {
        auto currentDepth = depths.lookup(operand);
        auto newDepth = depths[value] + 1;
        if (newDepth > currentDepth) {
          depths[operand] = newDepth;
          worklist.insert(operand);
        }
      }
    }
  }
}

void heteacc::sortByOpcode(SmallVectorImpl<Value> &values,
                           Optional<OperationName> smallestOpcode) {
  llvm::sort(std::begin(values), std::end(values), [&](Value lhs, Value rhs) {
    auto *lhsOp = lhs.getDefiningOp();
    auto *rhsOp = rhs.getDefiningOp();
    if (!lhsOp && !rhsOp) {
      return lhs.cast<BlockArgument>().getArgNumber() <
             rhs.cast<BlockArgument>().getArgNumber();
    } else if (lhsOp && !rhsOp) {
      return true;
    } else if (!lhsOp && rhsOp) {
      return false;
    }
    if (smallestOpcode.has_value()) {
      if (lhsOp->getName() == smallestOpcode.value()) {
        return rhsOp->getName() != smallestOpcode.value();
      } else if (rhsOp->getName() == smallestOpcode.value()) {
        return false;
      }
    }
    // Avoid tiebreaks.
    if (lhsOp->getName().getStringRef() == rhsOp->getName().getStringRef()) {
      return lhsOp->isBeforeInBlock(rhsOp);
    }
    return lhsOp->getName().getStringRef() < rhsOp->getName().getStringRef();
  });
}
