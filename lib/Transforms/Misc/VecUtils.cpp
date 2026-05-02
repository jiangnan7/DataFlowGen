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

#include "heteacc/Misc/VecUtils.h"
using namespace mlir;

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

  // Support linear access memory.
  if (op->hasTrait<mlir::OpTrait::ConstantLike>() || op->hasAttr("slp.group")) {
    if (dyn_cast<memref::LoadOp>(op) || dyn_cast<memref::StoreOp>(op)) {
      if (op->hasAttr("affineCoeff"))
        return true;
      return false;
    }
    return true;
  }

  // if (isa<arith::AddIOp, arith::AddFOp,
  //               arith::MulIOp, arith::MulFOp>(op)) {
  //   for (auto operand : op->getOperands()) {
  //     if (operand.getType().isa<mlir::IndexType>()) {
  //       return false;
  //     }
  //   }
  //   return true;
  // }

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
  llvm::outs() << "lhsIndices.size() = " << lhsIndices.size()
               << ", rhsIndices.size() = " << rhsIndices.size() << "\n";

  if (lhsIndices.size() != rhsIndices.size()) {
    return false;
  }

  Attribute lhsaffineCoeff = lhsLoad->getAttr("affineCoeff");
  Attribute lhsaffineOffset = lhsLoad->getAttr("affineOffset");
  Attribute lhsaffineMap = lhsLoad->getAttr("map");

  Attribute rhsaffineCoeff = rhsLoad->getAttr("affineCoeff");
  Attribute rhsaffineOffset = rhsLoad->getAttr("affineOffset");
  Attribute rhsaffineMap = rhsLoad->getAttr("map");

  if ((lhsaffineCoeff == rhsaffineCoeff) &&
      (lhsaffineOffset == rhsaffineOffset)) {
    return true;
  }

  // identity mapping
  if (lhsaffineMap == rhsaffineMap) {
    return true;
  }
  // TODO: Support multi-dimensional memory
  // if (auto lhsConstIndex =
  // dyn_cast<arith::ConstantIndexOp>(lhsIndices[0].getDefiningOp())) {
  //   if (auto rhsConstIndex =
  //   dyn_cast<arith::ConstantIndexOp>(rhsIndices[0].getDefiningOp())) {
  //       llvm::outs() << "lhsIndices[0] = " << lhsIndices[0] << ",
  //       rhsIndices[0] = " << rhsIndices[0] << "\n"; int lhsValue =
  //       lhsConstIndex.value(); int rhsValue = rhsConstIndex.value();

  //       llvm::outs() << "lhsConstIndex = " << lhsValue << ", rhsConstIndex =
  //       " << rhsValue << "\n";

  //     return (lhsValue + 1 == rhsValue) || (rhsValue + 1 == lhsValue);
  //   }
  // }

  return false;
}

bool heteacc::consecutiveStores(Value lhs, Value rhs) {
  if (getAllStoreUsers(lhs).empty() || getAllStoreUsers(rhs).empty()) {
    return false;
  }
  llvm::outs() << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
               << "\n";
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
  llvm::outs() << "lhsIndices.size() = " << lhsIndices.size()
               << ", rhsIndices.size() = " << rhsIndices.size() << "\n";

  if (lhsIndices.size() != rhsIndices.size()) {
    return false;
  }

  Attribute lhsaffineCoeff = lhsStore->getAttr("affineCoeff");
  Attribute lhsaffineOffset = lhsStore->getAttr("affineOffset");
  Attribute lhsaffineMap = lhsStore->getAttr("map");

  Attribute rhsaffineCoeff = rhsStore->getAttr("affineCoeff");
  Attribute rhsaffineOffset = rhsStore->getAttr("affineOffset");
  Attribute rhsaffineMap = rhsStore->getAttr("map");

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
      llvm::outs() << "Found store operation user for value.\n";
      storeOp.dump(); // Print details of the store operation
      storeUsers.push_back(storeOp);
    }
  }
  if (storeUsers.empty()) {
    llvm::outs() << "No store operation users found for value.\n";
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
  llvm::outs() << "computeDeadOps\n";
  SmallPtrSet<Operation *, 32> deadOps;
  llvm::SmallSetVector<Operation *, 32> worklist;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isOpTriviallyDead(op)) {
      op->dump();
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

/// Compute depths ot all operations 'above' the seed. The seed is assumed to
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
