#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "heteacc/Misc/Utils.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Vectorization/Seeding.h"
using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

SmallVector<Value, 4> SeedAnalysis::next() {
  if (this->availableOps.empty()) {
    if (!this->availableComputed) {
      computeAvailableOps();
      this->availableComputed = true;
    } else {
      return {};
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Available ops for seeding: "
                          << this->availableOps.size() << "\n");

  auto seed = nextSeed();
  LLVM_DEBUG(llvm::dbgs() << "Next seed size: " << seed.size() << "\n");

  for (auto value : seed) {
    if (auto *definingOp = value.getDefiningOp()) {
      this->availableOps.remove(definingOp);
    }
  }
  return seed;
}

void SeedAnalysis::update(ArrayRef<Superword *> convertedSuperwords) {
  SmallPtrSet<Operation *, 32> convertedOps;
  for (auto *superword : convertedSuperwords) {
    for (auto element : *superword) {
      if (auto *definingOp = element.getDefiningOp()) {
        convertedOps.insert(definingOp);
      }
    }
  }
  // Construct anew and swap because erasing many ops one after one takes a lot
  // of time in set vectors.
  llvm::SmallSetVector<Operation *, 32> newAvailableOps;
  for (auto *op : availableOps) {
    if (!convertedOps.contains(op)) {
      newAvailableOps.insert(op);
    }
  }
  availableOps.swap(newAvailableOps);
}

TopDownAnalysis::TopDownAnalysis(Operation *rootOp, unsigned width)
    : SeedAnalysis{rootOp, width} {}

void TopDownAnalysis::computeAvailableOps() {
  this->rootOp->walk([&](Operation *op) {
    if (!vectorizable(op) || op->hasTrait<OpTrait::ConstantLike>()) {
      return;
    }
    this->availableOps.insert(op);
  });
  LLVM_DEBUG(llvm::dbgs() << "Computed " << this->availableOps.size()
                          << " available ops for vectorization\n");
}

SmallVector<Value, 4> TopDownAnalysis::nextSeed() const {
  // Strategy 1: Use slp.group attributes if available.
  DenseMap<int64_t, SmallVector<Operation *, 4>> groupMap;
  for (auto *op : this->availableOps) {
    if (op->hasAttr("slp.group")) {
      int64_t group = op->getAttr("slp.group").cast<IntegerAttr>().getInt();
      groupMap[group].push_back(op);
    }
  }

  if (!groupMap.empty()) {
    // Find the first group (by group ID) that has enough elements.
    SmallVector<std::pair<int64_t, SmallVector<Operation *, 4>>> sortedGroups(
        groupMap.begin(), groupMap.end());
    llvm::sort(sortedGroups, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });

    for (auto &[groupId, operations] : sortedGroups) {
      SmallVector<Value, 4> groupSeed;
      for (auto *op : operations) {
        if (!availableOps.contains(op) || !vectorizable(op))
          continue;
        for (auto result : op->getResults()) {
          groupSeed.push_back(result);
        }
      }
      if (groupSeed.size() >= 2) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Found seed from slp.group=" << groupId << " with "
                   << groupSeed.size() << " elements\n");
        return groupSeed;
      }
    }
  }

  // No slp.group found - no vectorization opportunities.
  return {};
}
