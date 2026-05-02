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

  llvm::outs() << this->availableOps.size() << " dd\n";
  this->rootOp->dump();
  auto seed = nextSeed();
  llvm::outs() << "\nnextSeed(): " << seed.size() << " \n";
  for (const auto op : seed) {
    llvm::outs() << op;
  }
  llvm::outs() << "cccccddddddd\n";
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
      LLVM_DEBUG({
        llvm::dbgs() << "Not vectorizable.\n";
        op->dump();
      });
      return;
    }
    this->availableOps.insert(op);
  });
}

SmallVector<Value, 4> TopDownAnalysis::nextSeed() const {

  DenseMap<int64_t, SmallVector<Operation *, 4>> groupMap;
  for (auto *op : this->availableOps) {
    if (op->hasAttr("slp.group")) {
      int64_t group = op->getAttr("slp.group").cast<IntegerAttr>().getInt();
      groupMap[group].push_back(op);
    }
  }

  SmallVector<SmallVector<Value, 4>> seeds;

  for (auto [groupId, operations] : groupMap) {

    SmallVector<Value, 4> groupSeed;
    for (auto *op : operations) {
      if (!availableOps.contains(op) || !vectorizable(op)) {
        continue;
      }
      for (auto result : op->getResults()) {
        groupSeed.push_back(result);
      }
    }

    seeds.push_back(groupSeed);
  }
  llvm::sort(std::begin(seeds), std::end(seeds), [&](auto &lhs, auto &rhs) {
    auto lhsGroup = lhs.front()
                        .getDefiningOp()
                        ->getAttr("slp.group")
                        .template cast<IntegerAttr>()
                        .getInt();
    auto rhsGroup = rhs.front()
                        .getDefiningOp()
                        ->getAttr("slp.group")
                        .template cast<IntegerAttr>()
                        .getInt();
    return lhsGroup < rhsGroup;
  });

  if (seeds.empty()) {
    return {};
  }

  auto &seed = seeds.front();
  assert(&seed);
  return seed;
}
