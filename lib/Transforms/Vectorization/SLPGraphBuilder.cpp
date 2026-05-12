#include "mlir/IR/IntegerSet.h"

#include "llvm/Support/Debug.h"

#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Vectorization/SLPGraphBuilder.h"
#include "heteacc/Vectorization/Seeding.h"

using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

SLPGraphBuilder::SLPGraphBuilder(SLPGraph &graph, unsigned maxNodeSize,
                                 unsigned maxLookAhead,
                                 bool allowDuplicateElements)
    : graph{graph}, maxNodeSize{maxNodeSize}, allowDuplicateElements{
                                                  allowDuplicateElements} {}

void SLPGraphBuilder::build(ArrayRef<Value> seed) {
  graph.superwordRoot = std::make_shared<Superword>(seed);
  graph.nodeRoot = std::make_shared<SLPNode>(graph.superwordRoot);
  nodeBySuperword[graph.superwordRoot.get()] = graph.nodeRoot;

  LLVM_DEBUG(llvm::dbgs() << "graph.superwordRoot->getElement(0): "
                          << graph.superwordRoot->getElement(0) << "\n");
  superwordsByValue[graph.superwordRoot->getElement(0)].emplace_back(
      graph.superwordRoot);
  // If topological mixing is allowed anyways, we do not need to compute any
  // depths.
  computeDepths(seed, valueDepths);
  LLVM_DEBUG({
    llvm::dbgs() << "Printing ArrayRef<Value> seed:\n";
    for (const auto &value : seed) {
      llvm::dbgs() << "Value: ";
      if (auto *defOp = value.getDefiningOp()) {
        llvm::dbgs() << "Defining Operation: " << defOp->getName() << "\n";
        defOp->dump();
      } else {
        llvm::dbgs() << "No defining operation\n";
      }
    }
  });

  buildWorklist.insert(graph.nodeRoot.get());
  buildGraph(graph.superwordRoot);
}

// Some helper functions in an anonymous namespace.
namespace {

bool continueBuilding(Superword const &superword,
                      DenseMap<Value, unsigned> const &valueDepths,
                      bool allowDuplicateElements) {
  if (!vectorizable(superword.begin(), superword.end())) {
    LLVM_DEBUG(llvm::dbgs() << "!vectorizable\n");
    return false;
  }
  if (!allowDuplicateElements && !allLeaf(superword.begin(), superword.end())) {
    if (SmallPtrSet<Value, 4>{std::begin(superword), std::end(superword)}
            .size() < superword.numLanes()) {
      return false;
    }
  }
  if (!allLeaf(superword.begin(), superword.end())) {
    for (size_t lane = 1; lane < superword.numLanes(); ++lane) {
      if (valueDepths.lookup(superword.getElement(lane)) !=
          valueDepths.lookup(superword.getElement(0))) {
        return false;
      }
    }
  }
  return true;
}

bool appendable(SLPNode const &node, OperationName const &opCode,
                ArrayRef<SmallVector<Value, 2>> allOperands,
                unsigned operandIndex, unsigned maxNodeSize) {
  // Check size.
  if (node.numSuperwords() == maxNodeSize) {
    return false;
  }
  // Check opcode.
  return std::all_of(
      std::begin(allOperands), std::end(allOperands),
      [&](auto const &operands) {
        auto const &operand = operands[operandIndex];
        if (operand.getDefiningOp() == nullptr)
          return false;
        if (operand.getDefiningOp()->getName() != opCode) {
          return false;
        }
        // Check if there are escaping users (reordering can semantically kill
        // them).
        return std::all_of(
            std::begin(operand.getUsers()), std::end(operand.getUsers()),
            [&](auto *user) { return node.contains(user->getResult(0)); });
      });
}

SmallVector<SmallVector<Value, 2>>
getAllOperandsSorted(Superword const &superword,
                     OperationName const &currentOpCode) {
  SmallVector<SmallVector<Value, 2>> allOperands;
  allOperands.reserve(superword.numLanes());
  for (auto value : superword) {
    allOperands.emplace_back(getOperands(value));
  }
  for (auto &operands : allOperands) {
    sortByOpcode(operands, currentOpCode);
  }
  return allOperands;
}

/// Models the semantics of a superword by comparing vector elements' operation
/// chains to the operation chains of the SLP graph's lanes. If they differ, the
/// semantics of the individual vector elements have changed.
struct SuperwordSemantics {
  SuperwordSemantics() = default;
  SuperwordSemantics(
      Superword *superword,
      DenseMap<Superword *, SuperwordSemantics> const &parentSemantics)
      : operandDifference{superword->numLanes()} {
    // Go through each lane and compare the element's actual computation chain
    // to the one of the SLP graph. This can be done by simply counting how
    // often every value appears in each of the computation chains.
    for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
      auto op = superword->getElement(lane).getDefiningOp();
      for (unsigned i = 0; i < superword->numOperands(); ++i) {
        auto elementOperand = op->getOperand(i);
        auto operandElement = superword->getOperand(i)->getElement(lane);
        if (elementOperand == operandElement) {
          continue;
        }
        if (operandDifference[lane][elementOperand] == 1) {
          // Keep the map as small as possible by erasing entries if there are
          // no differences.
          operandDifference[lane].erase(elementOperand);
        } else {
          --operandDifference[lane][elementOperand];
        }
        if (operandDifference[lane][operandElement] == -1) {
          // Keep the map as small as possible by erasing entries if there are
          // no differences.
          operandDifference[lane].erase(operandElement);
        } else {
          ++operandDifference[lane][operandElement];
        }
      }
    }
    // Aggregate the semantics of the current SLP vector with the ones of its
    // operand SLP vectors.
    for (unsigned lane = 0; lane < superword->numLanes(); ++lane) {
      for (auto *operandWord : superword->getOperands()) {
        auto const &semantics = parentSemantics.lookup(operandWord);
        if (semantics.operandDifference.empty()) {
          continue;
        }
        for (auto const &differenceEntry : semantics.operandDifference[lane]) {
          auto newDifference =
              operandDifference[lane].lookup(differenceEntry.first) +
              differenceEntry.second;
          if (newDifference == 0) {
            operandDifference[lane].erase(differenceEntry.first);
          } else {
            operandDifference[lane][differenceEntry.first] = newDifference;
          }
        }
      }
    }
  }

  /// Returns true if the superword's lane has altered semantics due to
  /// reordering, otherwise false.
  bool areSemanticsAlteredInLane(size_t lane) const {
    return !operandDifference[lane].empty();
  }

  // Models computation chain differences for each lane.
  // positive: surplus of that value in the SLP graph's computation chain.
  // negative: deficiency of that value in the SLP graph's computation chain.
  // no entry/zero: no difference
  SmallVector<DenseMap<Value, int>, 4> operandDifference;
};

} // end namespace

void SLPGraphBuilder::buildGraph(std::shared_ptr<Superword> const &superword) {
  // Stop growing graph
  if (!continueBuilding(*superword, valueDepths, allowDuplicateElements)) {
    LLVM_DEBUG(llvm::dbgs() << " !continueBuilding \n");
    return;
  }
  auto currentNode = nodeBySuperword[superword.get()];
  auto const &currentOpCode = superword->begin()->getDefiningOp()->getName();
  auto const &arity = superword->begin()->getDefiningOp()->getNumOperands();
  // Recursion call to grow graph further
  // 1. Commutative
  if (commutative(superword->begin(), superword->end())) {
    LLVM_DEBUG(llvm::dbgs() << "commutative \n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "no commutative \n");
  }
  if (commutative(superword->begin(), superword->end())) {
    // A. Coarsening Mode
    auto allOperands = getAllOperandsSorted(*superword, currentOpCode);
    for (unsigned i = 0; i < arity; ++i) {
      SmallVector<Value, 4> superwordValues;
      for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
        superwordValues.emplace_back(allOperands[lane][i]);
      }
      LLVM_DEBUG({
        llvm::dbgs() << " superwordValues \n";
        for (auto it = superwordValues.begin(); it != superwordValues.end();
             ++it) {
          llvm::dbgs() << "Value: ";
          if (auto *defOp = it->getDefiningOp()) {
            llvm::dbgs() << "Defining Operation: " << defOp->getName() << "\n";
            defOp->dump();
          } else {
            llvm::dbgs() << "No defining operation\n";
          }
        }
      });

      if (ofVectorizableType(std::begin(superwordValues),
                             std::end(superwordValues))) {
        //不断添加和它相关的操作。mul -> 常数 -> load操作。
        LLVM_DEBUG(llvm::dbgs() << "ofVectorizableType\n");
        auto operandNode =
            addOperandToNode(superwordValues, currentNode, superword);
        buildWorklist.insert(operandNode.get());
      }
    }
    // B. Normal Mode: Finished building multi-node
    if (currentNode->isSuperwordRoot(*superword)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\ncurrentNode->isSuperwordRoot(*superword)\n");
      // NOTE: reorderOperands disabled - shuffling superwords is unsafe when
      // operands have been reordered by more than a simple commutative swap.
      // The semantics of the node's root cannot be changed as it accumulates
      // everything regardless of order.
      if (currentNode->numSuperwords() > 1) {
        DenseMap<Superword *, SuperwordSemantics> semantics;
        for (unsigned i = currentNode->numSuperwords(); i-- > 0;) {
          auto *nodeWord = currentNode->getSuperword(i).get();
          semantics.try_emplace(nodeWord, nodeWord, semantics);
        }
        for (unsigned i = 0; i < currentNode->numSuperwords(); ++i) {
          for (unsigned lane = 0; lane < currentNode->numLanes(); ++lane) {
            if (semantics.lookup(currentNode->getSuperword(i).get())
                    .areSemanticsAlteredInLane(lane)) {
              currentNode->getSuperword(i)->markSemanticsAlteredInLane(lane);
            }
          }
        }
      }
      for (auto const &operandNode : currentNode->getOperands()) {
        LLVM_DEBUG(llvm::dbgs() << "buildWorklist\n");
        if (buildWorklist.erase(operandNode.get())) {
          buildGraph(
              operandNode->getSuperword(operandNode->numSuperwords() - 1));
        }
      }
    }
  }
  // 2. Non-Commutative
  else {

    for (size_t i = 0; i < arity; ++i) {
      LLVM_DEBUG(llvm::dbgs() << "Non-Commutative size\n");
      SmallVector<Value, 4> operandValues;
      for (size_t lane = 0; lane < currentNode->numLanes(); ++lane) {
        auto operand =
            currentNode->getValue(lane, 0).getDefiningOp()->getOperand(i);
        LLVM_DEBUG(llvm::dbgs() << "operand: " << operand << "\n");
        operandValues.emplace_back(operand);
      }
      if (ofVectorizableType(std::begin(operandValues),
                             std::end(operandValues))) {
        auto operandNode =
            addOperandToNode(operandValues, currentNode, superword);
        buildWorklist.insert(operandNode.get());
        buildGraph(operandNode->getSuperword(0));
      }
    }
  }
}

void SLPGraphBuilder::reorderOperands(SLPNode *multinode) const {
  auto const &numOperands = multinode->numOperands();
  assert(numOperands > 0 && "trying to reorder a node with zero operands");
  SmallVector<SmallVector<Value, 4>> finalOrder{multinode->numLanes()};
  SmallVector<SmallVector<Mode, 4>> mode{multinode->numLanes()};
  // 1. Strip first lane
  for (size_t i = 0; i < numOperands; ++i) {
    auto value = multinode->getOperand(i)->getValue(0, 0);
    finalOrder[0].emplace_back(value);
    mode[0].emplace_back(modeFromValue(value));
  }
  // 2. For all other lanes, find best candidate
  for (size_t lane = 1; lane < multinode->numLanes(); ++lane) {
    SmallVector<Value> candidates;
    for (auto const &operand : multinode->getOperands()) {
      candidates.emplace_back(operand->getValue(lane, 0));
    }
    // Look for a matching candidate
    for (size_t i = 0; i < numOperands; ++i) {
      // Skip if we can't vectorize
      if (mode[lane - 1][i] == Mode::FAILED) {
        finalOrder[lane].emplace_back(nullptr);
        mode[lane].emplace_back(Mode::FAILED);
        continue;
      }
      auto last = finalOrder[lane - 1][i];
      auto const &bestResult = getBest(mode[lane - 1][i], last, candidates);
      // Update output
      finalOrder[lane].emplace_back(bestResult.first);
      // Detect 'SPLAT' mode
      if (i == 1 && bestResult.first == last) {
        mode[lane].emplace_back(Mode::SPLAT);
      } else {
        mode[lane].emplace_back(bestResult.second);
      }
    }
    // Distribute remaining candidates in case we encountered a 'FAILED'.
    for (auto candidate : candidates) {
      for (size_t i = 0; i < numOperands; ++i) {
        if (finalOrder[lane][i] == nullptr) {
          finalOrder[lane][i] = candidate;
          break;
        }
      }
    }
  }
  for (size_t operandIndex = 0; operandIndex < multinode->numOperands();
       ++operandIndex) {
    for (size_t lane = 0; lane < multinode->numLanes(); ++lane) {
      if (multinode->getOperand(operandIndex)->getValue(lane, 0) !=
          finalOrder[lane][operandIndex]) {
        multinode->getOperand(operandIndex)
            ->setValue(lane, 0, finalOrder[lane][operandIndex]);
      }
    }
  }
}

std::pair<Value, SLPGraphBuilder::Mode>
SLPGraphBuilder::getBest(Mode mode, Value last,
                         SmallVector<Value> &candidates) const {
  Value best;
  Mode resultMode = mode;
  SmallVector<Value> bestCandidates;
  if (mode == Mode::FAILED) {
    // Don't select now, let others choose first
    best = nullptr;
  } else if (mode == Mode::SPLAT) {
    // Look for other 'SPLAT' candidates
    for (auto &operand : candidates) {
      if (operand == last) {
        best = operand;
        break;
      }
    }
  } else {
    // Default value
    best = candidates.front();
    for (auto &candidate : candidates) {
      if (mode == Mode::LOAD) {
        if (consecutiveLoads(last, candidate)) {
          bestCandidates.emplace_back(candidate);
        }
      } else if (last.getDefiningOp() && candidate.getDefiningOp()) {
        if (last.getDefiningOp()->getName() ==
            candidate.getDefiningOp()->getName()) {
          bestCandidates.emplace_back(candidate);
        }
      }
    }
    // 1. If we have a trivial solution, use it
    // No matches
    if (bestCandidates.empty()) {
      resultMode = Mode::FAILED;
    }
    // Single match
    else if (bestCandidates.size() == 1) {
      best = bestCandidates.front();
    }
    // 2. Look-ahead to choose from best candidates
    else if (mode == Mode::OPCODE) {
      best = scoreModel->getBest(last, bestCandidates);
    }
  }
  // Remove best from candidates
  if (best != nullptr) {
    candidates.erase(
        std::find(std::begin(candidates), std::end(candidates), best));
  }
  return {best, resultMode};
}

// === Utilities === //

SLPGraphBuilder::Mode SLPGraphBuilder::modeFromValue(Value value) {
  if (auto *definingOp = value.getDefiningOp()) {
    if (definingOp->hasTrait<OpTrait::ConstantLike>()) {
      return Mode::CONST;
    } else if (dyn_cast<memref::LoadOp>(definingOp)) {
      return Mode::LOAD;
    }
    return Mode::OPCODE;
  }
  return Mode::SPLAT;
}

std::shared_ptr<Superword> SLPGraphBuilder::appendSuperwordToNode(
    ArrayRef<Value> values, std::shared_ptr<SLPNode> const &node,
    std::shared_ptr<Superword> const &usingSuperword) {
  auto superword = std::make_shared<Superword>(values);
  superwordsByValue[values[0]].emplace_back(superword);
  nodeBySuperword[superword.get()] = node;
  node->addSuperword(superword);
  usingSuperword->addOperand(superword);
  return superword;
}

std::shared_ptr<SLPNode> SLPGraphBuilder::addOperandToNode(
    ArrayRef<Value> operandValues, std::shared_ptr<SLPNode> const &node,
    std::shared_ptr<Superword> const &usingSuperword) {
  auto superword = std::make_shared<Superword>(operandValues);
  superwordsByValue[operandValues[0]].emplace_back(superword);
  auto operandNode =
      nodeBySuperword
          .try_emplace(superword.get(), std::make_shared<SLPNode>(superword))
          .first->second;
  nodeBySuperword[superword.get()] = operandNode;
  node->addOperand(operandNode);
  usingSuperword->addOperand(superword);
  return operandNode;
}
