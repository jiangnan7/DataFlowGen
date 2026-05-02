#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "heteacc/Misc/Utils.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Transforms/Passes.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"

using namespace mlir;
using namespace heteacc;

// === PatternVisitor === //

void PatternVisitor::visit(BroadcastSuperword const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeReduction const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeConstant const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(CreateConsecutiveLoad const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(CreateGatherLoad const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(CreateConsecutiveStore const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeAddI const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeAddF const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeMulI const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

void PatternVisitor::visit(VectorizeMulF const *pattern,
                           Superword const *superword) {
  visitDefault(pattern, superword);
}

// === LeafPatternVisitor === //

ArrayRef<Value> LeafPatternVisitor::getRequiredScalarValues(
    SLPVectorizationPattern const *pattern, Superword const *superword) {
  pattern->accept(*this, superword);
  return this->scalarValues;
}

void LeafPatternVisitor::visitDefault(SLPVectorizationPattern const *pattern,
                                      Superword const *superword) {
  this->scalarValues.clear();
}

void LeafPatternVisitor::visit(BroadcastSuperword const *pattern,
                               Superword const *superword) {
  this->scalarValues.assign({superword->getElement(0)});
}

double CostModel::getScalarCost(Value value) {
  if (conversionState->alreadyComputed(value)) {
    return 0;
  }
  auto it = cachedScalarCost.try_emplace(value, computeScalarCost(value));
  if (it.second) {
    if (auto *definingOp = value.getDefiningOp()) {
      for (auto operand : definingOp->getOperands()) {
        if (isExtractionProfitable(operand)) {
          cachedScalarCost[value] += getExtractionCost(operand);
        } else {
          cachedScalarCost[value] += getScalarCost(operand);
        }
      }
    }
    return cachedScalarCost[value];
  }
  return it.first->second;
}

double CostModel::getSuperwordCost(Superword *superword,
                                   SLPVectorizationPattern *pattern) {
  if (conversionState->alreadyComputed(superword)) {
    return 0;
  }
  pattern->accept(*this, superword);
  double vectorCost = cost;
  for (auto *operand : superword->getOperands()) {
    if (!conversionState->alreadyComputed(operand)) {
      //如果不带slp.group，或者是isa blockargument ，就加代价
      // auto* operandPattern = patternApplicator.bestMatch(operand);

      for (auto it = operand->begin(); it != operand->end(); ++it) {
        llvm::outs() << "Value: ";
        if (auto *defOp = it->getDefiningOp()) {
          llvm::outs() << "Defining Operation: " << defOp->getName() << "\n";
          defOp->dump();
          ;
        } else {
          llvm::outs() << "No defining operation\n";
        }
      }
      // operandPattern->accept(*this, operand);
      vectorCost += cost;
    }
  }
  return vectorCost;
}

bool CostModel::isExtractionProfitable(Value value) {
  auto extractionCost = getExtractionCost(value);
  if (extractionCost == MAX_COST) {
    return false;
  }
  auto scalarCost = getScalarCost(value);
  return extractionCost < scalarCost;
}

void CostModel::setConversionState(
    std::shared_ptr<ConversionState> newConversionState) {
  conversionState = std::move(newConversionState);
  conversionState->addScalarCallbacks(
      [&](Value value) { updateCost(value, 0, false); },
      [&](Value value) { cachedScalarCost.erase(value); });
  conversionState->addExtractionCallbacks(
      [&](Value value) { updateCost(value, 0, true); },
      [&](Value value) { cachedScalarCost.erase(value); });
}

double
CostModel::getBlockCost(Block *block,
                        SmallPtrSetImpl<Operation *> const &deadOps) const {
  double blockCost = 0;
  block->walk([&](Operation *op) {
    if (deadOps.contains(op)) {
      llvm::outs() << "don't need\n";
      op->dump();
      return WalkResult::skip();
    }
    for (auto const &result : op->getResults()) {
      blockCost += computeScalarCost(result);
      break;
    }
    return WalkResult::advance();
  });
  return blockCost;
}

void CostModel::updateCost(Value value, double newCost, bool updateUses) {
  auto it = cachedScalarCost.try_emplace(value, newCost);
  if (!it.second && it.first->second <= newCost) {
    return;
  }
  if (updateUses) {
    for (auto *user : value.getUsers()) {
      for (auto const &result : user->getResults()) {
        if (cachedScalarCost.count(result)) {
          updateCost(result, getScalarCost(result), true);
        }
      }
    }
  }
}

double CostModel::getExtractionCost(Value value) const {
  if (conversionState->isExtractable(value)) {
    auto valuePosition = conversionState->getSuperwordContainingValue(value);
    return 1;
  }
  return MAX_COST;
}

double CostModel::computeScalarCost(Value value) const {
  // Individual costs taken from NodePatterns.cpp
  auto *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return 0;
  }
  if (dyn_cast<arith::ConstantOp>(definingOp)) {
    return 0;
  }
  if (value.getType().isa<mlir::VectorType>())
    return 0;
  // TODO: other type.
  // Assume a default cost of 1, i.e. that there roughly is a 1:1 mapping of
  // operations to actual assembly instructions.
  return 1;
}

void CostModel::visitDefault(SLPVectorizationPattern const *pattern,
                             Superword const *superword) {
  this->cost = 1;
}

void CostModel::visit(BroadcastSuperword const *pattern,
                      Superword const *superword) {
  this->cost = 1;
  for (auto element : leafVisitor.getRequiredScalarValues(pattern, superword)) {
    this->cost += getScalarCost(element);
  }
}

void CostModel::visit(VectorizeConstant const *pattern,
                      Superword const *superword) {
  this->cost = 0;
}
