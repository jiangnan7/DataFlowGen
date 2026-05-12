#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "heteacc/Misc/Utils.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Transforms/Passes.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

// === SLPVectorizationPattern === //
SLPVectorizationPattern::SLPVectorizationPattern(
    ConversionManager &conversionManager)
    : conversionManager{conversionManager} {}

void SLPVectorizationPattern::rewriteSuperword(Superword *superword,
                                               RewriterBase &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "setupConversionFor\n");
  conversionManager.setupConversionFor(superword, this);
  LLVM_DEBUG(llvm::dbgs() << "rewrite\n");

  LLVM_DEBUG({
    for (auto it = superword->begin(); it != superword->end(); ++it) {
      llvm::dbgs() << "Value: ";
      if (auto *defOp = it->getDefiningOp()) {
        llvm::dbgs() << "Defining Operation: " << defOp->getName() << "\n";
        defOp->dump();
      } else {
        llvm::dbgs() << "No defining operation\n";
      }
    }
  });

  auto vectorOp = this->rewrite(superword, rewriter);

  auto consecutiveStorePattern =
      std::make_unique<CreateConsecutiveStore>(this->conversionManager);
  if (succeeded(consecutiveStorePattern->match(superword))) {
    consecutiveStorePattern->rewrite(superword, vectorOp, rewriter);
  } else {
    auto indirectStorePattern =
        std::make_unique<VectorizeIndirectStore>(this->conversionManager);
    if (succeeded(indirectStorePattern->match(superword))) {
      indirectStorePattern->rewrite(superword, vectorOp, rewriter);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "update\n");
  conversionManager.update(superword, vectorOp, this);
}

// Helper functions in anonymous namespace.
namespace {

/// Get the value that is reduced by the `pos`-th iter arg in a dataflow loop.
/// Mirrors MLIR's affine reduction detection, but for `dataflow::ForOp`.
static Value getSupportedReduction(dataflow::ForOp forOp, unsigned pos,
                                   arith::AtomicRMWKind &kind) {
  SmallVector<Operation *> combinerOps;
  Value reducedVal = matchReduction(forOp.getRegionIterArgs(), pos, combinerOps);
  if (!reducedVal)
    return nullptr;

  if (combinerOps.size() != 1)
    return nullptr;

  Operation *combinerOp = combinerOps.back();
  std::optional<arith::AtomicRMWKind> maybeKind =
      llvm::TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(
          combinerOp)
          .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
          .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
          .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
          .Case([](arith::AndIOp) { return arith::AtomicRMWKind::andi; })
          .Case([](arith::OrIOp) { return arith::AtomicRMWKind::ori; })
          .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
          .Case([](arith::MinFOp) { return arith::AtomicRMWKind::minf; })
          .Case([](arith::MaxFOp) { return arith::AtomicRMWKind::maxf; })
          .Case([](arith::MinSIOp) { return arith::AtomicRMWKind::mins; })
          .Case([](arith::MaxSIOp) { return arith::AtomicRMWKind::maxs; })
          .Case([](arith::MinUIOp) { return arith::AtomicRMWKind::minu; })
          .Case([](arith::MaxUIOp) { return arith::AtomicRMWKind::maxu; })
          .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> {
            return std::nullopt;
          });
  if (!maybeKind)
    return nullptr;

  kind = *maybeKind;
  return reducedVal;
}

/// Identity function (log-space stripping removed, kept for interface compat).
Value stripLogOrValue(Value value, RewriterBase &rewriter) { return value; }

static Attribute getZeroAttrForType(Type type, Builder &builder) {
  if (auto floatType = type.dyn_cast<FloatType>())
    return builder.getFloatAttr(floatType, 0.0);
  if (auto intType = type.dyn_cast<IntegerType>())
    return builder.getIntegerAttr(intType, 0);
  if (type.isa<IndexType>())
    return builder.getIndexAttr(0);
  llvm_unreachable("unsupported zero attribute type");
}

static Value createIndexOffset(Value baseIndex, Value laneIndex, int64_t coeff,
                               Location loc, RewriterBase &rewriter) {
  if (baseIndex == laneIndex || coeff == 0)
    return rewriter.create<arith::ConstantIndexOp>(loc, 0);

  Value delta = rewriter.create<arith::SubIOp>(loc, laneIndex, baseIndex);
  if (coeff == 1)
    return delta;

  Value coeffVal = rewriter.create<arith::ConstantIndexOp>(loc, coeff);
  return rewriter.create<arith::MulIOp>(loc, delta, coeffVal);
}

static Value packIndexVector(Location loc, ArrayRef<Value> offsets,
                             RewriterBase &rewriter) {
  auto vectorType = VectorType::get({static_cast<int64_t>(offsets.size())},
                                    rewriter.getIndexType());
  auto zeroElements =
      DenseElementsAttr::get(vectorType, rewriter.getIndexAttr(0));
  Value packed = rewriter.create<arith::ConstantOp>(loc, vectorType,
                                                    zeroElements);
  for (auto [lane, offset] : llvm::enumerate(offsets)) {
    Value lanePos =
        rewriter.create<arith::ConstantIntOp>(loc, lane, 32);
    packed = rewriter.create<vector::InsertElementOp>(loc, offset, packed,
                                                      lanePos);
  }
  return packed;
}

/// Helper: find supported reductions for a dataflow::ForOp.
static SmallVector<heteacc::LoopReduction, 2>
findSupportedReductions(dataflow::ForOp forOp) {
  SmallVector<heteacc::LoopReduction, 2> reductions;
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    arith::AtomicRMWKind kind;
    if (Value value = getSupportedReduction(forOp, i, kind))
      reductions.emplace_back(heteacc::LoopReduction{kind, i, value});
  }
  return reductions;
}
} // namespace

// === Broadcast === //

LogicalResult BroadcastSuperword::match(Superword *superword) {
  Operation *firstOp = nullptr;
  if (superword->numLanes() <= 1)
    return failure();
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    if (superword->hasAlteredSemanticsInLane(lane)) {
      return failure();
    }
    if (auto *definingOp = superword->getElement(lane).getDefiningOp()) {
      if (lane == 0) {
        firstOp = definingOp;
        continue;
      }
      if (definingOp == nullptr)
        return failure();
      LLVM_DEBUG(definingOp->dump());
      if (firstOp == nullptr)
        return failure();
      LLVM_DEBUG(firstOp->dump());

      if (!OperationEquivalence::isEquivalentTo(
              definingOp, firstOp, OperationEquivalence::Flags::None)) {
        return failure();
      }
    } else if (firstOp ||
               superword->getElement(lane) != superword->getElement(0)) {
      return failure();
    }
  }
  return success();
}

Value BroadcastSuperword::rewrite(Superword *superword,
                                  RewriterBase &rewriter) {
  auto element = stripLogOrValue(superword->getElement(0), rewriter);
  return rewriter.create<vector::BroadcastOp>(
      element.getLoc(), superword->getVectorType(), element);
}

void BroadcastSuperword::accept(PatternVisitor &visitor,
                                Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeReduction === //

LogicalResult VectorizeReduction::match(Superword *superword) {
  LLVM_DEBUG(llvm::dbgs() << "VectorizeReduction\n");
  Operation *parentOp = nullptr;
  for (auto it = superword->begin(); it != superword->end(); ++it) {
    if (auto *defOp = it->getDefiningOp()) {
      parentOp = defOp->getParentOp();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No defining operation\n");
    }
  }
  if (!parentOp)
    return failure();
  LLVM_DEBUG(parentOp->dump());
  auto forOp = dyn_cast<dataflow::ForOp>(parentOp);
  if (!forOp || (forOp.getNumRegionIterArgs() == 0))
    return failure();

  auto *lastDefOp = superword->getElement(superword->numLanes() - 1)
                        .getDefiningOp();
  if (!lastDefOp || lastDefOp->getUsers().empty())
    return failure();
  Operation *terminatorOp = *lastDefOp->getUsers().begin();
  if (terminatorOp == nullptr ||
      !terminatorOp->mightHaveTrait<OpTrait::IsTerminator>()) {
    return failure();
  }

  auto supportedReductions = findSupportedReductions(forOp);
  if (supportedReductions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "supportedReductions empty\n");
    return failure();
  }

  LLVM_DEBUG({
    superword->getElement(0).getDefiningOp()->dump();
    if (superword->numLanes() > 1)
      superword->getElement(superword->numLanes() - 1).getDefiningOp()->dump();
  });
  return success();
}

static bool isNeutralElementConst(arith::AtomicRMWKind reductionKind,
                                  Value value, OpBuilder &builder) {
  Type scalarTy = value.getType();
  if (!VectorType::isValidElementType(scalarTy))
    return false;
  Attribute valueAttr =
      getIdentityValueAttr(reductionKind, scalarTy, builder, value.getLoc());
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp()))
    return constOp.getValue() == valueAttr;
  return false;
}

Value VectorizeReduction::rewrite(Superword *superword,
                                  RewriterBase &rewriter) {

  auto *defOp = superword->getElement(0).getDefiningOp();
  if (!defOp)
    return Value{};
  Operation *parentOp = defOp->getParentOp();
  auto forOp = dyn_cast<dataflow::ForOp>(parentOp);

  auto supportedReductions = findSupportedReductions(forOp);
  OpBuilder builder(forOp);
  auto createInitialVector = [&](arith::AtomicRMWKind reductionKind,
                                 Value oldOperand) -> arith::ConstantOp {
    Type scalarTy = oldOperand.getType();
    if (!VectorType::isValidElementType(scalarTy))
      return nullptr;

    Attribute valueAttr = getIdentityValueAttr(reductionKind, scalarTy, builder,
                                               oldOperand.getLoc());
    auto vecAttr =
        DenseElementsAttr::get(superword->getVectorType(), valueAttr);
    auto newConstOp =
        builder.create<arith::ConstantOp>(oldOperand.getLoc(), vecAttr);

    return newConstOp;
  };

  // Vectorize 'iter_args'.
  SmallVector<Value, 8> vecIterOperands;

  for (auto redAndOperand :
       llvm::zip(supportedReductions, forOp.getIterOperands())) {
    vecIterOperands.push_back(createInitialVector(
        std::get<0>(redAndOperand).kind, std::get<1>(redAndOperand)));
  }

  auto vecForOp = builder.create<dataflow::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), vecIterOperands,
      /*bodyBuilder=*/[](OpBuilder &, Location, Value, ValueRange) {
        // Make sure we don't create a default terminator in the loop body as
        // the proper terminator will be added during vectorization.
      });

  LLVM_DEBUG(vecForOp->getParentOp()->dump());
  // Replace all operations in the forOp body with corresponding operations in
  // vecForOp body.
  Block &forOpBody = forOp.getRegion().front();
  Block &vecForOpBody = vecForOp.getRegion().front();
  vecForOp.getRegion().takeBody(forOp.getRegion());
  for (auto &op : forOpBody.getOperations()) {

    builder.setInsertionPointToEnd(&vecForOpBody);

    builder.clone(op);
  }

  for (auto resultTuple :
       llvm::zip(forOp->getResults(), vecForOp->getResults())) {
    Value replacedResult = std::get<0>(resultTuple);
    Value replacementResult = std::get<1>(resultTuple);

    // Replace all uses of the replaced result with the replacement result
    replacedResult.replaceAllUsesWith(replacementResult);
  }

  LLVM_DEBUG(vecForOp->dump());

  OpBuilder vecbuilder(vecForOp);
  for (unsigned i = 0; i < vecForOp.getNumRegionIterArgs(); ++i) {
    // First, we reduce the vector returned from the loop into a scalar.
    Value reducedRes =
        vector::getVectorReductionOp(supportedReductions[i].kind, vecbuilder,

                                     vecForOp.getLoc(), vecForOp.getResult(i));
    LLVM_DEBUG({
      llvm::dbgs() << "\n[early-vect]+++++ creating a vector reduction: "
                   << reducedRes;
    });
    // Then we combine it with the original (scalar) initial value unless it
    // is equal to the neutral element of the reduction.
    Value origInit = forOp.getOperand(forOp.getNumControlOperands() + i);
    Value finalRes = reducedRes;
    if (!isNeutralElementConst(supportedReductions[i].kind, origInit,
                               vecbuilder))
      finalRes = arith::getReductionOp(supportedReductions[i].kind, builder,
                                       vecForOp.getLoc(), reducedRes, origInit);
    // state.registerLoopResultScalarReplacement(forOp.getResult(i), finalRes);
    forOp.getResult(i).replaceAllUsesWith(finalRes);
  }
  LLVM_DEBUG(vecForOp->getParentOp()->dump());
  return vecForOp->getNumResults() ? vecForOp.getResult(0) : Value{};
}

void VectorizeReduction::accept(PatternVisitor &visitor,
                                Superword const *superword) const {
  visitor.visit(this, superword);
}

// Helper functions in anonymous namespace.
namespace {

/// Creates a constant of type vectorType consisting of all attributes in
/// [begin, end).
template <typename AttributeIterator>
DenseElementsAttr denseElements(AttributeIterator begin, AttributeIterator end,
                                VectorType const &vectorType) {
  if (auto floatType =
          vectorType.getElementType().template dyn_cast<FloatType>()) {
    if (floatType.getWidth() == 32) {
      SmallVector<float, 4> array;
      while (begin != end) {
        array.push_back(static_cast<float>(
            begin->template cast<FloatAttr>().getValue().convertToDouble()));
        ++begin;
      }
      return DenseElementsAttr::get(vectorType,
                                    static_cast<llvm::ArrayRef<float>>(array));
    }
    SmallVector<double, 4> array;
    while (begin != end) {
      array.push_back(
          begin->template cast<FloatAttr>().getValue().convertToDouble());
      ++begin;
    }
    return DenseElementsAttr::get(vectorType,
                                  static_cast<llvm::ArrayRef<double>>(array));
  } else if (auto indexType =
                 vectorType.getElementType().template dyn_cast<IndexType>()) {
    if (indexType.isSignlessIntOrIndex()) {
      SmallVector<int64_t, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<IntegerAttr>().getInt());
        ++begin;
      }
      return DenseElementsAttr::get(
          vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
    } else if (indexType.isUnsignedInteger()) {
      SmallVector<uint64_t, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<IntegerAttr>().getUInt());
        ++begin;
      }
      return DenseElementsAttr::get(
          vectorType, static_cast<llvm::ArrayRef<uint64_t>>(array));
    }
    SmallVector<int64_t, 4> array;
    while (begin != end) {
      array.push_back(begin->template cast<IntegerAttr>().getSInt());
      ++begin;
    }
    return DenseElementsAttr::get(vectorType,
                                  static_cast<llvm::ArrayRef<int64_t>>(array));
  } else if (auto intType =
                 vectorType.getElementType().template dyn_cast<IntegerType>()) {
    if (intType.isSignlessIntOrIndex()) {
      if (intType.getWidth() == 32) {
        SmallVector<int32_t, 4> array;
        while (begin != end) {
          array.push_back(static_cast<int32_t>(
              begin->template cast<IntegerAttr>().getInt()));
          ++begin;
        }
        return DenseElementsAttr::get(
            vectorType, static_cast<llvm::ArrayRef<int32_t>>(array));
      } else if (intType.getWidth() == 64) {
        SmallVector<int64_t, 4> array;
        while (begin != end) {
          array.push_back(begin->template cast<IntegerAttr>().getInt());
          ++begin;
        }
        return DenseElementsAttr::get(
            vectorType, static_cast<llvm::ArrayRef<int64_t>>(array));
      }
    } else if (intType.isUnsignedInteger()) {
      SmallVector<uint64_t, 4> array;
      while (begin != end) {
        array.push_back(begin->template cast<IntegerAttr>().getUInt());
        ++begin;
      }
      return DenseElementsAttr::get(
          vectorType, static_cast<llvm::ArrayRef<uint64_t>>(array));
    }
    SmallVector<int64_t, 4> array;
    while (begin != end) {
      array.push_back(begin->template cast<IntegerAttr>().getSInt());
      ++begin;
    }
    return DenseElementsAttr::get(vectorType,
                                  static_cast<llvm::ArrayRef<int64_t>>(array));
  }
  llvm_unreachable("illegal vector element type");
}
} // namespace

// === VectorizeConstant === //

Value VectorizeConstant::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Attribute, 4> constants;
  for (auto value : *superword) {
    // ConstantOp.
    if (auto op = value.getDefiningOp<arith::ConstantOp>()) {
      constants.emplace_back(op.getValue());
    }
  }
  for (auto attr : constants) {
    LLVM_DEBUG(llvm::dbgs() << "  " << attr << "\n");
  }
  auto const &elements = denseElements(
      std::begin(constants), std::end(constants), superword->getVectorType());
  conversionManager.getOrCreateConstant(superword->getLoc(), elements);
  return rewriter.create<arith::ConstantOp>(
      superword->getElement(0).getDefiningOp()->getLoc(),
      superword->getVectorType(), elements);
}

void VectorizeConstant::accept(PatternVisitor &visitor,
                               Superword const *superword) const {
  visitor.visit(this, superword);
}

// === CreateConsecutiveLoad === //

LogicalResult CreateConsecutiveLoad::match(Superword *superword) {
  // Pattern only applicable to consecutive loads.
  LLVM_DEBUG(llvm::dbgs() << "CreateConsecutiveLoad\n");
  return success(consecutiveLoads(superword->begin(), superword->end()));
}

Value CreateConsecutiveLoad::rewrite(Superword *superword,
                                     RewriterBase &rewriter) {

  SmallVector<Value, 8> combinedIndices;

  auto element = superword->getElement(0);
  if (auto loadOp = dyn_cast<memref::LoadOp>(element.getDefiningOp())) {
    auto indices = loadOp.getIndices();
    combinedIndices.append(indices.begin(), indices.end());
  }
  return rewriter.create<vector::TransferReadOp>(
      superword->getLoc(), superword->getVectorType(),
      dyn_cast<memref::LoadOp>(superword->getElement(0).getDefiningOp())
          .getMemref(),
      combinedIndices);
}

void CreateConsecutiveLoad::accept(PatternVisitor &visitor,
                                   Superword const *superword) const {
  visitor.visit(this, superword);
}

// === CreateGatherLoad === //

LogicalResult CreateGatherLoad::match(Superword *superword) {
  if (failed(
          OpSpecificVectorizationPattern<memref::LoadOp>::match(superword))) {
    return failure();
  }
  if (consecutiveLoads(superword->begin(), superword->end()))
    return failure();

  auto firstLoad = dyn_cast<memref::LoadOp>(
      superword->getElement(0).getDefiningOp());
  if (!firstLoad)
    return failure();

  auto firstCoeff = firstLoad->getAttr("affineCoeff");
  auto firstOffset = firstLoad->getAttr("affineOffset");
  auto firstMap = firstLoad->getAttr("map");
  for (size_t lane = 1; lane < superword->numLanes(); ++lane) {
    auto loadOp =
        dyn_cast<memref::LoadOp>(superword->getElement(lane).getDefiningOp());
    if (!loadOp || loadOp.getMemref() != firstLoad.getMemref())
      return failure();
    if (loadOp.getIndices().size() != firstLoad.getIndices().size())
      return failure();
    if (loadOp->getAttr("affineCoeff") != firstCoeff ||
        loadOp->getAttr("affineOffset") != firstOffset ||
        loadOp->getAttr("map") != firstMap) {
      return failure();
    }
  }
  return success();
}

// === CreateConsecutiveStore === //

LogicalResult CreateConsecutiveStore::match(Superword *superword) {
  // Pattern only applicable to consecutive loads.
  LLVM_DEBUG(llvm::dbgs() << "CreateConsecutiveStore\n");
  return success(consecutiveStores(superword->begin(), superword->end()));
}

Value CreateConsecutiveStore::rewrite(Superword *superword, Value vector,
                                      RewriterBase &rewriter) {

  SmallVector<Value, 8> combinedIndices;

  auto element = superword->getElement(0);
  SmallVector<memref::StoreOp, 8> allStoreOps = getAllStoreUsers(element);
  if (isa<memref::StoreOp>(allStoreOps[0])) {
    auto indices = allStoreOps[0].getIndices();
    combinedIndices.append(indices.begin(), indices.end());
  }

  rewriter.create<vector::TransferWriteOp>(
      superword->getLoc(), vector, allStoreOps[0].getMemref(), combinedIndices);
  return element;
}

void CreateConsecutiveStore::accept(PatternVisitor &visitor,
                                    Superword const *superword) const {
  visitor.visit(this, superword);
}

// Helper function in anonymous namespace.
namespace {
template <typename T>
DenseElementsAttr constantPassThrough(VectorType const &vectorType) {
  SmallVector<T, 4> elements;
  for (auto i = 0; i < vectorType.getNumElements(); ++i) {
    elements.emplace_back(T());
  }
  return DenseElementsAttr::get(vectorType, static_cast<ArrayRef<T>>(elements));
}
} // namespace

Value CreateGatherLoad::rewrite(Superword *superword, RewriterBase &rewriter) {
  auto firstLoad = cast<memref::LoadOp>(superword->getElement(0).getDefiningOp());
  Location loc = superword->getLoc();

  SmallVector<int64_t, 4> coeffs;
  if (auto coeffAttr = firstLoad->getAttrOfType<ArrayAttr>("affineCoeff")) {
    coeffs.reserve(coeffAttr.size());
    for (Attribute attr : coeffAttr)
      coeffs.push_back(attr.cast<IntegerAttr>().getInt());
  } else {
    coeffs.assign(firstLoad.getIndices().size(), 0);
    if (!coeffs.empty())
      coeffs.back() = 1;
  }

  SmallVector<Value, 4> offsets;
  auto baseIndices = firstLoad.getIndices();
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    if (lane == 0) {
      offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      continue;
    }

    auto laneLoad =
        cast<memref::LoadOp>(superword->getElement(lane).getDefiningOp());
    Value totalOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto [dim, baseIndex] : llvm::enumerate(baseIndices)) {
      int64_t coeff = dim < coeffs.size() ? coeffs[dim] : 0;
      Value offset = createIndexOffset(baseIndex,
                                       laneLoad.getIndices()[dim],
                                       coeff, loc, rewriter);
      totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, offset);
    }
    offsets.push_back(totalOffset);
  }

  Value indexVector = packIndexVector(loc, offsets, rewriter);
  auto maskType = VectorType::get(superword->getVectorType().getShape(),
                                  rewriter.getI1Type());
  auto maskAttr = DenseElementsAttr::get(maskType, rewriter.getBoolAttr(true));
  Value mask =
      conversionManager.getOrCreateConstant(loc, maskAttr);

  auto passThruAttr = DenseElementsAttr::get(
      superword->getVectorType(),
      getZeroAttrForType(superword->getElementType(), rewriter));
  Value passThru =
      conversionManager.getOrCreateConstant(loc, passThruAttr);

  SmallVector<Value, 4> gatherIndices(baseIndices.begin(), baseIndices.end());
  if (!gatherIndices.empty())
    gatherIndices.back() = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  return rewriter.create<vector::GatherOp>(
      loc, superword->getVectorType(), firstLoad.getMemref(), gatherIndices,
      indexVector, mask, passThru);
}

void CreateGatherLoad::accept(PatternVisitor &visitor,
                              Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeAdd === //

Value VectorizeAddI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::AddIOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeAddI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

Value VectorizeAddF::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::AddFOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeAddF::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeMul === //

Value VectorizeMulI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  LLVM_DEBUG(llvm::dbgs() << "\nVectorizeMulI::rewrite\n");
  for (unsigned i = 0; i < superword->numOperands(); ++i) {

    LLVM_DEBUG({
      llvm::dbgs() << superword->getOperand(i)->getElement(0) << "\n";
      llvm::dbgs() << superword->getOperand(i)->getElement(1) << "\n";
    });
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }

  return rewriter.create<arith::MulIOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeMulI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

Value VectorizeMulF::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::MulFOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeMulF::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeIndexCast === //

Value VectorizeIndexCast::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }

  auto numLanes = static_cast<int64_t>(superword->numLanes());
  auto resultType = VectorType::get({numLanes}, rewriter.getIndexType());

  return rewriter.create<arith::IndexCastOp>(superword->getLoc(), resultType,
                                             operands[0]);
}

void VectorizeIndexCast::accept(PatternVisitor &visitor,
                                Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeIndirectLoad === //

LogicalResult VectorizeIndirectLoad::match(Superword *superword) {
  for (auto value : *superword) {
    auto loadOp = value.getDefiningOp<memref::LoadOp>();
    if (!loadOp)
      return failure();

    auto indices = loadOp.getIndices();
    if (indices.size() != 1)
      return failure();

    auto *indexDef = indices[0].getDefiningOp();
    if (!indexDef || !isa<arith::IndexCastOp>(indexDef))
      return failure();
  }

  auto firstLoad =
      dyn_cast<memref::LoadOp>(superword->getElement(0).getDefiningOp());
  for (size_t lane = 1; lane < superword->numLanes(); ++lane) {
    auto loadOp =
        dyn_cast<memref::LoadOp>(superword->getElement(lane).getDefiningOp());
    if (loadOp.getMemref() != firstLoad.getMemref())
      return failure();
  }

  if (consecutiveLoads(superword->begin(), superword->end()))
    return failure();

  return success();
}

Value VectorizeIndirectLoad::rewrite(Superword *superword,
                                     RewriterBase &rewriter) {
  auto firstLoad =
      cast<memref::LoadOp>(superword->getElement(0).getDefiningOp());
  Location loc = superword->getLoc();

  Value indexVector;
  if (superword->numOperands() > 0) {
    for (unsigned i = 0; i < superword->numOperands(); ++i) {
      auto *operandWord = superword->getOperand(i);
      if (operandWord->getElement(0).getType().isa<IndexType>()) {
        indexVector = conversionManager.getValue(operandWord);
        break;
      }
    }
  }

  if (!indexVector) {
    SmallVector<Value, 4> indices;
    for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
      auto loadOp =
          cast<memref::LoadOp>(superword->getElement(lane).getDefiningOp());
      indices.push_back(loadOp.getIndices()[0]);
    }
    auto indexVecType = VectorType::get(
        {static_cast<int64_t>(superword->numLanes())}, rewriter.getIndexType());
    Value packed = rewriter.create<arith::ConstantOp>(
        loc, indexVecType,
        DenseElementsAttr::get(indexVecType, rewriter.getIndexAttr(0)));
    for (auto [lane, idx] : llvm::enumerate(indices)) {
      Value lanePos = rewriter.create<arith::ConstantIntOp>(loc, lane, 32);
      packed =
          rewriter.create<vector::InsertElementOp>(loc, idx, packed, lanePos);
    }
    indexVector = packed;
  }

  return rewriter.create<dataflow::VectorIndexLoadOp>(
      loc, superword->getVectorType(), firstLoad.getMemref(), indexVector);
}

void VectorizeIndirectLoad::accept(PatternVisitor &visitor,
                                   Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeIndirectStore === //

LogicalResult VectorizeIndirectStore::match(Superword *superword) {
  for (auto value : *superword) {
    auto storeOps = getAllStoreUsers(value);
    if (storeOps.empty())
      return failure();

    auto storeOp = storeOps.front();
    auto indices = storeOp.getIndices();
    if (indices.size() != 1)
      return failure();

    auto *indexDef = indices[0].getDefiningOp();
    if (!indexDef || !isa<arith::IndexCastOp>(indexDef))
      return failure();
  }

  auto firstStore = getAllStoreUsers(superword->getElement(0)).front();
  for (size_t lane = 1; lane < superword->numLanes(); ++lane) {
    auto storeOp = getAllStoreUsers(superword->getElement(lane)).front();
    if (storeOp.getMemref() != firstStore.getMemref())
      return failure();
  }

  if (consecutiveStores(superword->begin(), superword->end()))
    return failure();

  return success();
}

Value VectorizeIndirectStore::rewrite(Superword *superword, Value vector,
                                      RewriterBase &rewriter) {
  Location loc = superword->getLoc();
  auto firstStore = getAllStoreUsers(superword->getElement(0)).front();

  SmallVector<Value, 4> indices;
  for (size_t lane = 0; lane < superword->numLanes(); ++lane) {
    auto storeOp = getAllStoreUsers(superword->getElement(lane)).front();
    indices.push_back(storeOp.getIndices()[0]);
  }

  auto indexVecType = VectorType::get(
      {static_cast<int64_t>(superword->numLanes())}, rewriter.getIndexType());
  Value packed = rewriter.create<arith::ConstantOp>(
      loc, indexVecType,
      DenseElementsAttr::get(indexVecType, rewriter.getIndexAttr(0)));
  for (auto [lane, idx] : llvm::enumerate(indices)) {
    Value lanePos = rewriter.create<arith::ConstantIntOp>(loc, lane, 32);
    packed =
        rewriter.create<vector::InsertElementOp>(loc, idx, packed, lanePos);
  }

  rewriter.create<dataflow::VectorIndexStoreOp>(loc, vector,
                                                firstStore.getMemref(), packed);
  return superword->getElement(0);
}

void VectorizeIndirectStore::accept(PatternVisitor &visitor,
                                    Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeCmpI === //

Value VectorizeCmpI::rewrite(Superword *superword, RewriterBase &rewriter) {
  auto firstCmp = cast<arith::CmpIOp>(superword->getElement(0).getDefiningOp());
  auto predicate = firstCmp.getPredicate();

  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }

  auto resultType = VectorType::get(
      {static_cast<int64_t>(superword->numLanes())}, rewriter.getI1Type());

  return rewriter.create<arith::CmpIOp>(superword->getLoc(), resultType,
                                        predicate, operands[0], operands[1]);
}

void VectorizeCmpI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeSubI === //

Value VectorizeSubI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::SubIOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeSubI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeSubF === //

Value VectorizeSubF::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i) {
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  }
  return rewriter.create<arith::SubFOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeSubF::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeCmpF === //

Value VectorizeCmpF::rewrite(Superword *superword, RewriterBase &rewriter) {
  auto firstCmp = cast<arith::CmpFOp>(superword->getElement(0).getDefiningOp());
  auto predicate = firstCmp.getPredicate();
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i)
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  auto resultType = VectorType::get(
      {static_cast<int64_t>(superword->numLanes())}, rewriter.getI1Type());
  return rewriter.create<arith::CmpFOp>(superword->getLoc(), resultType,
                                        predicate, operands[0], operands[1]);
}

void VectorizeCmpF::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeAndI === //

Value VectorizeAndI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i)
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  return rewriter.create<arith::AndIOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeAndI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeOrI === //

Value VectorizeOrI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i)
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  return rewriter.create<arith::OrIOp>(superword->getLoc(),
                                       superword->getVectorType(), operands);
}

void VectorizeOrI::accept(PatternVisitor &visitor,
                          Superword const *superword) const {
  visitor.visit(this, superword);
}

// === VectorizeXOrI === //

Value VectorizeXOrI::rewrite(Superword *superword, RewriterBase &rewriter) {
  SmallVector<Value, 2> operands;
  for (unsigned i = 0; i < superword->numOperands(); ++i)
    operands.emplace_back(conversionManager.getValue(superword->getOperand(i)));
  return rewriter.create<arith::XOrIOp>(superword->getLoc(),
                                        superword->getVectorType(), operands);
}

void VectorizeXOrI::accept(PatternVisitor &visitor,
                           Superword const *superword) const {
  visitor.visit(this, superword);
}

// === SLPPatternApplicator === //

void SLPPatternApplicator::matchAndRewrite(Superword *superword,
                                           RewriterBase &rewriter) const {
  LLVM_DEBUG(llvm::dbgs() << "Finding best pattern match...\n");
  auto *pattern = bestMatch(superword);
  if (!pattern) {
    LLVM_DEBUG({
      llvm::dbgs() << "WARNING: could not apply any pattern to superword:\n";
      for (auto it = superword->begin(); it != superword->end(); ++it) {
        if (auto *defOp = it->getDefiningOp())
          llvm::dbgs() << "  " << defOp->getName() << "\n";
        else
          llvm::dbgs() << "  <block argument>\n";
      }
    });
    return;
  }
  pattern->rewriteSuperword(superword, rewriter);
}

void SLPPatternApplicator::rewriteStore(Superword *superword,
                                        RewriterBase &rewriter) const {
  auto *pattern = bestMatch(superword);
  if (!pattern)
    return;
  pattern->rewriteSuperword(superword, rewriter);
}

void SLPPatternApplicator::setPatterns(
    SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>
        &&slpVectorizationPatterns) {
  this->patterns = std::move(slpVectorizationPatterns);
}
