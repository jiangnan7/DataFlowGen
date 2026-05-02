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
  llvm::outs() << "setupConversionFor\n";
  conversionManager.setupConversionFor(superword, this);
  llvm::outs() << "rewrite\n";

  for (auto it = superword->begin(); it != superword->end(); ++it) {
    llvm::outs() << "Value: ";
    if (auto *defOp = it->getDefiningOp()) {
      llvm::outs() << "Defining Operation: " << defOp->getName() << "\n";
      defOp->dump();
      ;
    } else {
      llvm::outs() << "No defining operation\n";
    }
  }

  auto vectorOp = this->rewrite(superword, rewriter);

  auto pattern =
      std::make_unique<CreateConsecutiveStore>(this->conversionManager);
  if (succeeded(pattern->match(superword))) {
    pattern->rewrite(superword, vectorOp, rewriter);
  }

  llvm::outs() << "update\n";
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

/// Strip the log space property off an operation if present, otherwise do
/// nothing.
Value stripLogOrValue(Value value, RewriterBase &rewriter) {
  // if (auto logType = value.getType().dyn_cast<LogType>()) {
  //   return rewriter.create<SPNStripLog>(value.getLoc(), value,
  //   logType.getBaseType());
  // }
  return value;
}

/// Might be useful in the future.
// NOLINTNEXTLINE(clang-diagnostic-unused-function)
[[maybe_unused]] Value castToFloatOrValue(Value value, FloatType targetType,
                                          RewriterBase &rewriter) {
  if (auto floatType = value.getType().dyn_cast<FloatType>()) {
    if (floatType.getWidth() < targetType.getWidth()) {
      return rewriter.create<arith::ExtFOp>(value.getLoc(), targetType, value);
    } else if (floatType.getWidth() > targetType.getWidth()) {
      return rewriter.create<arith::TruncFOp>(value.getLoc(), targetType,
                                              value);
    } else {
      return value;
    }
  } else if (auto intType = value.getType().dyn_cast<IntegerType>()) {
    if (intType.isSigned()) {
      return rewriter.create<arith::SIToFPOp>(value.getLoc(), targetType,
                                              value);
    }
    return rewriter.create<arith::UIToFPOp>(value.getLoc(), targetType, value);
  } else if (value.getType().isa<IndexType>()) {
    auto valueAsInt = rewriter.create<arith::IndexCastOp>(
        value.getLoc(), rewriter.getI64Type(), value);
    return rewriter.create<arith::UIToFPOp>(value.getLoc(), targetType,
                                            valueAsInt);
  }
  llvm_unreachable("value cannot be cast to float");
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
        exit(0);
      definingOp->dump();
      if (firstOp == nullptr)
        return failure();
      firstOp->dump();

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
  llvm::outs() << "VectorizeReduction\n";
  Operation *parentOp;
  for (auto it = superword->begin(); it != superword->end(); ++it) {
    if (auto *defOp = it->getDefiningOp()) {
      parentOp = defOp->getParentOp();
    } else {
      llvm::outs() << "No defining operation\n";
    }
  }
  // Operation *parentOp =
  // superword->getElement(0).getDefiningOp()->getParentOp();
  parentOp->dump();
  auto forOp = dyn_cast<dataflow::ForOp>(parentOp);
  if (!forOp || (forOp.getNumRegionIterArgs() == 0))
    return failure();

  if (true) {
    Operation *terminatorOp = *superword->getElement(superword->numLanes() - 1)
                                   .getDefiningOp()
                                   ->getUsers()
                                   .begin();
    if (terminatorOp == nullptr ||
        !terminatorOp->mightHaveTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
  }
  SmallVector<LoopReduction, 2> supportedReductions;
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    arith::AtomicRMWKind kind;
    if (Value value = getSupportedReduction(forOp, i, kind))
      supportedReductions.emplace_back(LoopReduction{kind, i, value});
  }
  llvm::outs() << "supportedReductionsempty\n";
  if (supportedReductions.empty())
    return failure();

  superword->getElement(0).getDefiningOp()->dump();
  superword->getElement(3).getDefiningOp()->dump();
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

  Operation *parentOp = superword->getElement(0).getDefiningOp()->getParentOp();
  auto forOp = dyn_cast<dataflow::ForOp>(parentOp);

  SmallVector<LoopReduction, 2> supportedReductions;
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
    arith::AtomicRMWKind kind;
    if (Value value = getSupportedReduction(forOp, i, kind))
      supportedReductions.emplace_back(LoopReduction{kind, i, value});
  }
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

  vecForOp->getParentOp()->dump();
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

  vecForOp->dump();

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
  vecForOp->getParentOp()->dump();
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
      // SmallVector<int32_t, 4> array;
      // while (begin != end) {
      //   array.push_back(begin->template cast<IntegerAttr>().getInt());
      //   ++begin;
      // }
      // return DenseElementsAttr::get(vectorType,
      // static_cast<llvm::ArrayRef<int32_t>>(array));
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
    llvm::outs() << "  " << attr << "\n";
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
  llvm::outs() << "CreateConsecutiveLoad\n";
  return success(consecutiveLoads(superword->begin(), superword->end()));
}

Value CreateConsecutiveLoad::rewrite(Superword *superword,
                                     RewriterBase &rewriter) {

  SmallVector<Value, 8> combinedIndices;

  // for(int i=0; i < superword->numLanes(); i++){
  auto element = superword->getElement(0);
  if (auto loadOp = dyn_cast<memref::LoadOp>(element.getDefiningOp())) {
    auto indices = loadOp.getIndices();
    combinedIndices.append(indices.begin(), indices.end());
  }
  // break;
  // }
  // ValueRange indices{
  //     combinedIndices,
  //     conversionManager.getOrCreateConstant(superword->getLoc(),
  //     rewriter.getIndexAttr(combinedIndices.size()))
  // };
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
  Value batchMem = nullptr;
  Value dynamicIndex = nullptr;
  // for (auto element : *superword) {
  //   auto batchRead = element.getDefiningOp<SPNBatchRead>();
  //   if (!batchRead) {
  //     return failure();
  //   }
  //   // We can only gather from the same memory location.
  //   if (!batchMem) {
  //     batchMem = batchRead.batchMem();
  //   } else if (batchRead.batchMem() != batchMem) {
  //     return failure();
  //   }
  //   if (!dynamicIndex) {
  //     dynamicIndex = batchRead.dynamicIndex();
  //     // We require the dynamic index to be 0.
  //     if (auto* definingOp = dynamicIndex.getDefiningOp()) {
  //       auto constant = dyn_cast<ConstantOp>(definingOp);
  //       if (!constant || !constant.getType().isIntOrIndex() ||
  //       constant.getValue().cast<IntegerAttr>().getInt() != 0) {
  //         return failure();
  //       }
  //     } else {
  //       return failure();
  //     }
  //   } else if (batchRead.dynamicIndex() != dynamicIndex) {
  //     return failure();
  //   }
  // }
  return failure();
}

// === CreateConsecutiveStore === //

LogicalResult CreateConsecutiveStore::match(Superword *superword) {
  // Pattern only applicable to consecutive loads.
  llvm::outs() << "CreateConsecutiveStore\n";
  return success(consecutiveStores(superword->begin(), superword->end()));
}

Value CreateConsecutiveStore::rewrite(Superword *superword, Value vector,
                                      RewriterBase &rewriter) {

  SmallVector<Value, 8> combinedIndices;

  // for(int i=superword->numLanes(); i > 0; ){
  //   auto element = superword->getElement(--i);
  // }
  auto element = superword->getElement(0);
  SmallVector<memref::StoreOp, 8> allStoreOps = getAllStoreUsers(element);
  if (isa<memref::StoreOp>(allStoreOps[0])) {
    auto indices = allStoreOps[0].getIndices();
    combinedIndices.append(indices.begin(), indices.end());
  }
  // break;
  // }

  auto storeop = rewriter.create<vector::TransferWriteOp>(
      superword->getLoc(),
      //  superword->getVectorType(),
      vector, allStoreOps[0].getMemref(), combinedIndices);
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
    elements.template emplace_back(T());
  }
  return DenseElementsAttr::get(vectorType, static_cast<ArrayRef<T>>(elements));
}
} // namespace

Value CreateGatherLoad::rewrite(Superword *superword, RewriterBase &rewriter) {
  Value base = nullptr;
  Value index = nullptr;
  SmallVector<uint32_t, 4> samples;
  SmallVector<bool, 4> maskBits;
  // for (auto element : *superword) {
  //   auto batchRead = cast<SPNBatchRead>(element.getDefiningOp());
  //   if (!base && !index) {
  //     base = batchRead.batchMem();
  //     index = batchRead.dynamicIndex();
  //   }
  //   samples.emplace_back(batchRead.staticIndex());
  //   maskBits.emplace_back(true);
  // }

  // // Access the base memref beginning at [0, 0].
  // SmallVector<Value, 2> indices{index, index};

  // auto loc = superword->getLoc();
  // auto vectorType = superword->getVectorType();

  // auto indexType = VectorType::get(vectorType.getShape(),
  // rewriter.getI32Type()); auto indexElements =
  // DenseElementsAttr::get(indexType,
  // static_cast<ArrayRef<uint32_t>>(samples)); auto indexVector =
  // conversionManager.getOrCreateConstant(loc, indexElements);

  // auto maskType = VectorType::get(vectorType.getShape(),
  // rewriter.getI1Type()); auto maskElements = DenseElementsAttr::get(maskType,
  // static_cast<ArrayRef<bool>>(maskBits)); auto mask =
  // conversionManager.getOrCreateConstant(loc, maskElements);

  // DenseElementsAttr passThroughElements;
  // if (superword->getElementType().isIntOrIndex()) {
  //   passThroughElements = constantPassThrough<int>(vectorType);
  // } else if (superword->getElementType().isF32()) {
  //   passThroughElements = constantPassThrough<float>(vectorType);
  // } else if (superword->getElementType().isF64()) {
  //   passThroughElements = constantPassThrough<double>(vectorType);
  // } else {
  //   llvm_unreachable("unsupported vector element type for gather op");
  // }
  // auto passThrough = conversionManager.getOrCreateConstant(loc,
  // passThroughElements);

  // return rewriter.create<vector::GatherOp>(loc, vectorType, base, indices,
  // indexVector, mask, passThrough);
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
  llvm::outs() << "\nVectorizeMulI::rewrite\n";
  for (unsigned i = 0; i < superword->numOperands(); ++i) {

    llvm::outs() << superword->getOperand(i)->getElement(0) << "\n";
    llvm::outs() << superword->getOperand(i)->getElement(1) << "\n";
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

// === SLPPatternApplicator === //

void SLPPatternApplicator::matchAndRewrite(Superword *superword,
                                           RewriterBase &rewriter) const {
  llvm::outs() << "bestMatch\n";
  auto *pattern = bestMatch(superword);
  llvm::outs() << "rewriteSuperword\n";
  if (!pattern) {
    llvm_unreachable("could not apply any pattern to superword");
  }
  pattern->rewriteSuperword(superword, rewriter);
}

void SLPPatternApplicator::rewriteStore(Superword *superword,
                                        RewriterBase &rewriter) const {
  llvm::outs() << "rewriteStore\n";
  auto *pattern = bestMatch(superword);
  llvm::outs() << "rewriteSuperword\n";

  // if (succeeded(pattern->match(superword))) {

  // }
  // SLPVectorizationPattern* bestMatch(Superword* superword) const override {
  //   SLPVectorizationPattern* bestPattern = nullptr;
  //   double bestCost = 0;

  //   for (auto const& pattern : patterns) {
  //     llvm::outs() << "pattern: \n";
  //     if (succeeded(pattern->match(superword))) {
  //       llvm::outs() << "Pattern matched.\n";
  //       auto cost = costModel->getSuperwordCost(superword, pattern.get());
  //       llvm::outs() << "curr: " << cost << "\n";
  //       if (!bestPattern || cost < bestCost) {
  //         bestPattern = pattern.get();
  //         bestCost = cost;
  //       }
  //     }
  //   }
  //   return bestPattern;
  // }

  // if (!pattern) {
  //   llvm_unreachable("could not apply any pattern to superword");
  // }
  pattern->rewriteSuperword(superword, rewriter);
}

void SLPPatternApplicator::setPatterns(
    SmallVectorImpl<std::unique_ptr<SLPVectorizationPattern>>
        &&slpVectorizationPatterns) {
  this->patterns = std::move(slpVectorizationPatterns);
}
