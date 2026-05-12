#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "heteacc/Misc/Utils.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Transforms/Passes.h"
#include "heteacc/Vectorization/BlockCostModel.h"
#include "heteacc/Vectorization/GraphConversion.h"
#include "heteacc/Vectorization/ReductionVectorization.h"
#include "heteacc/Vectorization/SLPPatternApplicator.h"
#include "heteacc/Vectorization/Seeding.h"
#include "heteacc/Vectorization/VectorizationPatternMacth.h"
using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

namespace {

// Note: slp.group attributes are assigned by the modified LLVM loop unroll pass.

/// Annotate memref.load/store operations with affine access metadata.
/// For simple linear accesses like memref.load %m[%i], this adds:
/// affineCoeff = [1] and affineOffset = 0. This enables the consecutive
/// load/store detection in the SLP framework.
static void annotateMemrefAccess(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());

  funcOp.walk([&](memref::LoadOp loadOp) {
    if (loadOp->hasAttr("affineCoeff"))
      return;

    auto indices = loadOp.getIndices();
    if (indices.empty())
      return;

    // Skip indirect access (index comes from index_cast = non-linear).
    for (auto idx : indices) {
      if (auto *defOp = idx.getDefiningOp()) {
        if (isa<arith::IndexCastOp>(defOp))
          return;
      }
    }

    SmallVector<int64_t> coeffs(indices.size(), 0);
    coeffs.back() = 1;
    loadOp->setAttr("affineCoeff", builder.getI64ArrayAttr(coeffs));
    loadOp->setAttr("affineOffset", builder.getI64IntegerAttr(0));
  });

  funcOp.walk([&](memref::StoreOp storeOp) {
    if (storeOp->hasAttr("affineCoeff"))
      return;

    auto indices = storeOp.getIndices();
    if (indices.empty())
      return;

    // Skip indirect access (index comes from index_cast = non-linear).
    for (auto idx : indices) {
      if (auto *defOp = idx.getDefiningOp()) {
        if (isa<arith::IndexCastOp>(defOp))
          return;
      }
    }

    SmallVector<int64_t> coeffs(indices.size(), 0);
    coeffs.back() = 1;
    storeOp->setAttr("affineCoeff", builder.getI64ArrayAttr(coeffs));
    storeOp->setAttr("affineOffset", builder.getI64IntegerAttr(0));
  });
}

/// Process a single block for SLP vectorization.
static void processSLPBlock(Block *block, MLIRContext *context,
                            unsigned maxAttempts,
                            unsigned maxSuccessfulIterations,
                            unsigned maxNodeSize, unsigned maxLookAhead,
                            bool reorderInstructionsDFS,
                            bool allowDuplicateElements) {
  IRRewriter graphRewriter{context};

  CostModelPatternApplicator<CostModel> applicator;
  auto *costModel = applicator.getCostModel();

  ConversionManager conversionManager{graphRewriter, block, costModel,
                                      reorderInstructionsDFS};

  llvm::SmallVector<std::unique_ptr<SLPVectorizationPattern>, 16> patterns;
  patterns.push_back(std::make_unique<BroadcastSuperword>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeConstant>(conversionManager));
  patterns.push_back(std::make_unique<CreateConsecutiveLoad>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeIndirectLoad>(conversionManager));
  patterns.push_back(std::make_unique<CreateGatherLoad>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeIndexCast>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeAddI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeMulI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeAddF>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeMulF>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeCmpI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeCmpF>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeSubI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeSubF>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeAndI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeOrI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeXOrI>(conversionManager));
  patterns.push_back(std::make_unique<VectorizeReduction>(conversionManager));

  applicator.setPatterns(std::move(patterns));

  Operation *parentOp = block->getParentOp();
  std::unique_ptr<SeedAnalysis> seedAnalysis =
      std::make_unique<TopDownAnalysis>(parentOp, 32);

  auto currentFunctionCost =
      costModel->getBlockCost(block, computeDeadOps(block));

  LLVM_DEBUG(llvm::dbgs() << "\nBlock cost before vectorization: "
                          << currentFunctionCost << "\n");

  SmallVector<Value, 8> seed;
  unsigned successfulIterations = 0;
  unsigned attempts = 0;

  while (successfulIterations < maxSuccessfulIterations &&
         attempts++ < maxAttempts) {
    seed.assign(seedAnalysis->next());
    if (seed.empty())
      break;

    LLVM_DEBUG(llvm::dbgs() << "Attempting SLP with seed of size "
                            << seed.size() << "\n");

    SLPGraph graph{seed, maxNodeSize, maxLookAhead, allowDuplicateElements};

    auto dependencyGraph = graph.dependencyGraph();
    LLVM_DEBUG(llvm::dbgs() << "#Nodes: " << dependencyGraph.numNodes()
                            << " #Edges: " << dependencyGraph.numEdges()
                            << "\n");

    auto order = conversionManager.startConversion(graph);
    LLVM_DEBUG(llvm::dbgs() << "Superwords to process: " << order.size()
                            << "\n");

    bool conversionFailed = false;
    for (auto *superword : order) {
      LLVM_DEBUG({
        llvm::dbgs() << "\nProcessing superword with " << superword->numLanes()
                     << " lanes:\n";
        for (auto it = superword->begin(); it != superword->end(); ++it) {
          if (auto *defOp = it->getDefiningOp())
            llvm::dbgs() << "  " << defOp->getName() << "\n";
        }
      });

      applicator.matchAndRewrite(superword, graphRewriter);

      if (!conversionManager.getConversionState().alreadyComputed(superword)) {
        conversionFailed = true;
        break;
      }
    }

    if (conversionFailed) {
      LLVM_DEBUG(llvm::dbgs() << "Pattern match failed, canceling.\n");
      conversionManager.cancelConversion();
      continue;
    }

    auto vectorizedFunctionCost =
        costModel->getBlockCost(block, computeDeadOps(block));

    LLVM_DEBUG(llvm::dbgs() << "Vectorized cost: " << vectorizedFunctionCost
                            << " (was " << currentFunctionCost << ")\n");

    if (vectorizedFunctionCost > currentFunctionCost) {
      LLVM_DEBUG(llvm::dbgs() << "Not profitable, canceling.\n");
      conversionManager.cancelConversion();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Profitable! Committing.\n");
      conversionManager.finishConversion();
      seedAnalysis->update(order);
      currentFunctionCost = vectorizedFunctionCost;
      ++successfulIterations;
    }
  }

  // Clean up dead operations if we vectorized at least once.
  if (successfulIterations > 0) {
    auto &computedVals =
        conversionManager.getConversionState().getComputedScalarValues();
    for (auto value : computedVals) {
      auto storeOps = getAllStoreUsers(value);
      if (!storeOps.empty()) {
        for (auto &store : storeOps) {
          if (store->use_empty())
            graphRewriter.eraseOp(store);
        }
      }
    }
    for (auto *op : computeDeadOps(block)) {
      if (op->use_empty())
        graphRewriter.eraseOp(op);
    }
  }
}

struct AutoVectorization
    : public PassWrapper<AutoVectorization, OperationPass<ModuleOp>> {
public:
  unsigned maxAttempts;
  unsigned maxSuccessfulIterations;
  unsigned maxNodeSize;
  unsigned maxLookAhead;
  bool reorderInstructionsDFS;
  bool allowDuplicateElements;
  Option<unsigned> factor{
      *this, "factor",
      llvm::cl::desc("Affine loop unroll factor before SLP vectorization"),
      llvm::cl::init(4)};
  Option<bool> skipUnroll{
      *this, "skip-unroll",
      llvm::cl::desc("Skip affine loop unroll before SLP vectorization"),
      llvm::cl::init(false)};

  AutoVectorization() = default;

  AutoVectorization(const AutoVectorization &other)
      : PassWrapper(other), maxAttempts(other.maxAttempts),
        maxSuccessfulIterations(other.maxSuccessfulIterations),
        maxNodeSize(other.maxNodeSize), maxLookAhead(other.maxLookAhead),
        reorderInstructionsDFS(other.reorderInstructionsDFS),
        allowDuplicateElements(other.allowDuplicateElements) {
    factor = other.factor;
    skipUnroll = other.skipUnroll;
  }

  AutoVectorization(unsigned maxAttempts, unsigned maxSuccessfulIterations,
                    unsigned maxNodeSize, unsigned maxLookAhead,
                    bool reorderInstructionsDFS, bool allowDuplicateElements,
                    unsigned factor, bool skipUnroll)
      : maxAttempts(maxAttempts),
        maxSuccessfulIterations(maxSuccessfulIterations),
        maxNodeSize(maxNodeSize), maxLookAhead(maxLookAhead),
        reorderInstructionsDFS(reorderInstructionsDFS),
        allowDuplicateElements(allowDuplicateElements) {
    this->factor = factor;
    this->skipUnroll = skipUnroll;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::math::MathDialect>();
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AutoVectorization)
  StringRef getArgument() const final { return "auto-vectorization"; }
  StringRef getDescription() const final {
    return "Apply SLP auto-vectorization to affine loops";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    // === Phase 1: Loop unrolling (adds slp.group attributes) ===
    if (!skipUnroll) {
      mlir::PassManager unrollPM(context);
      unrollPM.nest<func::FuncOp>().addPass(
          mlir::createLoopUnrollPass(static_cast<int>(factor)));
      if (failed(unrollPM.run(module))) {
        signalPassFailure();
        return;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n=== IR after affine loop unroll (factor=" << factor
                     << ") ===\n";
        module.print(llvm::dbgs());
        llvm::dbgs() << "\n=== end IR after affine loop unroll ===\n";
      });
    }

    // === Phase 2: Generate dataflow IR ===
    {
      mlir::PassManager dfPM(context);
      dfPM.nest<func::FuncOp>().addPass(heteacc::createGenerateDataflowPass());
      if (failed(dfPM.run(module))) {
        signalPassFailure();
        return;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n=== IR after generate-dataflow ===\n";
        module.print(llvm::dbgs());
        llvm::dbgs() << "\n=== end IR after generate-dataflow ===\n";
      });
    }

    // === Phase 3: Optimize dataflow (simplify load/store access) ===
    {
      mlir::PassManager optPM(context);
      optPM.nest<func::FuncOp>().addPass(heteacc::createOptimizeDataflowPass());
      if (failed(optPM.run(module))) {
        signalPassFailure();
        return;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n=== IR after optimize-dataflow ===\n";
        module.print(llvm::dbgs());
        llvm::dbgs() << "\n=== end IR after optimize-dataflow ===\n";
      });
    }

    // === Phase 4: Hybrid branch prediction (if -> select) ===
    {
      mlir::PassManager brPM(context);
      brPM.nest<func::FuncOp>().addPass(
          heteacc::createHybridBranchPredictionPass());
      if (failed(brPM.run(module))) {
        LLVM_DEBUG(llvm::dbgs() << "hybrid-branch-prediction failed, "
                                << "continuing without it.\n");
      }

      LLVM_DEBUG({
        llvm::dbgs() << "\n=== IR after hybrid-branch-prediction ===\n";
        module.print(llvm::dbgs());
        llvm::dbgs() << "\n=== end IR after hybrid-branch-prediction ===\n";
      });
    }

    // === Phase 5: Annotate memref access patterns ===
    module.walk([&](func::FuncOp funcOp) { annotateMemrefAccess(funcOp); });

    // === Phase 6: SLP Vectorization ===
    module.walk([&](dataflow::TaskOp task) {
      llvm::DenseMap<mlir::Operation *, mlir::Block *> op2block;
      for (auto &op : llvm::make_early_inc_range(task.getBody().front())) {
        if (op.getNumRegions() > 0) {
          for (auto &region : op.getRegions()) {
            for (auto &block : region) {
              op2block[&op] = &block;
            }
          }
        }
      }

      for (const auto &entry : op2block) {
        mlir::Block *block = entry.second;

        if (block->empty() ||
            !block->back().hasTrait<OpTrait::IsTerminator>())
          continue;

        bool hasNestedFor = false;
        block->walk([&](dataflow::ForOp) { hasNestedFor = true; });
        if (hasNestedFor)
          continue;

        processSLPBlock(block, context, maxAttempts, maxSuccessfulIterations,
                        maxNodeSize, maxLookAhead, reorderInstructionsDFS,
                        allowDuplicateElements);
      }
    });

    // Also handle the case where no dataflow.task exists (simple loops that
    // weren't converted to dataflow). Process scf.for loop bodies.
    bool hasDataflowTasks = false;
    module.walk([&](dataflow::TaskOp) { hasDataflowTasks = true; });

    if (!hasDataflowTasks) {
      module.walk([&](scf::ForOp forOp) {
        bool hasNestedFor = false;
        forOp.getBody()->walk([&](scf::ForOp) { hasNestedFor = true; });
        if (hasNestedFor)
          return;

        Block *body = forOp.getBody();
        if (body->empty() || !body->back().hasTrait<OpTrait::IsTerminator>())
          return;

        processSLPBlock(body, context, maxAttempts, maxSuccessfulIterations,
                        maxNodeSize, maxLookAhead, reorderInstructionsDFS,
                        allowDuplicateElements);
      });
    }

    // === Phase 6.5: Vectorize conditional reduction loops ===
    vectorizeConditionalReduction(module, factor);

    LLVM_DEBUG({
      llvm::dbgs() << "\n=== IR after conditional reduction vectorization ===\n";
      module.print(llvm::dbgs());
      llvm::dbgs() << "\n=== end ===\n";
    });

    // === Phase 7: Verification, Canonicalization and CSE ===
    if (failed(mlir::verify(module))) {
      LLVM_DEBUG(llvm::dbgs() << "WARNING: IR verification failed after "
                              << "vectorization. Skipping canonicalization.\n");
      return;
    }

    {
      mlir::PassManager pm(context);
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      if (failed(pm.run(module))) {
        signalPassFailure();
        return;
      }
    }

    // === Phase 8: Strip SLP attributes from final output ===
    module.walk([&](Operation *op) {
      op->removeAttr("slp.group");
      op->removeAttr("slp.lane");
    });
  }
};

} // namespace

std::unique_ptr<Pass> heteacc::createAutoVectorizationPass(
    unsigned maxAttempts, unsigned maxSuccessfulIterations,
    unsigned maxNodeSize, unsigned maxLookAhead, bool reorderInstructionsDFS,
    bool allowDuplicateElements, unsigned factor, bool skipUnroll) {
  return std::make_unique<AutoVectorization>(
      maxAttempts, maxSuccessfulIterations, maxNodeSize, maxLookAhead,
      reorderInstructionsDFS, allowDuplicateElements, factor, skipUnroll);
}
