#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IntegerSet.h"
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
#include "heteacc/Vectorization/SLPPatternApplicator.h"
#include "heteacc/Vectorization/Seeding.h"
#include "heteacc/Vectorization/VectorizationPatternMacth.h"
using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "auto-vectorization"

namespace {

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
  StringRef getDescription() const final { return ""; }
  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();
    ConversionTarget target(getContext());

    if (!skipUnroll) {
      mlir::PassManager unrollPM(&getContext());
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

      auto processBlock = [&](mlir::Operation *op, mlir::Block *block) {
        IRRewriter graphRewriter{context};
        llvm::outs() << "\nProcessing operation: " << op->getName() << "\n";

        CostModelPatternApplicator<CostModel> applicator;
        auto *costModel = applicator.getCostModel();

        ConversionManager conversionManager{graphRewriter, block, costModel,
                                            reorderInstructionsDFS};

        llvm::SmallVector<std::unique_ptr<SLPVectorizationPattern>, 16>
            patterns;
        patterns.push_back(
            std::make_unique<BroadcastSuperword>(conversionManager));

        patterns.push_back(
            std::make_unique<VectorizeConstant>(conversionManager));
        patterns.push_back(
            std::make_unique<CreateConsecutiveLoad>(conversionManager));
        patterns.push_back(
            std::make_unique<CreateGatherLoad>(conversionManager));
        patterns.push_back(std::make_unique<VectorizeAddI>(conversionManager));
        patterns.push_back(std::make_unique<VectorizeMulI>(conversionManager));
        patterns.push_back(std::make_unique<VectorizeAddF>(conversionManager));
        patterns.push_back(std::make_unique<VectorizeMulF>(conversionManager));
        patterns.push_back(
            std::make_unique<VectorizeReduction>(conversionManager));

        applicator.setPatterns(std::move(patterns));

        std::unique_ptr<SeedAnalysis> seedAnalysis;
        seedAnalysis = std::make_unique<TopDownAnalysis>(op, 32);

        auto currentFunctionCost =
            costModel->getBlockCost(block, computeDeadOps(block));

        llvm::outs() << "\nCost: " << currentFunctionCost << "\n";

        SmallVector<Value, 8> seed;
        unsigned successfulIterations = 0;
        unsigned attempts = 0;

        while (successfulIterations < maxSuccessfulIterations &&
               attempts++ < maxAttempts) {
          seed.assign(seedAnalysis->next());
          if (seed.empty()) {
            break;
          }

          SLPGraph graph{seed, maxNodeSize, maxLookAhead,
                         allowDuplicateElements};

          auto dependencyGraph = graph.dependencyGraph();
          llvm::outs() << "#Nodes in dependency graph: "
                       << dependencyGraph.numNodes() << "\n";
          llvm::outs() << "#Edges in dependency graph: "
                       << dependencyGraph.numEdges() << "\n";

          auto order = conversionManager.startConversion(graph);
          llvm::outs() << "matchAndRewrite: " << order.size() << "\n";
          // Traverse the SLP graph and apply the vectorization patterns.
          for (auto *superword : order) {

            llvm::outs() << "\nTraverse the SLP graph and apply the "
                            "vectorization patterns.\n";
            for (auto it = superword->begin(); it != superword->end(); ++it) {
              llvm::outs() << "Value: ";
              if (auto *defOp = it->getDefiningOp()) {
                llvm::outs()
                    << "Defining Operation: " << defOp->getName() << "\n";
                defOp->dump();
              } else {
                llvm::outs() << "No defining operation\n";
              }
            }
            task.getLaunchOp()->getParentOp()->dump();
            applicator.matchAndRewrite(superword, graphRewriter);

            // if(superword->isStoreValue){
            //     applicator.rewriteStore(superword, graphRewriter);
            // }
          }

          auto vectorizedFunctionCost =
              costModel->getBlockCost(block, computeDeadOps(block));

          llvm::outs() << "\nvectorizedFunctionCost: " << vectorizedFunctionCost
                       << "\n";

          // Vectorization not profitable.
          if (vectorizedFunctionCost > currentFunctionCost) {

            llvm::outs() << "cancelConversion\n";
            task.getLaunchOp()->getParentOp()->dump();
            conversionManager.cancelConversion();
          }
          // Vectorization profitable.
          else {
            llvm::outs() << "finishConversion\n";
            conversionManager.finishConversion();
            seedAnalysis->update(order);
            currentFunctionCost = vectorizedFunctionCost;

            ++successfulIterations;
          }
        }
        // if(auto forOp = dyn_cast<dataflow::ForOp>(op)){
        //     ArrayRef<LoopReduction> reductions;
        //     if (forOp.getNumIterOperands() > 0) {
        //         // auto it = strategy.reductionLoops.find(forOp);
        //         // assert(it != strategy.reductionLoops.end() &&
        //         //     "Reduction descriptors not found when vectorizing a
        //         reduction loop");
        //         // reductions = it->second;
        //         // assert(reductions.size() == forOp.getNumIterOperands() &&
        //         //     "The size of reductions array must match the number of
        //         iter_args");

        //         // Vectorize 'iter_args'.
        //         SmallVector<Value, 8> vecIterOperands;
        //         for (auto operand : forOp.getIterOperands())
        //             vecIterOperands.push_back(Value);
        //     }
        // }

        // A lot of operations won't be needed anymore if we vectorized at least
        // once.
        if (successfulIterations > 0) {
          llvm::outs() << "successfulIterations\n";
          task.getLaunchOp()->getParentOp()->dump();
          auto &computedVals =
              conversionManager.getConversionState().getComputedScalarValues();
          for (auto value : computedVals) {
            auto storeOps = getAllStoreUsers(value);
            if (!storeOps.empty()) { //得确定是已经转换的storeop
              for (auto &store : storeOps)
                graphRewriter.eraseOp(store);
            }
          }
          for (auto *op : computeDeadOps(block)) {
            graphRewriter.eraseOp(op);
          }
        }
      };

      assert(op2block.size() > 0);
      llvm::outs() << " op2block.size() " << op2block.size();
      for (const auto &entry : op2block) {
        mlir::Operation *op = entry.first;
        mlir::Block *block = entry.second;
        processBlock(op, block);
      }
    });

    mlir::PassManager pm(&getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }
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
