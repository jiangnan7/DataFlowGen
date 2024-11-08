#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Support/FileUtilities.h"

#include "heteacc/Transforms/Passes.h"
#include "heteacc/Transforms/Model.h"

using namespace std;
using namespace mlir;
using namespace heteacc;


void heteacc::getLatencyMap(llvm::json::Object *config,
                             llvm::StringMap<int64_t> &latencyMap) {
  auto Latency =
      config->getObject(config->getString("Latency").value_or("Latency"));

  latencyMap["fadd"] = Latency->getInteger("fadd").value_or(2);
  latencyMap["fmadd"] = Latency->getInteger("fmul").value_or(3);
  latencyMap["fdiv"] = Latency->getInteger("fdiv").value_or(15);
  latencyMap["fcmp"] = Latency->getInteger("fcmp").value_or(1);
  latencyMap["fexp"] = Latency->getInteger("fexp").value_or(8);
}


namespace {
struct EvaluationModel : public EvaluationModelBase<EvaluationModel> {

    EvaluationModel() = default;
    EvaluationModel(std::string TargetSpec) { targetSpec = TargetSpec;}
    void runOnOperation() override {

    // Read target specification JSON file.
    std::string errorMessage;
    auto configFile = mlir::openInputFile(targetSpec, &errorMessage);
    if (!configFile) {
      llvm::errs() << errorMessage << "\n";
      return signalPassFailure();
    }

    // Parse JSON file into memory.
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) {
      llvm::errs() << "failed to parse the target spec json file\n";
      return signalPassFailure();
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) {
      llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
      return signalPassFailure();
    }

    // Collect profiling latency and DSP usage data, where default values are
    // based on ARA.
    // llvm::StringMap<int64_t> latencyMap;
    // getLatencyMap(configObj, latencyMap);
    // llvm::StringMap<int64_t> dspUsageMap;
    // getDspUsageMap(configObj, dspUsageMap);
    // ScaleHLSEstimator(latencyMap, dspUsageMap, true).estimateFunc(func);
  }
};
} // namespace

std::unique_ptr<Pass> heteacc::createEvaluationModelPass(std::string TargetSpec) {
  return std::make_unique<EvaluationModel>(TargetSpec);
}
