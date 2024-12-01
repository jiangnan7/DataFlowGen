#include "mlir/IR/IntegerSet.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "heteacc/Transforms/Passes.h"
#include "heteacc/Misc/VecUtils.h"
#include "heteacc/Misc/Utils.h"
#include <nlohmann/json.hpp>
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h" 
#include <string>
#include <regex>

using json = nlohmann::json;

using namespace mlir;
using namespace heteacc;


// In this pass, we perform the following analysis:
// 1. Extract innermost loop information for different loop bands.
// 2. Extract potential vectorization factors. (iter arg)
// 3. Output data flow information.

struct HybridAnalysis : public HybridAnalysisBase<HybridAnalysis>  {

    HybridAnalysis() = default;
    HybridAnalysis(std::string output) { outputPath = output;}

    void runOnOperation() override {
        auto func = getOperation();
        std::string errorMessage;
        std::string finalPath;
        if (!outputPath.empty()) {
            finalPath = outputPath;
        } else {
            auto fileName = func->getLoc().dyn_cast<FileLineColLoc>().getFilename().str(); 
            finalPath = std::regex_replace(fileName, std::regex("\\.mlir$"), ".json");
        }

        auto outputFile = mlir::openOutputFile(finalPath , &errorMessage);
        if (!outputFile) {
            llvm::errs() << errorMessage << "\n";
            return;
        }

        // 1. Extract innermost loop information for different loop bands
        // 2. Extract potential vectorization factors (iteration arguments)
        
        llvm::DenseMap<int, mlir::heteacc::dataflow::ForOp> bandMap;

        int loopBand = -1;
        func.walk([&](dataflow::ForOp forOp) {
            
        int loopLevel = -1;
        if (auto loopBandAttr = forOp->getAttr("Loop_Band")) {
            loopBand = loopBandAttr.cast<IntegerAttr>().getInt();
        }
        if (auto loopLevelAttr = forOp->getAttr("Loop_Level")) {
            loopLevel = loopLevelAttr.cast<IntegerAttr>().getInt();
        } 

        if (loopBand != -1) {
            if (bandMap.count(loopBand) == 0) {
                bandMap[loopBand] = forOp;
            } else {
                mlir::heteacc::dataflow::ForOp existingOp = bandMap[loopBand];
                int existingLevel = existingOp->getAttrOfType<mlir::IntegerAttr>("Loop_Level").getInt();

                if (loopLevel > existingLevel) {
                    bandMap[loopBand] = forOp;
                }
            }
        }
        });

        nlohmann::json loopsJson = nlohmann::json::object();    
        for (auto &bandEntry : bandMap) {
            int band = bandEntry.first;
            dataflow::ForOp maxLevelOp = bandEntry.second;
            
            nlohmann::json loopInfo;
            loopInfo["loop_level"] = maxLevelOp->getAttrOfType<mlir::IntegerAttr>("Loop_Level").getInt();

            loopInfo["lower_bound"] = maxLevelOp.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>().getValue().cast<mlir::IntegerAttr>().getInt();
            loopInfo["upper_bound"] = maxLevelOp.getUpperBound().getDefiningOp<mlir::arith::ConstantOp>().getValue().cast<mlir::IntegerAttr>().getInt();
            loopInfo["step"] = maxLevelOp.getStep().getDefiningOp<mlir::arith::ConstantOp>().getValue().cast<mlir::IntegerAttr>().getInt();


            int maxUnrollFactor = loopInfo["upper_bound"].get<int>() /  loopInfo["step"].get<int>(); 
            int minUnrollFactor = 1; 
            int maxVectorFactor = loopInfo["upper_bound"].get<int>() / loopInfo["step"].get<int>();
            int minVectorFactor = 1; 

            nlohmann::json unroll_factor;
            unroll_factor= {
                {"max", maxUnrollFactor},
                {"min", minUnrollFactor}
            };

            nlohmann::json vector_factor;
            vector_factor = {
                {"max", maxVectorFactor},
                {"min", minVectorFactor}
            };

            loopInfo["unroll_factor"] = unroll_factor;
            loopInfo["vector_factor"] = vector_factor;

            loopsJson["band" + std::to_string(band)] = loopInfo;
        }


        nlohmann::json outputJson;
        outputJson["LoopInfo"] = loopsJson;

        // 3. output data flow information.
        std::vector<std::string> opNames;    
        llvm::DenseMap<Value, std::vector<std::string>> valueToUsers;
        int i = 0;
        llvm::DenseMap<mlir::Operation*, std::string> op2name;
        llvm::DenseMap<mlir::Operation*, std::string> opMap;
        func.walk([&](mlir::Operation *op) {
            std::string operationName = op->getName().getStringRef().str();
            opNames.push_back(operationName);
            op2name[op] = operationName +  std::to_string(i++);
            std::string typeStr;
            llvm::raw_string_ostream os(typeStr);
            op->print(os);
            opMap[op] = os.str();
        });

        func.walk([&](mlir::Operation *op) {
            nlohmann::json opJson;
            auto loc = op->getLoc().dyn_cast<mlir::FileLineColLoc>();
            if (loc) {
                opJson["line"] = loc.getLine();
            } else {
                opJson["line"] = -1;  
            }
            
            opJson["name"] = op2name[op];
            opJson["instance"] = opMap[op];
            
            if (op->getNumRegions() > 0) {
                opJson["isBlock"] = true;
                for (mlir::Region &region : op->getRegions()) {
                    nlohmann::json regionJson = nlohmann::json::array(); 

                    for (mlir::Block &block : region.getBlocks()) {
                        nlohmann::json blockJson;
                        blockJson["operations"] = nlohmann::json::array(); 

                        for (mlir::Operation &blockOp : block.getOperations()) {
                            
                            blockJson["operations"].push_back(opMap[&blockOp]);
                        }
                        regionJson.push_back(blockJson);
                    }
                    opJson["regions"].push_back(regionJson);
                }
            } else {
                opJson["isBlock"] = false;
            }

     
            opJson["result_types"] = nlohmann::json::array();
            for (const auto &result : op->getResults()) {
                std::string typeStr;
                llvm::raw_string_ostream os(typeStr);
                result.getType().print(os);
                opJson["result_types"].push_back(os.str());
                for (auto &user : result.getUses()) {
                    opJson["to"].push_back(op2name[user.getOwner()]);
                }
            }

            opJson["operands"] = nlohmann::json::array();  
            opJson["from"] = nlohmann::json::array();    
            for (auto operand : op->getOperands()) {
                std::string typeStr;
                llvm::raw_string_ostream os(typeStr);
                operand.print(os);
                if (auto definingOp = operand.getDefiningOp()) {
                    opJson["operands"].push_back(os.str());
                    opJson["from"].push_back(op2name[definingOp]); 
                } else {

                    opJson["operands"].push_back(os.str()); 
                    opJson["from"].push_back(os.str());
                }
            }

            outputJson["DFG"].push_back(opJson);
        });

        outputJson["operations"] = opNames;
        outputFile->os() << outputJson.dump(4); 
        outputFile->keep(); 
    }
};

std::unique_ptr<Pass> heteacc::createHybridAnalysisPass(std::string output) {
  return std::make_unique<HybridAnalysis>(output);
}
