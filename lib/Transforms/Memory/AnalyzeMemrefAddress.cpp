#include "mlir/IR/IntegerSet.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "heteacc/Transforms/Passes.h"

using namespace mlir;
using namespace heteacc;
#include <queue>

namespace {

// using mlir::affine::AffineForOp;
// using mlir::affine::AffineLoadOp;
// using mlir::affine::AffineStoreOp;
using mlir::func::FuncOp;

inline bool isConstant(AffineExpr affineExpr){
  if (auto constExpr = affineExpr.dyn_cast<mlir::AffineConstantExpr>()) {
    return true;
  }else{
    return false;
  }
}

std::vector<unsigned> affineExprAnalysis(AffineExpr affineExpr, std::pair<std::map<unsigned, int64_t>, int64_t> &factorTable){
  std::vector<unsigned> effectedPositions;

  if (auto binExpr = affineExpr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
    // llvm::errs() << "AffineBinaryOpExpr: " ; binExpr.dump();
    auto LHS = binExpr.getLHS(); auto RHS = binExpr.getRHS();
    assert(!isConstant(LHS) || !isConstant(RHS));
    auto binOpKind = binExpr.getKind();
    switch (binOpKind)
    {
    case AffineExprKind::Add:{
      // llvm::errs() << "ADD\n";
      int64_t ConstVal = 0;
      if(!isConstant(LHS) && !isConstant(RHS)){
        effectedPositions = affineExprAnalysis(LHS, factorTable);
        auto RightDims = affineExprAnalysis(RHS, factorTable);
        effectedPositions.insert(effectedPositions.end(), RightDims.begin(), RightDims.end());
      }else if(isConstant(LHS)){
        ConstVal = LHS.dyn_cast<mlir::AffineConstantExpr>().getValue();
        effectedPositions = affineExprAnalysis(RHS, factorTable);
      }else{
        ConstVal = RHS.dyn_cast<mlir::AffineConstantExpr>().getValue();
        effectedPositions = affineExprAnalysis(LHS, factorTable);
      }
      factorTable.second += ConstVal;
      return effectedPositions;
    }
    case AffineExprKind::Mul:{
      /// RHS of mul is always a constant or a symbolic expression.
      // llvm::errs() << "MUL\n";
      int64_t ConstVal;
      if(!isConstant(LHS) && !isConstant(RHS)){
        llvm::errs() << "dim*dim!!!!how to do???\n";
        ConstVal = LHS.dyn_cast<mlir::AffineConstantExpr>().getValue();
        effectedPositions = affineExprAnalysis(LHS, factorTable);
        auto RightDims = affineExprAnalysis(LHS, factorTable);
        effectedPositions.insert(effectedPositions.end(), RightDims.begin(), RightDims.end());
      }else if(isConstant(LHS)){
        ConstVal = LHS.dyn_cast<mlir::AffineConstantExpr>().getValue();
        effectedPositions = affineExprAnalysis(RHS, factorTable);
      }else{
        ConstVal = RHS.dyn_cast<mlir::AffineConstantExpr>().getValue();
        effectedPositions = affineExprAnalysis(LHS, factorTable);
      }
      for(int dim : effectedPositions){
        factorTable.first[dim] *= ConstVal;
      }
      return effectedPositions;
    }
    case AffineExprKind::Mod:
      /// RHS of mod is always a constant or a symbolic expression with a positive value.
      break;
    case AffineExprKind::FloorDiv:
      /// RHS of floordiv is always a constant or a symbolic expression.
    case AffineExprKind::CeilDiv:
      /// RHS of ceildiv is always a constant or a symbolic expression.
      break;
    default:
      break;
    }
  }


  if (auto dimExpr = affineExpr.dyn_cast<mlir::AffineDimExpr>()) {
    unsigned dimIndex = dimExpr.getPosition();
    if(factorTable.first.find(dimIndex) == factorTable.first.end()){
      factorTable.first[dimIndex] = 1;
    }
    effectedPositions.push_back(dimIndex);
    return effectedPositions;
  }
  if (auto symExpr = affineExpr.dyn_cast<mlir::AffineSymbolExpr>()) {

    unsigned symIndex = symExpr.getPosition();

    llvm::errs() << "symIndex: " << symIndex << ", how to do???\n";
    return effectedPositions;
  }
  return effectedPositions;
}

// A pass that manually walks the IR
struct AnalyzeMemrefAddress : AnalyzeMemrefAddressBase<AnalyzeMemrefAddress> {

  void runOnOperation(){
    auto FunctionOp = dyn_cast<FuncOp>(getOperation());
    mlir::OpBuilder builder(&getContext());
    llvm::errs() << "running my Pass!\n"; 
    llvm::errs() << "---------analyzing the loop levels of the kernel--------\n"; 
    std::map<AffineForOp, int> LoopLevels;
    //outter most level is 0
    std::queue<Block*> blockQue;
    for(auto &bb : FunctionOp.getBody().getBlocks()){
      blockQue.push(&bb);
    }
    std::map<int, Value> level2IdxVarMap;
    while(!blockQue.empty()){
      auto currBlock = blockQue.front();
      blockQue.pop();
      
      for (auto &op : *currBlock) {

        if (auto affineFor = dyn_cast<AffineForOp>(op)) {
          int maxDepth = 0; 
          std::function<int(mlir::Operation*, int)> visitOp;
          visitOp = [&](mlir::Operation* op, int currentDepth) -> int {
            int innerMax = 0;
            if (auto forOp = dyn_cast<AffineForOp>(op)) {
              // llvm::errs() << "currentDepth: " << currentDepth << "\n"; 

              level2IdxVarMap[currentDepth] = forOp.getInductionVar();
              auto value = forOp.getInductionVar();
              for (auto &innerOp : forOp.getBody()->getOperations()) {
                innerMax = std::max(innerMax, visitOp(&innerOp, currentDepth + 1));
              } 
              return innerMax;
            }else{
              return currentDepth;
            }
          };
          maxDepth = visitOp(affineFor, 0);
          llvm::errs() << "the max depth of the nested loop: " << maxDepth << "\n";
          LoopLevels[affineFor] = maxDepth;
        }else{
          for(auto &rg : op.getRegions()){
            for(auto &bb : rg.getBlocks()){
              blockQue.push(&bb);
            }
          }
        }
      }
    }

    llvm::errs() << "---------each affineFor--------\n"; 
    for(auto &elem : LoopLevels){
      AffineForOp outterFor = elem.first;
      int totalLevel = elem.second;
      std::vector<AffineForOp> toTraverse;
      toTraverse.push_back(outterFor);
      while(!toTraverse.empty()){
        auto currentFor = toTraverse.back();
        toTraverse.pop_back();
        for (auto &op : currentFor.getBody()->getOperations()){
          if(auto innerFor = dyn_cast<AffineForOp>(op)){
            toTraverse.push_back(innerFor);
          }else if(dyn_cast<AffineLoadOp>(op) || dyn_cast<AffineStoreOp>(op)){
            std::vector<int> effectedLevels;
            AffineMap LSAffineMap;
            MemRefType memRef;
            if(auto load = dyn_cast<AffineLoadOp>(op)){
              llvm::errs() << "Affine Load: "; load.dump();
              LSAffineMap = load.getAffineMap();
              memRef = load.getMemRefType();
              //test to get address claculation
              llvm::errs() << "********test to get address claculation***************\n";
              for(auto idxop : load.getMapOperands()){
                for(auto &elem : level2IdxVarMap){
                  if(elem.second == idxop){
                    llvm::errs() << "level: " << elem.first << "\n";
                    effectedLevels.push_back(elem.first);
                    break;
                  }
                }
              }
              llvm::errs() << "*****************************************************\n";
            }else{
              auto store = dyn_cast<AffineStoreOp>(op);
              llvm::errs() << "Affine Store: "; store.dump();
              LSAffineMap = store.getAffineMap();
              memRef = store.getMemRefType();
              //test to get address claculation
              llvm::errs() << "********test to get address claculation***************\n";
              for(auto idxop : store.getMapOperands()){
                for(auto &elem : level2IdxVarMap){
                  if(elem.second == idxop){
                    llvm::errs() << "level: " << elem.first << "\n";
                    effectedLevels.push_back(elem.first);
                    break;
                  }
                }
              }
              llvm::errs() << "*****************************************************\n";
            }

            llvm::errs() << "AffineMap: "; LSAffineMap.dump();
            auto memoryShape = memRef.getShape();
            
            std::map<unsigned, int64_t> subDimsSpaceMap;
            subDimsSpaceMap[memoryShape.size()-1] = 1;
            for(unsigned dim = memoryShape.size()-1; dim > 0; dim--){
              subDimsSpaceMap[dim-1] = subDimsSpaceMap[dim] * memoryShape[dim];
            }

            int i = 0;
            std::vector<int64_t> affineAttrs;
            std::map<int, int64_t> coeffMap;
            int64_t TotalOffset = 0;
            // Traverse all the affineExpr, and extract coeff of each loop level. Finally multiply with memory size in this dimension
            for (auto affineExpr : LSAffineMap.getResults()) {
              std::pair<std::map<unsigned, int64_t>, int64_t> factorTable; factorTable.second = 0;
              affineExprAnalysis(affineExpr, factorTable);
              llvm::errs() << "memory size: " << subDimsSpaceMap[i] << "\n";
              llvm::errs() << "offset: " << factorTable.second << "; ";
              llvm::errs() << "coeff: ";
              TotalOffset += factorTable.second * subDimsSpaceMap[i];
              for(auto &elem : factorTable.first){
                unsigned level = effectedLevels[elem.first];
                llvm::errs() << "d" << level << " " << elem.second << " ";
                coeffMap[level] += (elem.second * subDimsSpaceMap[i]);
              }
              llvm::errs() << "\n";
              i++;
            }
            llvm::errs() << "elem type size: "; memRef.getElementType().dump();
            //Add attributes
            for(unsigned l = 0; l < totalLevel; l++){
              affineAttrs.push_back(coeffMap[l]);
            }
            auto affineCoeff = builder.getI64ArrayAttr(affineAttrs);
            auto affineOffset = builder.getI64IntegerAttr(TotalOffset);
            op.setAttr("affineCoeff", affineCoeff);
            op.setAttr("affineOffset", affineOffset);
          }else if(dyn_cast<memref::LoadOp>(op) || dyn_cast<memref::StoreOp>(op)){
            llvm::errs() << "-------------------------------------------------------------\n";
            std::vector<Value> addrValues;
            std::set<int64_t> involvedLevels;
            mlir::ValueRange indices;
            if(auto load = dyn_cast<memref::LoadOp>(op)){
              llvm::errs() << "memref Load: "; load.dump();
              indices = load.getIndices();
              // addrValues.insert(addrValues.end(), load.getIndices().begin(), load.getIndices().end());
            }else if(auto store = dyn_cast<memref::StoreOp>(op)){
              llvm::errs() << "memref Store: "; store.dump();
              indices = store.getIndices();
              // addrValues.insert(addrValues.end(), store.getIndices().begin(), store.getIndices().end());
            }
            for(auto index : indices){
              addrValues.push_back(index);
              while(!addrValues.empty()){
                auto addrValue = addrValues.back();
                addrValues.pop_back();
                if(auto op = addrValue.getDefiningOp()){
                  llvm::errs() << "It is op: "; op->dump();
                  addrValues.insert(addrValues.end(), op->getOperands().begin(), op->getOperands().end());
                }else{
                  llvm::errs() << "It is NOT op: ";
                  for(auto &elem : level2IdxVarMap){
                    if(elem.second == addrValue){
                      llvm::errs() << "level: " << elem.first << "\n";
                      involvedLevels.insert(elem.first);
                      break;
                    }
                  }
                }
              }
            }
            llvm::errs() << "involved levels: ";
            for(auto level : involvedLevels){
              llvm::errs() << level <<" ";
            }
            llvm::errs() << "\n";
            llvm::errs() << "-------------------------------------------------------------\n";
            std::vector<int64_t> tempV;
            tempV.assign(involvedLevels.begin(), involvedLevels.end());
            auto invLevels = builder.getI64ArrayAttr(tempV);
            op.setAttr("involvedLevels", invLevels);
          }
        }
      }
    }

    /*
    getOperation()->walk([&](AffineForOp affineFor) {
      llvm::errs() << "<<<<<<<<<temp loop>>>>>>>>>\n"; affineFor->dump();
      int step = affineFor.getStep();
      int lowerbound = affineFor.getConstantLowerBound();
      int upperbound = affineFor.getConstantUpperBound();
      llvm::errs() << "step: " << step << "; lowerbound: " << lowerbound << "; upperbound: " << upperbound << "\n";
      llvm::errs() << "InductionVar: " ; 
      mlir::Value loopIdv = affineFor.getInductionVar();
      loopIdv.dump();
      // for(auto user : affineFor.getInductionVar().getUsers()){
      //   user->dump();
      // }
      
      for (auto &op : affineFor.getBody()->getOperations()) {
        if(dyn_cast<AffineForOp>(op)){
          llvm::errs() << "disgusting affineFor\n" ; 
        }
        if(auto load = dyn_cast<AffineLoadOp>(op)){
          llvm::errs() << "Affine Load: "; load.dump(); 
          auto affineMap = load.getAffineMap();
          auto memoryShape = load.getMemRefType().getShape();
          // 遍历 AffineMap 中的每个结果维度
          int i = 0;
          std::vector<mlir::Attribute> affineAttrs;
          std::map<unsigned, int64_t> coeffMap;
          int64_t TotalOffset = 0;
          // Traverse all the affineExpr, and extract coeff of each loop level. Finally multiply with memory size in this dimension
          for (auto affineExpr : affineMap.getResults()) {
            std::pair<std::map<unsigned, int64_t>, int64_t> factorTable; factorTable.second = 0;
            affineExprAnalysis(affineExpr, factorTable);
            llvm::errs() << "memory size: " << memoryShape[i] << "\n";
            llvm::errs() << "offset: " << factorTable.second << "; ";
            llvm::errs() << "coeff: ";
            TotalOffset += factorTable.second * memoryShape[i];
            for(auto &elem : factorTable.first){
              unsigned level = elem.first;
              llvm::errs() << "d" << level << " " << elem.second << " ";
              coeffMap[level] += (elem.second * memoryShape[i]);
            }
            llvm::errs() << "\n";
            i++;
          }
          llvm::errs() << "elem type size: "; load.getMemRefType().getElementType().dump();

        }else if(auto store = dyn_cast<AffineStoreOp>(op)){
          llvm::errs() << "Affine Store: "; store.dump(); 
          auto affineMap = store.getAffineMap();

          for (auto affineExpr : affineMap.getResults()) {
            // pair<coeffMap<level, coeff>, offset>
            std::pair<std::map<unsigned, int64_t>, int64_t> factorTable; factorTable.second = 0;
            affineExprAnalysis(affineExpr, factorTable);
            llvm::errs() << "offset: " << factorTable.second << "\n";
            llvm::errs() << "coeff: ";
            for(auto &elem : factorTable.first){
              unsigned dim = elem.first;
              llvm::errs() << "d" << dim << " " << elem.second << " ";
            }
            llvm::errs() << "\n";
          }
        }
        // llvm::errs() << i++ << ": " << op->getName().getStringRef() << "  "; op->dump();
      }
    });
    */
  }
};

}

std::unique_ptr<Pass> heteacc::createAnalyzeMemrefAddressPass() {
  return std::make_unique<AnalyzeMemrefAddress>();
}
