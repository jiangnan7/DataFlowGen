#include "heteacc/Graph/GraphGen.h"
#include "heteacc/Misc/Utils.h"
#include "heteacc/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "graph"

void GraphGen::buildMemoryGraph(func::FuncOp func) {

  SmallVector<Block *, 4> targetBlocks;
  func.walk([&](dataflow::ExecutionBlockOp op) {
    targetBlocks.push_back(&op.getBody().front());
  });

  enum class PartitionKind { CYCLIC, BLOCK, NONE };
  using Partition = std::pair<PartitionKind, int64_t>;
  DenseMap<Value, SmallVector<Partition, 4>> partitionsMap;

  int memID = 0;
  for (auto block : targetBlocks) {
    MemAccessesMap accessesMap;
    getMemAccessesMap(*block, accessesMap);

    for (auto [memref, loadStores] : accessesMap) {
      auto memrefType = memref.getType().cast<MemRefType>();
      Type elementType = memrefType.getElementType();
      ArrayRef<int64_t> shape = memrefType.getShape();
      int64_t totalElements = std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<int64_t>());
      SmallVector<Operation *, 16> loads;
      SmallVector<Operation *, 16> stores;

      for (auto *op : loadStores) {
        if (isa<memref::LoadOp>(op) || isa<vector::TransferReadOp>(op) ||
            isa<dataflow::VectorIndexLoadOp>(op)) {
          loads.push_back(op);
        } else if (isa<memref::StoreOp>(op) ||
                   isa<vector::TransferWriteOp>(op) ||
                   isa<dataflow::VectorIndexStoreOp>(op)) {
          stores.push_back(op);
        }
      }
      SmallVector<std::pair<Operation *, Operation *>, 16> accessPairs;
      while (!loads.empty() && !stores.empty()) {
        Operation *loadOp = loads.pop_back_val();
        Operation *storeOp = stores.pop_back_val();

        accessPairs.push_back({loadOp, storeOp});
        this->memop2id[loadOp] = memID;
        this->memop2id[storeOp] = memID;
        this->id2size[memID] = totalElements;
        memID++;
      }
      SmallVector<std::pair<Operation *, Operation *>, 16> loadGroups;
      while (!loads.empty()) {
        Operation *loadOp1 = loads.pop_back_val();
        Operation *loadOp2 = nullptr;

        if (!loads.empty()) {
          loadOp2 = loads.pop_back_val();
        }
        loadGroups.push_back({loadOp1, loadOp2});
        this->memop2id[loadOp1] = memID;
        if (loadOp2) {
          this->memop2id[loadOp2] = memID;
        }
        this->id2size[memID] = totalElements;
        memID++;
      }
      while (!stores.empty()) {
        Operation *storeOp1 = stores.pop_back_val();
        this->memop2id[storeOp1] = memID;
        this->id2size[memID] = totalElements;
        memID++;
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "\n=== memop2id ===\n";);
  for (const auto &[op, id] : memop2id) {
    op->setAttr("ID", mlir::IntegerAttr::get(
                          mlir::IntegerType::get(op->getContext(), 32), id));
    if (this->dependency_graph->memID2Node.find(id) ==
        this->dependency_graph->memID2Node.end()) {
      this->dependency_graph->memID2Node[id] =
          this->dependency_graph->createBufferMemory(id, id2size[id]);
    }

    LLVM_DEBUG(llvm::dbgs()
                   << "Op: " << *op << " --> ID: " << id << "memID2Node[id]: "
                   << this->dependency_graph->memID2Node[id]->getName()
                   << "\n";);
  }

  LLVM_DEBUG(llvm::dbgs() << "====================\n";);
}

namespace {

static AffineMap getIdentityAffineMap(const SmallVectorImpl<Value> &operands,
                                      unsigned rank, MLIRContext *context) {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(rank);
  unsigned dimCount = 0;
  unsigned symbolCount = 0;

  for (auto operand : operands) {
    if (isValidDim(operand))
      exprs.push_back(getAffineDimExpr(dimCount++, context));
    else if (isValidSymbol(operand))
      exprs.push_back(getAffineSymbolExpr(symbolCount++, context));
    else
      return AffineMap();
  }
  return AffineMap::get(dimCount, symbolCount, exprs, context);
}

static SmallVector<AffineMap, 4>
getDimAccessMaps(Operation *op, AffineValueMap valueMap, int64_t dim) {
  // Only keep the mapping result of the target dimension.
  auto baseMap = AffineMap::get(valueMap.getNumDims(), valueMap.getNumSymbols(),
                                valueMap.getResult(dim));

  // Get the permuation map from the transfer read/write op.
  AffineMap permuteMap;
  ArrayRef<int64_t> vectorShape;
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    permuteMap = readOp.getPermutationMap();
    vectorShape = readOp.getVectorType().getShape();
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    permuteMap = writeOp.getPermutationMap();
    vectorShape = writeOp.getVectorType().getShape();
  }

  SmallVector<AffineMap, 4> maps({baseMap});
  if (!permuteMap)
    return maps;

  // Traverse each dimension of the transfered vector.
  for (unsigned i = 0, e = permuteMap.getNumResults(); i < e; ++i) {
    auto dimExpr = permuteMap.getResult(i).dyn_cast<AffineDimExpr>();

    // If the permutation result of the current dimension is equal to the target
    // dimension, we push back the access map of each element of the vector into
    // the "maps" to be returned.
    if (dimExpr && dimExpr.getPosition() == dim) {
      for (int64_t offset = 0, size = vectorShape[i]; offset < size; ++offset) {
        auto map = AffineMap::get(baseMap.getNumDims(), baseMap.getNumSymbols(),
                                  baseMap.getResult(0) + offset);
        maps.push_back(map);
      }
      break;
    }
  }
  return maps;
}

static AffineValueMap getAffineValueMap(Operation *op) {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  SmallVector<Value, 4> operands;
  if (auto loadOp = dyn_cast<mlir::AffineReadOpInterface>(op)) {
    operands = loadOp.getMapOperands();
    map = loadOp.getAffineMap();

  } else if (auto storeOp = dyn_cast<mlir::AffineWriteOpInterface>(op)) {
    operands = storeOp.getMapOperands();
    map = storeOp.getAffineMap();

  } else if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    operands = readOp.getIndices();
    map = getIdentityAffineMap(operands, readOp.getShapedType().getRank(),
                               readOp.getContext());
  } else {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    operands = writeOp.getIndices();
    map = getIdentityAffineMap(operands, writeOp.getShapedType().getRank(),
                               writeOp.getContext());
  }

  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  return AffineValueMap(map, operands);
}

SmallVector<int64_t> createPermutationMap(ArrayRef<Value> vec1,
                                          ArrayRef<Value> vec2) {
  if (llvm::SmallDenseSet<Value>(vec1.begin(), vec1.end()) !=
      llvm::SmallDenseSet<Value>(vec2.begin(), vec2.end()))
    return {};

  SmallVector<int64_t> permutation_map(vec1.size());
  llvm::SmallDenseMap<Value, int> index_map;

  for (size_t i = 0; i < vec1.size(); ++i) {
    index_map[vec1[i]] = i;
  }
  for (size_t i = 0; i < vec2.size(); ++i) {
    permutation_map[i] = index_map[vec2[i]];
  }
  return permutation_map;
}

struct MemoryScheduling : public MemorySchedulingBase<MemoryScheduling> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();

    // Collect target basic blocks to be considered.
    SmallVector<Block *, 4> targetBlocks;

    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands);
    for (auto &band : targetBands)
      targetBlocks.push_back(band.back().getBody());

    enum class PartitionKind { CYCLIC, BLOCK, NONE };
    using Partition = std::pair<PartitionKind, int64_t>;
    DenseMap<Value, SmallVector<Partition, 4>> partitionsMap;

    for (auto block : targetBlocks) {
      MemAccessesMap accessesMap;
      getMemAccessesMap(*block, accessesMap);

      for (auto [memref, loadStores] : accessesMap) {
        auto memrefType = memref.getType().cast<MemRefType>();
        auto &partitions = partitionsMap[memref];

        // If the current partitionsMap is empty, initialize it with no
        // partition.
        if (partitions.empty())
          partitions = SmallVector<Partition, 4>(
              memrefType.getRank(), Partition(PartitionKind::NONE, 1));

        LLVM_DEBUG(llvm::outs()
                       << "\n----------\nArray partition for " << memref;);

        // Find the best partition solution for each dimensions of the
        // memref.
        for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
          // Collect all array access indices of the current dimension.
          SmallVector<AffineValueMap, 4> indices;

          LLVM_DEBUG(llvm::outs() << "\n\nDimension " << dim << "";);

          for (auto accessOp : loadStores) {
            auto valueMap = getAffineValueMap(accessOp);
            if (valueMap.getAffineMap().isEmpty())
              continue;

            auto dimMaps = getDimAccessMaps(accessOp, valueMap, dim);
            for (auto dimMap : dimMaps) {
              // Construct the new valueMap.
              AffineValueMap dimValueMap(dimMap, valueMap.getOperands());
              (void)dimValueMap.canonicalize();

              // Only add unique index.
              if (find_if(indices, [&](auto index) {
                    return index.getAffineMap() == dimValueMap.getAffineMap() &&
                           index.getOperands() == dimValueMap.getOperands();
                  }) == indices.end()) {
                indices.push_back(dimValueMap);
                LLVM_DEBUG(llvm::outs()
                               << "\nIndex: " << dimValueMap.getResult(0););
              }
            }
          }
          auto accessNum = indices.size();

          // Find the max array access distance in the current block.
          unsigned maxDistance = 0;
          unsigned maxCommonDivisor = 0;
          bool requireMux = false;

          for (unsigned i = 0; i < accessNum; ++i) {
            for (unsigned j = i + 1; j < accessNum; ++j) {
              auto lhsIndex = indices[i];
              auto rhsIndex = indices[j];
              auto lhsExpr = lhsIndex.getResult(0);
              auto rhsExpr = rhsIndex.getResult(0);

              if (lhsIndex.getOperands() != rhsIndex.getOperands()) {
                // Here, we try to find a permutation map to make the two index
                // identical.
                auto possiblePermutation = createPermutationMap(
                    lhsIndex.getOperands(), rhsIndex.getOperands());

                if (possiblePermutation.empty()) {
                  // If no permutation map is found, we need to use a mux to
                  // select value from the partitioned array. Meanwhile, we
                  // cannot calculate the distance in this case, so continue.
                  requireMux = true;
                  continue;
                } else {
                  // If a permutation map is found, we need to apply it to the
                  // rhsExpr.
                  SmallVector<AffineExpr, 4> dimReplacements;
                  SmallVector<AffineExpr, 4> symReplacements;
                  for (auto i : possiblePermutation) {
                    if (i < rhsIndex.getNumDims())
                      dimReplacements.push_back(
                          getAffineDimExpr(i, func.getContext()));
                    else
                      symReplacements.push_back(getAffineSymbolExpr(
                          i - rhsIndex.getNumDims(), func.getContext()));
                  }
                  rhsExpr = rhsExpr.replaceDimsAndSymbols(dimReplacements,
                                                          symReplacements);
                }
              }

              LLVM_DEBUG(llvm::outs() << "\nDistance: "
                                      << "(" << lhsExpr << ")"
                                      << " - "
                                      << "(" << rhsExpr << ")";);
              auto newExpr =
                  simplifyAffineExpr(rhsExpr - lhsExpr, lhsIndex.getNumDims(),
                                     lhsIndex.getNumSymbols());

              if (auto constDistance = newExpr.dyn_cast<AffineConstantExpr>()) {
                LLVM_DEBUG(llvm::outs() << " = " << constDistance.getValue(););

                unsigned distance = std::abs(constDistance.getValue());
                maxDistance = std::max(maxDistance, distance);
                maxCommonDivisor = std::gcd(distance, maxCommonDivisor);
              } else
                requireMux = true;
            }
          }
          ++maxDistance;

          // This means all accesses have the same index, and this dimension
          // should not be partitioned.
          if (maxDistance == 1)
            continue;

          // Determine array partition factor and kind.
          // TODO: take storage type into consideration.
          int64_t factor = 1;
          PartitionKind kind = PartitionKind::NONE;
          if (accessNum >= maxDistance) {
            // This means some elements are accessed more than once or exactly
            // once, and successive elements are accessed. In most cases, apply
            // "cyclic" partition should be the best solution.
            factor = maxDistance;
            kind = PartitionKind::CYCLIC;
          } else if (maxCommonDivisor > 1) {
            // This means the memory access is perfectly strided.
            factor = maxDistance;
            while (factor % maxCommonDivisor != 0)
              factor++;
            kind = PartitionKind::CYCLIC;
          } else {
            // This means elements are accessed in a descrete manner however not
            // strided. Typically, "block" partition will be the most benefitial
            // partition strategy.
            factor = accessNum;
            kind = PartitionKind::BLOCK;
          }

          LLVM_DEBUG(llvm::outs() << "\nStretegy: " << " factor=" << factor
                                  << " kind=" << static_cast<int>(kind););

          // TODO: For now, we always pick the partition with the largest
          // factor.
          if (factor > partitions[dim].second) {
            LLVM_DEBUG(llvm::outs() << " (update)";);

            // The rationale here is if the accessing partition index cannot be
            // determined and partition factor is more than 3, a multiplexer
            // will be generated and the memory access operation will be wrapped
            // into a function call, which will cause dependency problems and
            // make the latency and II even worse.
            if (requireMux) {
              for (auto i = 3; i > 0; --i)
                if (factor % i == 0) {
                  partitions[dim] = Partition(kind, i);
                  break;
                }
            } else
              partitions[dim] = Partition(kind, factor);
          }
        }

        LLVM_DEBUG(llvm::outs() << "\n\nAccesses: ";);
        for (auto op : loadStores)
          LLVM_DEBUG(llvm::outs() << "\n" << *op;);
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> heteacc::createMemorySchedulingPass() {
  return std::make_unique<MemoryScheduling>();
}
