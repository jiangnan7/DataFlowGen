#include "heteacc/Dialect/DataFlow/DataFlow.h"
#include "heteacc/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace heteacc;

#define DEBUG_TYPE "chain-fusion"

namespace {

static bool isSupportedBinaryOp(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op);
}

static StringRef getOpcodeName(Operation *op) {
  if (isa<arith::AddIOp>(op))
    return "Add";
  if (isa<arith::SubIOp>(op))
    return "Sub";
  if (isa<arith::MulIOp>(op))
    return "Mul";
  llvm_unreachable("unsupported chain operation");
}

static bool isCommutativeChainOp(Operation *op) {
  return isa<arith::AddIOp, arith::MulIOp>(op);
}

struct ChainCandidate {
  SmallVector<Operation *, 8> ops;
  SmallVector<Value, 8> inputs;
  SmallVector<std::string, 8> opcodes;
};

static bool appendOpToChain(Operation *op, Value previousResult,
                            ChainCandidate &candidate) {
  if (!isSupportedBinaryOp(op) || op->getNumOperands() != 2 ||
      op->getNumResults() != 1)
    return false;

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);

  if (candidate.ops.empty()) {
    candidate.inputs.push_back(lhs);
    candidate.inputs.push_back(rhs);
  } else if (lhs == previousResult) {
    candidate.inputs.push_back(rhs);
  } else if (rhs == previousResult && isCommutativeChainOp(op)) {
    candidate.inputs.push_back(lhs);
  } else {
    return false;
  }

  candidate.ops.push_back(op);
  candidate.opcodes.push_back(getOpcodeName(op).str());
  return true;
}

static ChainCandidate buildChain(Operation *root) {
  ChainCandidate candidate;
  Operation *current = root;
  Value previousResult;

  while (current && isSupportedBinaryOp(current)) {
    if (!appendOpToChain(current, previousResult, candidate))
      break;

    previousResult = current->getResult(0);
    if (!previousResult.hasOneUse())
      break;

    Operation *user = *previousResult.getUsers().begin();
    if (user->getBlock() != current->getBlock() || !isSupportedBinaryOp(user))
      break;
    current = user;
  }

  return candidate;
}

static bool hasInternalDef(ArrayRef<Operation *> ops, Value value) {
  auto *defOp = value.getDefiningOp();
  return defOp && llvm::is_contained(ops, defOp);
}

struct OperatorChainFusion
    : public PassWrapper<OperatorChainFusion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OperatorChainFusion)

  explicit OperatorChainFusion(unsigned minChainLength = 2)
      : minChainLength(minChainLength) {}

  StringRef getArgument() const final { return "chain-fusion"; }
  StringRef getDescription() const final {
    return "Fuse linear arithmetic chains into dataflow.chain ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, dataflow::DataFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());
    SmallVector<Operation *, 16> roots;

    module.walk([&](Operation *op) {
      if (!isSupportedBinaryOp(op))
        return;
      for (Value operand : op->getOperands()) {
        if (hasInternalDef({op}, operand))
          return;
      }
      bool hasChainParent = llvm::any_of(op->getOperands(), [](Value operand) {
        return isSupportedBinaryOp(operand.getDefiningOp());
      });
      if (!hasChainParent)
        roots.push_back(op);
    });

    for (Operation *root : roots) {
      if (!root->getBlock())
        continue;

      ChainCandidate candidate = buildChain(root);
      if (candidate.ops.size() < minChainLength)
        continue;

      Operation *lastOp = candidate.ops.back();
      if (lastOp->use_empty())
        continue;

      rewriter.setInsertionPointAfter(lastOp);
      auto opcodeAttr = rewriter.getStrArrayAttr(candidate.opcodes);
      auto chainOp = rewriter.create<dataflow::ChainOp>(
          lastOp->getLoc(), lastOp->getResult(0).getType(), candidate.inputs,
          opcodeAttr);

      LLVM_DEBUG(llvm::dbgs() << "Fused chain with " << candidate.ops.size()
                              << " ops\n");

      lastOp->getResult(0).replaceAllUsesExcept(chainOp.getResult(), chainOp);
      for (Operation *op : llvm::reverse(candidate.ops)) {
        if (op->use_empty())
          rewriter.eraseOp(op);
      }
    }
  }

private:
  unsigned minChainLength;
};

} // namespace

std::unique_ptr<Pass>
heteacc::createOperatorChainFusionPass(unsigned minChainLength) {
  return std::make_unique<OperatorChainFusion>(minChainLength);
}
