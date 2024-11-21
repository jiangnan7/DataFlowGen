#ifndef OP_VECTORIZATION_TD
#define OP_VECTORIZATION_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "Vectorization.td"

def Vec_PrintOp : Vec_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Vector_FMAOp :
  Vec_Op<"fma", [
       Pure, AllTypesMatch<["lhs", "rhs", "acc", "result"]>
     ] # ElementwiseMappable.traits>,
    Arguments<(ins AnyVector:$lhs,
                   AnyVector:$rhs,
                   AnyVector:$acc)>,
    Results<(outs AnyVector:$result)> {
  let summary = "vector fused multiply-add";
  let description = [{
    Multiply-add expressions operate on n-D vectors and compute a fused
    pointwise multiply-and-accumulate: `$result = `$lhs * $rhs + $acc`.
    All operands and result have the same vector type. The semantics
    of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#int-fma). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.

    Example:

    ```mlir
    %3 = vectorization.fma %0, %1, %2: vector<8x16xf32>
    ```
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs)";
  let extraClassDeclaration = [{
    VectorType getVectorType() { return getLhs().getType().cast<VectorType>(); }
  }];
}


#endif // OP_VECTORIZATION_TD