#ifndef OP_DATAFLOW_TD
#define OP_DATAFLOW_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "heteacc/Dialect/DataFlow/DataFlow.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

//===----------------------------------------------------------------------===//
// Dataflow Operations
//===----------------------------------------------------------------------===//


def LaunchOp : DataFlowOp<"launch", [RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">,
      ParentOneOf<["func::FuncOp", "AffineForOp", "ForOp"]>]> {
  let summary = "Represent a dataflow dispatch";
  let description = [{
    Dispatch op has a transparent region that contains a list of task ops to be
    dispatched. This op is designed to organize and manipulate task ops at a
    high level and will be lowered to schedule op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;


  let extraClassDeclaration = [{
    /// Get the terminator yield op.
    YieldOp getYieldOp();
  }];
}


def TaskOp : DataFlowOp<"task", [
      DeclareOpInterfaceMethods<StageLikeInterface>, RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"YieldOp">, HasParent<"LaunchOp">]> {
  let summary = "Represent a dataflow task";
  let description = [{
    Task op has a transparent region that contains a list of ops to be executed
    in the task. This op is designed to represent the task partition at a high
    level and will be lowered to node op for dataflow scheduling.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Get the parent dispatch op.
    LaunchOp getLaunchOp();

    /// Get the terminator yield op.
    YieldOp getYieldOp();

    bool hasHierarchy() {
      return cast<StageLikeInterface>(this->getOperation()).hasHierarchy();
    }
  }];
}


def ExecutionBlockOp : DataFlowOp<"execution", [
      SingleBlock,
      ParentOneOf<["func::FuncOp", "ForOp", "AffineForOp","IfOp"]>]> {
  let summary = "Represent a dataflow schedule";
  let description = [{
    Schedule op has an isolated region to contain a list of dataflow node ops to
    be scheduled. This op can be explicitly marked as legal when all the
    dataflow violations have been resolved and all the nodes has been scheduled.
  }];
  // let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "attr-dict-with-keyword ( `:` type($results)^ )? $body";
  // let hasVerifier = 1;
  // let hasCanonicalizer = 1;

  // let extraClassDeclaration = [{
  //   /// FIXME: Check whether the schedule is dependence free.
  //   bool isDependenceFree();
  // }];
}

def ForOp : DataFlowOp<"for", [AutomaticAllocationScope, DeclareOpInterfaceMethods<LoopLikeOpInterface,
       ["getSingleInductionVar", "getSingleLowerBound", "getSingleStep",
        "getSingleUpperBound"]>,
       AllTypesMatch<["lowerBound", "upperBound", "step"]>,
       SingleBlockImplicitTerminator<"YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "for operation";
  let description = [{
    The "tor.for" operation represents a loop take 3 SSA value as operands
    that represent the lower bound, upper bound and step respectively.  The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value always has type index, which is the
    size of the machine word. The step is a value of type index, required to be
    positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    "scf.yield".

    ```mlir
    tor.for %iv = %lb to %ub step %step on [0, 2] {
      ... // body
    }
    ```
    ```mlir
    %sum_0 = constant 0.0 : f32
    // iter_args binds initial values to the loop's region arguments.
    %sum = scf.for %iv = %lb to %ub step %step on (l to r)
        iter_args(%sum_iter = %sum_0) -> (f32) {
      %t = load %buffer[%iv] : memref<1024xf32>
      %sum_next = addf %sum_iter, %t : f32
      // Yield current iteration sum to next iteration %sum_iter or to %sum
      // if final iteration.
      scf.yield %sum_next : f32
    }
    ```
  }];

  let arguments = (ins SignlessIntegerLike: $lowerBound,
                       SignlessIntegerLike: $upperBound,
                       SignlessIntegerLike: $step,
                       Variadic<AnyType> : $initArgs);

  let results = (outs Variadic<AnyType>: $results);
  let regions = (region SizedRegion<1> : $region);
  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value" : $lowerBound, "Value" : $upperBound, "Value" : $step,
      CArg<"ValueRange", "llvm::None"> : $initArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>", 
        "nullptr">)>
  ];
  
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{

    using BodyBuilderFn = function_ref<void(OpBuilder &, Location, Value, ValueRange)>;
    unsigned getNumInductionVars() { return 1; }
    unsigned getNumControlOperands() { return 3; }
    Value getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front(getNumInductionVars());
    }
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - getNumInductionVars();
    }
    BlockArgument getRegionIterArg(unsigned index) {
      assert(index < getNumRegionIterArgs() &&
        "expected an index less than the number of region iter args");
      return getBody()->getArguments().drop_front(getNumInductionVars())[index];
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }
    MutableArrayRef<OpOperand> getIterOpOperands() {
      return
        getOperation()->getOpOperands().drop_front(getNumControlOperands());
    }
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
  }];

} 


def IfOp : DataFlowOp<"if", [DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getNumRegionInvocations", "getRegionInvocationBounds"]>,
    SingleBlockImplicitTerminator<"YieldOp">, RecursiveMemoryEffects,
    NoRegionArguments]> {

  let summary = "if-then-else operation";
  let description = [{
    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    scf.if %b  {
      ...
    } else {
      ...
    }
    ```

  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$withElseRegion)>

  ];

  let extraClassDeclaration = [{

    Block* thenBlock();
    YieldOp thenYield();
    Block* elseBlock();
    YieldOp elseYield();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

}

def YieldOp : DataFlowOp<"yield", [NoMemoryEffect, ReturnLike, Terminator,
      ParentOneOf<["LaunchOp", "TaskOp", "ForOp", "ExecutionBlockOp", "IfOp"]>]> {
  let summary = "Terminate and yield results of a dispatch or task op";

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "$results attr-dict `:` type($results)";

  let builders = [OpBuilder<(ins), "build($_builder, $_state, llvm::None);">];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : DataFlowOp<"select", [
    AllTypesMatch<["true_value", "false_value", "result"]>,
    DeclareOpInterfaceMethods<InferIntRangeInterface>,
  ] # ElementwiseMappable.traits> {
  let summary = "select operation";
  let description = [{
    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    Example:

    ```mlir
    // Custom form of scalar selection.
    %x = arith.select %cond, %true, %false : i32

    // Generic form of the same operation.
    %x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    // Element-wise vector selection.
    %vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    // Full vector selection.
    %vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
  }];

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);



  // FIXME: Switch this to use the declarative assembly format.
  let hasCustomAssemblyFormat = 1;
}

def MergeOp : DataFlowOp<"merge", [
    AllTypesMatch<["true_value", "false_value", "result"]>,
  ] # ElementwiseMappable.traits> {
  let summary = "merge operation";
  let description = [{
  
  }];

  let arguments = (ins AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);


  let assemblyFormat = [{
  
     $true_value ` ` `or` $false_value  attr-dict `:` type($result)
  }];
}

def EnableOp : DataFlowOp<"enable", []> {

  let summary = "Enable operation for scf::ForOp";
  let description = [{
      The `dataflow.enable` operation represents a conditional branch to `scf::ForOp`
    based on a signal. If the signal is 1, the operation triggers a jump to
    the `scf::ForOp`.

    Example:

    ```mlir
    %1 = ... // Some computation
    %signal = cmpi "eq", %1, constant 1 : i1
    dataflow.enable(%signal) : i1
    ```

  }];
  let arguments = (ins AnyType:$value);
  let results = (outs I1:$signal);
// `(` $signal `)` attr-dict (`:` type($signal)^)?
  let assemblyFormat = [{
  
    `(` $value `:` type($value) `)` attr-dict `:` type($signal)
  }];

}

def StateOp : DataFlowOp<"state", []> {

  let summary = "";
  let description = [{
 
    ```mlir
      dataflow.state %13, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
    ```
  }];

  let arguments = (ins
    I1 : $cond,
    StrAttr:$instanceName,
    StrAttr:$primitiveName
  );
  

  let assemblyFormat = [{
    $cond `,` $instanceName ` ` `or` $primitiveName attr-dict `:` type($cond)
  }];
}


//===----------------------------------------------------------------------===//
// MemoryOp 
//===----------------------------------------------------------------------===//

def AddressOp : DataFlowOp<"addr", []> {
  let summary = "address operation";
  let description = [{
    AddressOp claculates address offset for access only.
  }];

  let arguments = (ins 
                    AnyType:$baseAddr,
                    Variadic<Index>:$dims,
                    I64ArrayAttr:$memShape); 
  let results = (outs AnyInteger:$result);
  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
   $baseAddr  `[` $dims `]` attr-dict `:`type($baseAddr)  `[` type($dims) `]` `->` type($result)
  }];

}


def LoadOp : DataFlowOp<"load", []> {
  let summary = "address operation";
  let description = [{
    non-linear Load
  }];

  let arguments = (ins 
                    AnyInteger:$address);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;

  // let assemblyFormat = [{
  //   $baseAddr $address attr-dict `:` type($baseAddr) type($address) `->` type($result)
  // }];
  let assemblyFormat = [{
    $address attr-dict `:`  type($address) `->` type($result)
  }];

}

def InputOp : DataFlowOp<"input", []> {
  let summary = "address operation";
  let description = [{
    linear Load
  }];

  let arguments = (ins 
                    AnyType:$baseAddr,
                    I64ArrayAttr:$AffineAttrs, 
                    I64Attr:$TotalOffset
                    );
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $baseAddr attr-dict `:` type($baseAddr) `->` type($result)
  }];

}

def StoreOp : DataFlowOp<"store", []> {
  let summary = "address operation";
  let description = [{
    non-linear Store
  }];

  let arguments = (ins 
                    AnyType:$value,
                    AnyInteger:$address);
  // let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;

  // let assemblyFormat = [{
  //   $value $baseAddr $address attr-dict `:` type($value) type($baseAddr) type($address)
  // }];
  let assemblyFormat = [{
    $value  $address attr-dict `:` type($value)  type($address)
  }];

}

def OutputOp : DataFlowOp<"output", []> {
  let summary = "address operation";
  let description = [{
    linear Store
  }];

  let arguments = (ins 
                    AnyType:$value,
                    AnyType:$baseAddr,
                    I64ArrayAttr:$AffineAttrs, 
                    I64Attr:$TotalOffset
                    );
  // let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $value $baseAddr attr-dict `:` type($value) type($baseAddr)
  }];

}


#endif // OP_DataFlow_TD