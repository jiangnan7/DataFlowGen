module {
  func.func @sumi3_mem(%arg0: memref<1000xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = dataflow.addr %arg0[%arg1] {memShape = [1000]} : memref<1000xi32>[index] -> i32
            %5 = dataflow.load %4 : i32 -> i32
            %6 = arith.muli %5, %5 : i32
            %7 = arith.muli %6, %5 : i32
            %8 = arith.addi %arg2, %7 : i32
            %9 = arith.addi %arg1, %c1 {Exe = "Loop"} : index
            %10 = arith.cmpi eq, %9, %c1000 {Exe = "Loop"} : index
            dataflow.state %10, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %8 : i32
          }
          dataflow.yield %3 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}