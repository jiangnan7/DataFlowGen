module {
  func.func @fir(%arg0: memref<1000xi32>, %arg1: memref<1000xi32>) -> i32 {
    %c999 = arith.constant 999 : index
    %c-1 = arith.constant -1 : index
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
            %5 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
            %6 = dataflow.addr %arg1[%5] {memShape = [1000]} : memref<1000xi32>[index] -> i32
            %7 = dataflow.load %6 : i32 -> i32
            %8 = arith.muli %5, %c-1 : index
            %9 = arith.addi %8, %c999 : index
            %10 = dataflow.addr %arg0[%9] {memShape = [1000]} : memref<1000xi32>[index] -> i32
            %11 = dataflow.load %10 : i32 -> i32
            %12 = arith.muli %7, %11 : i32
            %13 = arith.addi %4, %12 : i32
            %14 = arith.addi %5, %c1 {Exe = "Loop"} : index
            %15 = arith.cmpi eq, %14, %c1000 {Exe = "Loop"} : index
            dataflow.state %15, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield %13 {execution_block = 1 : i32} : i32
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