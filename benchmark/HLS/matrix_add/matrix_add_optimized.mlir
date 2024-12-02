module {
  func.func @matrix_add(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = dataflow.launch : i32 {
      dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
      %1 = dataflow.task : i32 {
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
          %7 = dataflow.execution : i32 {
            %8 = dataflow.addr %arg0[%arg4] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %9 = dataflow.load %8 : i32 -> i32
            %10 = arith.cmpi ne, %9, %c0_i32 : i32
            %11 = arith.addi %arg5, %9 : i32
            %12 = dataflow.select %10, %11, %arg5 : i32
            %13 = arith.addi %arg4, %c1 {Exe = "Loop"} : index
            %14 = arith.cmpi eq, %13, %c1024 {Exe = "Loop"} : index
            dataflow.state %14, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %12 : i32
          }
          dataflow.yield %7 : i32
        } {Loop_Band = 3 : i32, Loop_Level = 0 : i32}
        dataflow.yield %6 : i32
      }
      %2 = dataflow.task : i32 {
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
          %7 = dataflow.execution : i32 {
            %8 = dataflow.addr %arg1[%arg4] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %9 = dataflow.load %8 : i32 -> i32
            %10 = arith.cmpi ne, %9, %c0_i32 : i32
            %11 = arith.addi %arg5, %9 : i32
            %12 = dataflow.select %10, %11, %arg5 : i32
            %13 = arith.addi %arg4, %c1 {Exe = "Loop"} : index
            %14 = arith.cmpi eq, %13, %c1024 {Exe = "Loop"} : index
            dataflow.state %14, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %12 : i32
          }
          dataflow.yield %7 : i32
        } {Loop_Band = 2 : i32, Loop_Level = 0 : i32}
        dataflow.yield %6 : i32
      }
      %3 = dataflow.task : i32 {
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
          %7 = dataflow.execution : i32 {
            %8 = dataflow.addr %arg2[%arg4] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %9 = dataflow.load %8 : i32 -> i32
            %10 = arith.cmpi ne, %9, %c0_i32 : i32
            %11 = arith.addi %arg5, %9 : i32
            %12 = dataflow.select %10, %11, %arg5 : i32
            %13 = arith.addi %arg4, %c1 {Exe = "Loop"} : index
            %14 = arith.cmpi eq, %13, %c1024 {Exe = "Loop"} : index
            dataflow.state %14, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %12 : i32
          }
          dataflow.yield %7 : i32
        } {Loop_Band = 1 : i32, Loop_Level = 0 : i32}
        dataflow.yield %6 : i32
      }
      %4 = dataflow.task : i32 {
        %6 = dataflow.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
          %7 = dataflow.execution : i32 {
            %8 = dataflow.addr %arg3[%arg4] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %9 = dataflow.load %8 : i32 -> i32
            %10 = arith.cmpi ne, %9, %c0_i32 : i32
            %11 = arith.addi %arg5, %9 : i32
            %12 = dataflow.select %10, %11, %arg5 : i32
            %13 = arith.addi %arg4, %c1 {Exe = "Loop"} : index
            %14 = arith.cmpi eq, %13, %c1024 {Exe = "Loop"} : index
            dataflow.state %14, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32} %12 : i32
          }
          dataflow.yield %7 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %6 : i32
      }
      %6 = arith.addi %1, %2 : i32
      %7 = arith.addi %6, %3 : i32
      %8 = arith.addi %7, %4 : i32
      dataflow.yield %8 : i32
    }
    return %0 : i32
  }
}