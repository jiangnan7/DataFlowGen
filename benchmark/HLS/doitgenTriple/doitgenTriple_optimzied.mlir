module {
  func.func @doitgenTriple(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<256xi32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    dataflow.launch {
      dataflow.task {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        dataflow.for %arg3 = %c0 to %c16 step %c1 {
          dataflow.execution {
            %0 = dataflow.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
              %4 = dataflow.execution : i32 {
                %5 = dataflow.addr %arg0[%arg4] {memShape = [16]} : memref<16xi32>[index] -> i32
                %6 = dataflow.load %5 : i32 -> i32
                %7 = arith.muli %arg3, %c16 : index
                %8 = arith.addi %arg4, %7 : index
                %9 = dataflow.addr %arg2[%8] {memShape = [256]} : memref<256xi32>[index] -> i32
                %10 = dataflow.load %9 : i32 -> i32
                %11 = arith.cmpi sgt, %6, %c0_i32 : i32
                %12 = arith.muli %6, %10 : i32
                %13 = arith.addi %12, %10 : i32
                %14 = arith.muli %13, %6 : i32
                %15 = arith.addi %arg5, %14 : i32
                %16 = dataflow.select %11, %15, %arg5 : i32
                %17 = arith.addi %arg4, %c1 {Exe = "Loop"} : index
                %18 = arith.cmpi eq, %17, %c16 {Exe = "Loop"} : index
                dataflow.state %18, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
                dataflow.yield {execution_block = 1 : i32} %16 : i32
              }
              dataflow.yield %4 : i32
            } {Loop_Band = 0 : i32, Loop_Level = 1 : i32}
            %1 = dataflow.addr %arg1[%arg3] {memShape = [16]} : memref<16xi32>[index] -> i32
            dataflow.store %0 %1 : i32 i32
            %2 = arith.addi %arg3, %c1 {Exe = "Loop"} : index
            %3 = arith.cmpi eq, %2, %c16 {Exe = "Loop"} : index
            dataflow.state %3, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield {execution_block = 1 : i32}
          }
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}