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
            %0 = dataflow.merge %c0 or %arg3 {Select = "Loop_Signal"} : index
            %1 = dataflow.for %arg4 = %c0 to %c16 step %c1 iter_args(%arg5 = %c0_i32) -> (i32) {
              %5 = dataflow.execution : i32 {
                %6 = dataflow.merge %c0_i32 or %arg5 {Select = "Loop_Signal"} : i32
                %7 = dataflow.merge %c0 or %arg4 {Select = "Loop_Signal"} : index
                %8 = dataflow.addr %arg0[%7] {memShape = [16]} : memref<16xi32>[index] -> i32
                %9 = dataflow.load %8 : i32 -> i32
                %10 = arith.muli %0, %c16 : index
                %11 = arith.addi %7, %10 : index
                %12 = dataflow.addr %arg2[%11] {memShape = [256]} : memref<256xi32>[index] -> i32
                %13 = dataflow.load %12 : i32 -> i32
                %14 = arith.cmpi sgt, %9, %c0_i32 : i32
                %15 = arith.muli %9, %13 : i32
                %16 = arith.addi %15, %13 : i32
                %17 = arith.muli %16, %9 : i32
                %18 = arith.addi %6, %17 : i32
                %19 = dataflow.select %14, %18, %6 : i32
                %20 = arith.addi %7, %c1 {Exe = "Loop"} : index
                %21 = arith.cmpi eq, %20, %c16 {Exe = "Loop"} : index
                dataflow.state %21, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
                dataflow.yield %19 {execution_block = 1 : i32} : i32
              }
              dataflow.yield %5 : i32
            } {Loop_Band = 0 : i32, Loop_Level = 1 : i32}
            %2 = dataflow.addr %arg1[%0] {memShape = [16]} : memref<16xi32>[index] -> i32
            dataflow.store %1 %2 : i32 i32
            %3 = arith.addi %0, %c1 {Exe = "Loop"} : index
            %4 = arith.cmpi eq, %3, %c16 {Exe = "Loop"} : index
            dataflow.state %4, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield  {execution_block = 1 : i32} : 
          }
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}