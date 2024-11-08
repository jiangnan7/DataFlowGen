module {
  func.func @dropout_gold(%arg0: memref<1024xi32>, %arg1: memref<128xi32>) -> i32 {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c7 = arith.constant 7 : index
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2:2 = dataflow.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %3:2 = dataflow.execution : i32, i32 {
            %4 = dataflow.merge %c0_i32 or %arg3 {Select = "Loop_Signal"} : i32
            %5 = dataflow.merge %c0_i32 or %arg4 {Select = "Loop_Signal"} : i32
            %6 = dataflow.merge %c0 or %arg2 {Select = "Loop_Signal"} : index
            %7 = arith.index_cast %6 : index to i32
            %8 = arith.andi %6, %c7 : index
            %9 = arith.cmpi slt, %8, %c0 : index
            %10 = arith.addi %8, %c7 : index
            %11 = arith.select %9, %10, %8 : index
            %12 = arith.cmpi eq, %11, %c0 : index
            %13 = arith.shrui %7, %c3_i32 : i32
            %14 = arith.index_cast %13 : i32 to index
            %15 = dataflow.addr %arg1[%14] {memShape = [128]} : memref<128xi32>[index] -> i32
            %16 = dataflow.load %15 : i32 -> i32
            %17 = dataflow.select %12, %16, %5 : i32
            %18 = arith.andi %17, %c1_i32 : i32
            %19 = arith.cmpi ne, %18, %c0_i32 : i32
            %20 = dataflow.addr %arg0[%6] {memShape = [1024]} : memref<1024xi32>[index] -> i32
            %21 = dataflow.load %20 : i32 -> i32
            %22 = arith.muli %21, %c2_i32 : i32
            %23 = dataflow.select %19, %22, %c0_i32 : i32
            %24 = arith.addi %4, %23 : i32
            %25 = arith.shrsi %17, %c1_i32 : i32
            %26 = arith.addi %6, %c1 {Exe = "Loop"} : index
            %27 = arith.cmpi eq, %26, %c1024 {Exe = "Loop"} : index
            dataflow.state %27, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            dataflow.yield %24, %25 {execution_block = 1 : i32} : i32, i32
          }
          dataflow.yield %3#0, %3#1 : i32, i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2#0 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}