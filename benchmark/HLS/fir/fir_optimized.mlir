#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (-d0 + 99)>
module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> i32 {
    %true = arith.constant true
    %c99 = arith.constant 99 : index
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg2 = %c0 to %c100 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = dataflow.addr %arg1[%arg2] {memShape = [100]} : memref<100xi32>[index] -> i32
            %5 = dataflow.load %4 {ID = 0 : i32, affineCoeff = [1], affineOffset = 0 : i64, map = #map} : i32 -> i32
            %6 = arith.muli %arg2, %c-1 : index
            %7 = arith.addi %6, %c99 : index
            %8 = dataflow.addr %arg0[%7] {memShape = [100]} : memref<100xi32>[index] -> i32
            %9 = dataflow.load %8 {ID = 0 : i32, affineCoeff = [-1], affineOffset = 99 : i64, map = #map1} : i32 -> i32
            %10 = arith.muli %5, %9 : i32
            %11 = arith.addi %arg3, %10 : i32
            %12 = arith.addi %arg2, %c1 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32} %11 : i32
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
