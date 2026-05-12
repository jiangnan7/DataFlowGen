#map = affine_map<(d0) -> (d0)>
module {
  func.func @histogram(%arg0: memref<100xi32>, %arg1: memref<100xi32>, %arg2: memref<100xi32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        dataflow.for %arg3 = %c0 to %c100 step %c1 {
          dataflow.execution {
            %0 = dataflow.addr %arg1[%arg3] {memShape = [100]} : memref<100xi32>[index] -> i32
            %1 = dataflow.load %0 {ID = 0 : i32, affineCoeff = [1], affineOffset = 0 : i64, map = #map} : i32 -> i32
            %2 = dataflow.addr %arg0[%arg3] {memShape = [100]} : memref<100xi32>[index] -> i32
            %3 = dataflow.load %2 {ID = 0 : i32, affineCoeff = [1], affineOffset = 0 : i64, map = #map} : i32 -> i32
            %4 = arith.index_cast %3 : i32 to index
            %5 = dataflow.addr %arg2[%4] {memShape = [100]} : memref<100xi32>[index] -> i32
            %6 = dataflow.load %5 {ID = 1 : i32, involvedLevels = [0]} : i32 -> i32
            %7 = arith.addi %6, %1 : i32
            %8 = dataflow.addr %arg2[%4] {memShape = [100]} : memref<100xi32>[index] -> i32
            dataflow.store %7 %8 {ID = 1 : i32, involvedLevels = [0]} : i32 i32
            %9 = arith.addi %arg3, %c1 {Exe = "Loop"} : index
            dataflow.yield {execution_block = 1 : i32}
          }
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}
