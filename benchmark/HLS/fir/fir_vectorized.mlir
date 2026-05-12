module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> i32 {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg2 = %c0 to %c100 step %c4 iter_args(%arg3 = %cst) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg1[%arg2], %c0_i32 : memref<100xi32>, vector<4xi32>
          %5 = arith.muli %4, %4 {slp.group = 2 : i64, slp.lane = 0 : i64} : vector<4xi32>
          %6 = arith.addi %5, %arg3 : vector<4xi32>
          dataflow.yield %6 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

