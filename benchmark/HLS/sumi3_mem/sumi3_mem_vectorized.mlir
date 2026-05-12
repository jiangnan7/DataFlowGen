module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @sumi3_mem(%arg0: memref<200xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant dense<0> : vector<4xi32>
    %c4 = arith.constant 4 : index
    %c200 = arith.constant 200 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c200 step %c4 iter_args(%arg2 = %cst) -> (vector<4xi32>) {
          %4 = vector.transfer_read %arg0[%arg1], %c0_i32 : memref<200xi32>, vector<4xi32>
          %5 = arith.muli %4, %4 {slp.group = 1 : i64, slp.lane = 0 : i64} : vector<4xi32>
          %6 = arith.muli %5, %4 {slp.group = 2 : i64, slp.lane = 0 : i64} : vector<4xi32>
          %7 = arith.addi %6, %arg2 : vector<4xi32>
          dataflow.yield %7 : vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        %3 = vector.reduction <add>, %2 : vector<4xi32> into i32
        dataflow.yield %3 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

