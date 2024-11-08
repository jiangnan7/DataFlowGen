module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @_Z12getTanhfloatPi(%arg0: memref<1000xi32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 3.7047620000000001 : f64
    %cst_2 = arith.constant 19.523810000000001 : f64
    %c150_i32 = arith.constant 150 : i32
    %0 = affine.for %arg1 = 0 to 1000 iter_args(%arg2 = %cst) -> (f32) {
      %1 = affine.load %arg0[%arg1] : memref<1000xi32>
      %2 = arith.cmpi slt, %1, %c150_i32 : i32
      %3 = scf.if %2 -> (f32) {
        %5 = arith.muli %1, %1 : i32
        %6 = arith.sitofp %5 : i32 to f64
        %7 = arith.addf %6, %cst_2 : f64
        %8 = arith.sitofp %1 : i32 to f64
        %9 = arith.mulf %7, %8 : f64
        %10 = arith.mulf %9, %8 : f64
        %11 = arith.addf %10, %cst_1 : f64
        %12 = arith.mulf %11, %8 : f64
        %13 = arith.truncf %12 : f64 to f32
        scf.yield %13 : f32
      } else {
        scf.yield %cst_0 : f32
      }
      %4 = arith.addf %arg2, %3 : f32
      affine.yield %4 : f32
    }
    return %0 : f32
  }
}