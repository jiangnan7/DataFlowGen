#map = affine_map<(d0) -> (d0)>
module {
  func.func @getTanh(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c19_i32 = arith.constant 19 : i32
    %c3_i32 = arith.constant 3 : i32
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = memref.load %arg0[%arg1] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 0 : i64} : memref<100xi32>
          %4 = arith.cmpi slt, %3, %c1_i32 {slp.group = 1 : i64, slp.lane = 0 : i64} : i32
          %5 = arith.muli %3, %3 : i32
          %6 = arith.addi %5, %c19_i32 : i32
          %7 = arith.muli %6, %3 : i32
          %8 = arith.muli %7, %3 : i32
          %9 = arith.addi %8, %c3_i32 : i32
          %10 = arith.muli %9, %3 : i32
          %11 = dataflow.select %4, %10, %c1_i32 : i32
          %12 = arith.addi %arg2, %11 {slp.group = 3 : i64, slp.lane = 0 : i64} : i32
          %13 = arith.addi %arg1, %c1 : index
          %14 = memref.load %arg0[%13] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 1 : i64} : memref<100xi32>
          %15 = arith.cmpi slt, %14, %c1_i32 {slp.group = 1 : i64, slp.lane = 1 : i64} : i32
          %16 = arith.muli %14, %14 : i32
          %17 = arith.addi %16, %c19_i32 : i32
          %18 = arith.muli %17, %14 : i32
          %19 = arith.muli %18, %14 : i32
          %20 = arith.addi %19, %c3_i32 : i32
          %21 = arith.muli %20, %14 : i32
          %22 = dataflow.select %15, %21, %c1_i32 : i32
          %23 = arith.addi %12, %22 {slp.group = 3 : i64, slp.lane = 1 : i64} : i32
          %24 = arith.addi %arg1, %c2 : index
          %25 = memref.load %arg0[%24] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 2 : i64} : memref<100xi32>
          %26 = arith.cmpi slt, %25, %c1_i32 {slp.group = 1 : i64, slp.lane = 2 : i64} : i32
          %27 = arith.muli %25, %25 : i32
          %28 = arith.addi %27, %c19_i32 : i32
          %29 = arith.muli %28, %25 : i32
          %30 = arith.muli %29, %25 : i32
          %31 = arith.addi %30, %c3_i32 : i32
          %32 = arith.muli %31, %25 : i32
          %33 = dataflow.select %26, %32, %c1_i32 : i32
          %34 = arith.addi %23, %33 {slp.group = 3 : i64, slp.lane = 2 : i64} : i32
          %35 = arith.addi %arg1, %c3 : index
          %36 = memref.load %arg0[%35] {affineCoeff = [1], affineOffset = 0 : i64, map = #map, slp.group = 0 : i64, slp.lane = 3 : i64} : memref<100xi32>
          %37 = arith.cmpi slt, %36, %c1_i32 {slp.group = 1 : i64, slp.lane = 3 : i64} : i32
          %38 = arith.muli %36, %36 : i32
          %39 = arith.addi %38, %c19_i32 : i32
          %40 = arith.muli %39, %36 : i32
          %41 = arith.muli %40, %36 : i32
          %42 = arith.addi %41, %c3_i32 : i32
          %43 = arith.muli %42, %36 : i32
          %44 = dataflow.select %37, %43, %c1_i32 : i32
          %45 = arith.addi %34, %44 {slp.group = 3 : i64, slp.lane = 3 : i64} : i32
          dataflow.yield %45 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

