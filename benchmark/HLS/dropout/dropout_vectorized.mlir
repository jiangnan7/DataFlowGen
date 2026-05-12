#map = affine_map<(d0) -> (d0)>
module {
  func.func @dropout(%arg0: memref<1024xi32>, %arg1: memref<128xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8 = arith.constant 8 : index
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        %2:2 = dataflow.for %arg2 = %c0 to %c1024 step %c4 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
          %3 = arith.index_cast %arg2 {slp.group = 0 : i64, slp.lane = 0 : i64} : index to i32
          %4 = arith.andi %arg2, %c7 : index
          %5 = arith.cmpi slt, %4, %c0 {slp.group = 2 : i64, slp.lane = 0 : i64} : index
          %6 = arith.addi %4, %c8 {slp.group = 3 : i64, slp.lane = 0 : i64} : index
          %7 = arith.select %5, %6, %4 {slp.group = 4 : i64, slp.lane = 0 : i64} : index
          %8 = arith.cmpi eq, %7, %c0 {slp.group = 5 : i64, slp.lane = 0 : i64} : index
          %9 = arith.shrui %3, %c3_i32 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = memref.load %arg1[%10] : memref<128xi32>
          %12 = dataflow.select %8, %11, %arg4 : i32
          %13 = arith.andi %12, %c1_i32 {slp.group = 7 : i64, slp.lane = 0 : i64} : i32
          %14 = arith.cmpi ne, %13, %c0_i32 {slp.group = 8 : i64, slp.lane = 0 : i64} : i32
          %15 = memref.load %arg0[%arg2] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %16 = arith.muli %15, %c2_i32 : i32
          %17 = dataflow.select %14, %16, %c0_i32 : i32
          %18 = arith.addi %arg3, %17 {slp.group = 10 : i64, slp.lane = 0 : i64} : i32
          %19 = arith.shrsi %12, %c1_i32 {slp.group = 11 : i64, slp.lane = 0 : i64} : i32
          %20 = arith.addi %arg2, %c1 : index
          %21 = arith.index_cast %20 {slp.group = 0 : i64, slp.lane = 1 : i64} : index to i32
          %22 = arith.andi %20, %c7 : index
          %23 = arith.cmpi slt, %22, %c0 {slp.group = 2 : i64, slp.lane = 1 : i64} : index
          %24 = arith.addi %22, %c8 {slp.group = 3 : i64, slp.lane = 1 : i64} : index
          %25 = arith.select %23, %24, %22 {slp.group = 4 : i64, slp.lane = 1 : i64} : index
          %26 = arith.cmpi eq, %25, %c0 {slp.group = 5 : i64, slp.lane = 1 : i64} : index
          %27 = arith.shrui %21, %c3_i32 : i32
          %28 = arith.index_cast %27 : i32 to index
          %29 = memref.load %arg1[%28] : memref<128xi32>
          %30 = dataflow.select %26, %29, %19 : i32
          %31 = arith.andi %30, %c1_i32 {slp.group = 7 : i64, slp.lane = 1 : i64} : i32
          %32 = arith.cmpi ne, %31, %c0_i32 {slp.group = 8 : i64, slp.lane = 1 : i64} : i32
          %33 = memref.load %arg0[%20] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %34 = arith.muli %33, %c2_i32 : i32
          %35 = dataflow.select %32, %34, %c0_i32 : i32
          %36 = arith.addi %18, %35 {slp.group = 10 : i64, slp.lane = 1 : i64} : i32
          %37 = arith.shrsi %30, %c1_i32 {slp.group = 11 : i64, slp.lane = 1 : i64} : i32
          %38 = arith.addi %arg2, %c2 : index
          %39 = arith.index_cast %38 {slp.group = 0 : i64, slp.lane = 2 : i64} : index to i32
          %40 = arith.andi %38, %c7 : index
          %41 = arith.cmpi slt, %40, %c0 {slp.group = 2 : i64, slp.lane = 2 : i64} : index
          %42 = arith.addi %40, %c8 {slp.group = 3 : i64, slp.lane = 2 : i64} : index
          %43 = arith.select %41, %42, %40 {slp.group = 4 : i64, slp.lane = 2 : i64} : index
          %44 = arith.cmpi eq, %43, %c0 {slp.group = 5 : i64, slp.lane = 2 : i64} : index
          %45 = arith.shrui %39, %c3_i32 : i32
          %46 = arith.index_cast %45 : i32 to index
          %47 = memref.load %arg1[%46] : memref<128xi32>
          %48 = dataflow.select %44, %47, %37 : i32
          %49 = arith.andi %48, %c1_i32 {slp.group = 7 : i64, slp.lane = 2 : i64} : i32
          %50 = arith.cmpi ne, %49, %c0_i32 {slp.group = 8 : i64, slp.lane = 2 : i64} : i32
          %51 = memref.load %arg0[%38] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %52 = arith.muli %51, %c2_i32 : i32
          %53 = dataflow.select %50, %52, %c0_i32 : i32
          %54 = arith.addi %36, %53 {slp.group = 10 : i64, slp.lane = 2 : i64} : i32
          %55 = arith.shrsi %48, %c1_i32 {slp.group = 11 : i64, slp.lane = 2 : i64} : i32
          %56 = arith.addi %arg2, %c3 : index
          %57 = arith.index_cast %56 {slp.group = 0 : i64, slp.lane = 3 : i64} : index to i32
          %58 = arith.andi %56, %c7 : index
          %59 = arith.cmpi slt, %58, %c0 {slp.group = 2 : i64, slp.lane = 3 : i64} : index
          %60 = arith.addi %58, %c8 {slp.group = 3 : i64, slp.lane = 3 : i64} : index
          %61 = arith.select %59, %60, %58 {slp.group = 4 : i64, slp.lane = 3 : i64} : index
          %62 = arith.cmpi eq, %61, %c0 {slp.group = 5 : i64, slp.lane = 3 : i64} : index
          %63 = arith.shrui %57, %c3_i32 : i32
          %64 = arith.index_cast %63 : i32 to index
          %65 = memref.load %arg1[%64] : memref<128xi32>
          %66 = dataflow.select %62, %65, %55 : i32
          %67 = arith.andi %66, %c1_i32 {slp.group = 7 : i64, slp.lane = 3 : i64} : i32
          %68 = arith.cmpi ne, %67, %c0_i32 {slp.group = 8 : i64, slp.lane = 3 : i64} : i32
          %69 = memref.load %arg0[%56] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %70 = arith.muli %69, %c2_i32 : i32
          %71 = dataflow.select %68, %70, %c0_i32 : i32
          %72 = arith.addi %54, %71 {slp.group = 10 : i64, slp.lane = 3 : i64} : i32
          %73 = arith.shrsi %66, %c1_i32 {slp.group = 11 : i64, slp.lane = 3 : i64} : i32
          dataflow.yield %72, %73 : i32, i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2#0 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}

