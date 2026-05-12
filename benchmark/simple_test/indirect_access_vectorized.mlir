#map = affine_map<(d0) -> (d0)>
module {
  func.func @test_gather(%arg0: memref<1024xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c256 step %c4 {
          %0 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<256xi32>, vector<4xi32>
          %1 = arith.index_cast %0 : vector<4xi32> to vector<4xindex>
          %2 = dataflow.vector.load %arg0[%1] : memref<1024xi32>[vector<4xindex>] -> vector<4xi32>
          vector.transfer_write %2, %arg2[%arg3] : vector<4xi32>, memref<256xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_histogram(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0> : vector<4xindex>
    %c0_i32 = arith.constant 0 : i32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c1024 step %c4 {
          %0 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<1024xi32>, vector<4xi32>
          %1 = arith.index_cast %0 : vector<4xi32> to vector<4xindex>
          %2 = dataflow.vector.load %arg2[%1] : memref<1024xi32>[vector<4xindex>] -> vector<4xi32>
          %3 = vector.transfer_read %arg0[%arg3], %c0_i32 : memref<1024xi32>, vector<4xi32>
          %4 = arith.addi %2, %3 : vector<4xi32>
          %5 = arith.addi %arg3, %c3 : index
          %6 = memref.load %arg1[%5] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %7 = arith.index_cast %6 : i32 to index
          %8 = arith.addi %arg3, %c2 : index
          %9 = memref.load %arg1[%8] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %10 = arith.index_cast %9 : i32 to index
          %11 = arith.addi %arg3, %c1 : index
          %12 = memref.load %arg1[%11] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %13 = arith.index_cast %12 : i32 to index
          %14 = memref.load %arg1[%arg3] {affineCoeff = [1], affineOffset = 0 : i64, map = #map} : memref<1024xi32>
          %15 = arith.index_cast %14 : i32 to index
          %16 = vector.insertelement %15, %cst[%c0_i32 : i32] : vector<4xindex>
          %17 = vector.insertelement %13, %16[%c1_i32 : i32] : vector<4xindex>
          %18 = vector.insertelement %10, %17[%c2_i32 : i32] : vector<4xindex>
          %19 = vector.insertelement %7, %18[%c3_i32 : i32] : vector<4xindex>
          dataflow.vector.store %4, %arg2[%19] : vector<4xi32> -> memref<1024xi32>[vector<4xindex>]
          dataflow.vector.store %1, %arg2[%19] : vector<4xindex> -> memref<1024xi32>[vector<4xindex>]
          %20 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<1024xi32>, vector<4xi32>
          %21 = arith.index_cast %20 : vector<4xi32> to vector<4xindex>
          dataflow.vector.store %21, %arg2[%19] : vector<4xindex> -> memref<1024xi32>[vector<4xindex>]
          %22 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<1024xi32>, vector<4xi32>
          %23 = arith.index_cast %22 : vector<4xi32> to vector<4xindex>
          dataflow.vector.store %23, %arg2[%19] : vector<4xindex> -> memref<1024xi32>[vector<4xindex>]
          %24 = dataflow.vector.load %arg2[%23] : memref<1024xi32>[vector<4xindex>] -> vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
  func.func @test_gather_scale(%arg0: memref<1024xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
    %cst = arith.constant dense<2> : vector<4xi32>
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    dataflow.launch {
      dataflow.task {
        dataflow.for %arg3 = %c0 to %c256 step %c4 {
          %0 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<256xi32>, vector<4xi32>
          %1 = arith.index_cast %0 : vector<4xi32> to vector<4xindex>
          %2 = dataflow.vector.load %arg0[%1] : memref<1024xi32>[vector<4xindex>] -> vector<4xi32>
          %3 = arith.muli %2, %cst : vector<4xi32>
          vector.transfer_write %3, %arg2[%arg3] : vector<4xi32>, memref<256xi32>
          %4 = vector.transfer_read %arg1[%arg3], %c0_i32 : memref<256xi32>, vector<4xi32>
          %5 = arith.index_cast %4 : vector<4xi32> to vector<4xindex>
          %6 = dataflow.vector.load %arg0[%5] : memref<1024xi32>[vector<4xindex>] -> vector<4xi32>
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
      }
    }
    return
  }
}

