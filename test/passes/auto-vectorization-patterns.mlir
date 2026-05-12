// RUN: %heteacc_opt %s --auto-vectorization 2>/dev/null | FileCheck %s

// Test multiple vectorization patterns applied through the auto-vectorization pass.

// --- Test 1: Integer addition (VectorizeAddI) ---
// CHECK-LABEL: func.func @test_addi
// CHECK: dataflow.for
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: vector.transfer_write {{.*}} : vector<4xi32>, memref<16xi32>
module {
  func.func @test_addi(%A: memref<16xi32>, %B: memref<16xi32>, %C: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xi32>
      %b = affine.load %B[%i] : memref<16xi32>
      %c = arith.addi %a, %b : i32
      affine.store %c, %C[%i] : memref<16xi32>
    }
    return
  }

// --- Test 2: Integer multiplication (VectorizeMulI) ---
// CHECK-LABEL: func.func @test_muli
// CHECK: dataflow.for
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: vector.transfer_read {{.*}} : memref<16xi32>, vector<4xi32>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: vector.transfer_write {{.*}} : vector<4xi32>, memref<16xi32>
  func.func @test_muli(%A: memref<16xi32>, %B: memref<16xi32>, %C: memref<16xi32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xi32>
      %b = affine.load %B[%i] : memref<16xi32>
      %c = arith.muli %a, %b : i32
      affine.store %c, %C[%i] : memref<16xi32>
    }
    return
  }

// --- Test 3: Float addition (VectorizeAddF) ---
// CHECK-LABEL: func.func @test_addf
// CHECK: dataflow.for
// CHECK: vector.transfer_read {{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: vector.transfer_read {{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: arith.addf {{.*}} : vector<4xf32>
// CHECK: vector.transfer_write {{.*}} : vector<4xf32>, memref<16xf32>
  func.func @test_addf(%A: memref<16xf32>, %B: memref<16xf32>, %C: memref<16xf32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xf32>
      %b = affine.load %B[%i] : memref<16xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %C[%i] : memref<16xf32>
    }
    return
  }

// --- Test 4: Float multiplication (VectorizeMulF) ---
// CHECK-LABEL: func.func @test_mulf
// CHECK: dataflow.for
// CHECK: vector.transfer_read {{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: vector.transfer_read {{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: arith.mulf {{.*}} : vector<4xf32>
// CHECK: vector.transfer_write {{.*}} : vector<4xf32>, memref<16xf32>
  func.func @test_mulf(%A: memref<16xf32>, %B: memref<16xf32>, %C: memref<16xf32>) {
    affine.for %i = 0 to 16 {
      %a = affine.load %A[%i] : memref<16xf32>
      %b = affine.load %B[%i] : memref<16xf32>
      %c = arith.mulf %a, %b : f32
      affine.store %c, %C[%i] : memref<16xf32>
    }
    return
  }

// --- Test 5: Conditional reduction (if_loop pattern) ---
// CHECK-LABEL: func.func @test_conditional_reduction
// CHECK: dataflow.for {{.*}} iter_args({{.*}}) -> (vector<4xi32>)
// CHECK: vector.transfer_read
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: arith.cmpi {{.*}} : vector<4xi32>
// CHECK: dataflow.select {{.*}} : vector<4xi1>, vector<4xi32>
// CHECK: vector.reduction <add>
  func.func @test_conditional_reduction(%arg0: memref<100xi32>) -> i32 {
    %c2 = arith.constant 2 : i32
    %c10 = arith.constant 10 : i32
    %c0 = arith.constant 0 : i32
    %0 = affine.for %i = 0 to 100 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %arg0[%i] : memref<100xi32>
      %m = arith.muli %v, %c2 : i32
      %cmp = arith.cmpi ugt, %m, %c10 : i32
      %res = scf.if %cmp -> (i32) {
        %add = arith.addi %m, %acc : i32
        scf.yield %add : i32
      } else {
        scf.yield %acc : i32
      }
      affine.yield %res : i32
    }
    return %0 : i32
  }

// --- Test 6: Unconditional reduction (sumi3 pattern) ---
// CHECK-LABEL: func.func @test_unconditional_reduction
// CHECK: dataflow.for {{.*}} iter_args({{.*}}) -> (vector<4xi32>)
// CHECK: vector.transfer_read
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.muli {{.*}} : vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: vector.reduction <add>
  func.func @test_unconditional_reduction(%arg0: memref<200xi32>) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = affine.for %i = 0 to 200 iter_args(%acc = %c0) -> (i32) {
      %v = affine.load %arg0[%i] : memref<200xi32>
      %sq = arith.muli %v, %v : i32
      %cube = arith.muli %sq, %v : i32
      %new_acc = arith.addi %acc, %cube : i32
      affine.yield %new_acc : i32
    }
    return %0 : i32
  }

// --- Test 7: Indirect access with dataflow.vector.load ---
// CHECK-LABEL: func.func @test_indirect_access
// CHECK: dataflow.for
// CHECK: vector.transfer_read
// CHECK: arith.index_cast {{.*}} : vector<4xi32> to vector<4xindex>
// CHECK: dataflow.vector.load
// CHECK: vector.transfer_read
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: dataflow.vector.store
  func.func @test_indirect_access(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>) {
    affine.for %i = 0 to 1024 {
      %val = affine.load %arg1[%i] : memref<1024xi32>
      %idx_i32 = affine.load %arg0[%i] : memref<1024xi32>
      %idx = arith.index_cast %idx_i32 : i32 to index
      %old = memref.load %arg2[%idx] : memref<1024xi32>
      %new = arith.addi %old, %val : i32
      memref.store %new, %arg2[%idx] : memref<1024xi32>
    }
    return
  }
}
