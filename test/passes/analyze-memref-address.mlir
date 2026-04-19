// RUN: %heteacc_opt %s --analyze-memref-address | FileCheck %s

module {
  func.func @analyze_affine_access(%A: memref<4x8xi32>) {
    affine.for %i = 0 to 4 {
      affine.for %j = 0 to 8 {
        %0 = affine.load %A[%i, %j] : memref<4x8xi32>
        affine.store %0, %A[%i, %j] : memref<4x8xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @analyze_affine_access
// CHECK: affine.load {{.*}} {affineCoeff = [8, 1], affineOffset = 0 : i64}
// CHECK: affine.store {{.*}} {affineCoeff = [8, 1], affineOffset = 0 : i64}
