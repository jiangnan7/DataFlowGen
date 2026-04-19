// RUN: %heteacc_opt %s --operation-fusion | FileCheck %s

module {
  func.func @fuse_mul_add(%a: vector<4xf32>, %b: vector<4xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
    %mul = arith.mulf %a, %b : vector<4xf32>
    %add = arith.addf %mul, %acc : vector<4xf32>
    return %add : vector<4xf32>
  }
}

// CHECK-LABEL: func.func @fuse_mul_add
// CHECK: %[[FMA:.+]] = vector.fma %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK: return %[[FMA]] : vector<4xf32>
// CHECK-NOT: arith.mulf
// CHECK-NOT: arith.addf
