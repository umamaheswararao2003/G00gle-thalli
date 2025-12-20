// RUN: google-opt %s --transform-interpreter | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_matmul_tiling
  func.func @test_matmul_tiling(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     tensor.extract_slice
    // CHECK:     tensor.extract_slice
    // CHECK:     google.matmul
    // CHECK:     tensor.insert_slice
    %C = google.matmul %A, %B : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
    return %C : tensor<128x512xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["google.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Tile M and N dimensions with size 32x64
    %tiled, %loops:2 = transform.structured.tile_using_for %matmul tile_sizes [32, 64, 0] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}