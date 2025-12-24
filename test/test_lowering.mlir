// RUN: google-opt %s --convert-google-to-linalg | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_binary_lowering
  func.func @test_binary_lowering(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.addf
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: linalg.generic
    // CHECK: arith.subf
    %1 = google.sub %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: linalg.generic
    // CHECK: arith.mulf
    %2 = google.mul %1, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    return %2 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_unary_lowering
  func.func @test_unary_lowering(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.negf
    %0 = google.neg %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: linalg.generic
    // CHECK: math.exp
    %1 = google.exp %0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: linalg.generic
    // CHECK: math.sqrt
    %2 = google.sqrt %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    return %2 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_activation_lowering
  func.func @test_activation_lowering(%arg0: tensor<128x256xf32>) -> tensor<128x256xf32> {
    // CHECK: linalg.generic
    // CHECK: arith.maximumf
    %0 = google.relu %arg0 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    // CHECK: linalg.generic
    // CHECK: math.tanh
    %1 = google.tanh %0 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    return %1 : tensor<128x256xf32>
  }

  // CHECK-LABEL: func.func @test_matmul_lowering
  func.func @test_matmul_lowering(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
    // CHECK: linalg.matmul
    %0 = google.matmul %arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32> -> tensor<128x512xf32>
    return %0 : tensor<128x512xf32>
  }

  // CHECK-LABEL: func.func @test_shape_lowering
  func.func @test_shape_lowering(%arg0: tensor<32x12x64xf32>) -> tensor<32x768xf32> {
    // CHECK: tensor.reshape
    %0 = google.reshape %arg0 {shape = [32, 768]} : tensor<32x12x64xf32> -> tensor<32x768xf32>
    return %0 : tensor<32x768xf32>
  }
}
