// RUN: google-opt %s | FileCheck %s

// Test file for all newly implemented operations

module {
  // CHECK-LABEL: func.func @test_binary_ops
  func.func @test_binary_ops(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: google.sub
    %0 = google.sub %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.mul
    %1 = google.mul %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.div
    %2 = google.div %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.pow
    %3 = google.pow %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    
    return %3 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_unary_ops
  func.func @test_unary_ops(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: google.neg
    %0 = google.neg %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.abs
    %1 = google.abs %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.sqrt
    %2 = google.sqrt %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.rsqrt
    %3 = google.rsqrt %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.exp
    %4 = google.exp %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.log
    %5 = google.log %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    // CHECK: google.ceil
    %6 = google.ceil %arg0 : tensor<4xf32> -> tensor<4xf32>
    
    return %6 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @test_activations
  func.func @test_activations(%arg0: tensor<128x256xf32>) -> tensor<32x1000xf32> {
    // CHECK: google.relu
    %0 = google.relu %arg0 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    // CHECK: google.gelu
    %1 = google.gelu %0 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    // CHECK: google.sigmoid
    %2 = google.sigmoid %1 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    // CHECK: google.tanh
    %3 = google.tanh %2 : tensor<128x256xf32> -> tensor<128x256xf32>
    
    // Dummy result for softmax test
    %4 = google.constant {value = dense<1.0> : tensor<32x1000xf32>} : tensor<32x1000xf32>
    
    // CHECK: google.softmax
    %5 = google.softmax %4 {axis = 1 : i64} : tensor<32x1000xf32> -> tensor<32x1000xf32>
    
    return %5 : tensor<32x1000xf32>
  }

  // CHECK-LABEL: func.func @test_shape_ops
  func.func @test_shape_ops(%arg0: tensor<32x12x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x768x128xf32> {
    // CHECK: google.reshape
    %0 = google.reshape %arg0 {shape = [32, 768]} : tensor<32x12x64xf32> -> tensor<32x768xf32>
    
    // CHECK: google.concat
    %1 = google.concat %arg1, %arg2 {axis = 1 : i64} : tensor<32x64xf32>, tensor<32x128xf32> -> tensor<32x192xf32>
    
    // Dummy tensor for transpose
    %2 = google.constant {value = dense<1.0> : tensor<32x768x128xf32>} : tensor<32x768x128xf32>
    
    // CHECK: google.transpose
    %3 = google.transpose %2 {perm = [0, 1, 2]} : tensor<32x768x128xf32> -> tensor<32x768x128xf32>
    
    // CHECK: google.slice
    %4 = google.slice %2 {start = [0, 0, 0], end = [32, 768, 64]} : tensor<32x768x128xf32> -> tensor<32x768x64xf32>
    
    // Dummy scalar for broadcast
    %5 = google.constant {value = dense<1.0> : tensor<768xf32>} : tensor<768xf32>
    
    // CHECK: google.broadcast
    %6 = google.broadcast %5 {shape = [32, 128, 768]} : tensor<768xf32> -> tensor<32x128x768xf32>
    
    return %3 : tensor<32x768x128xf32>
  }

  // CHECK-LABEL: func.func @test_selection_ops
  func.func @test_selection_ops(%arg0: tensor<32xf32>, %arg1: tensor<32xi1>) -> tensor<32xf32> {
    %min = google.constant {value = dense<0.0> : tensor<f32>} : tensor<f32>
    %max = google.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
    
    // CHECK: google.clamp
    %0 = google.clamp %arg0, %min, %max : tensor<32xf32>, tensor<f32>, tensor<f32> -> tensor<32xf32>
    
    %true_val = google.constant {value = dense<1.0> : tensor<32xf32>} : tensor<32xf32>
    %false_val = google.constant {value = dense<0.0> : tensor<32xf32>} : tensor<32xf32>
    
    // CHECK: google.select
    %1 = google.select %arg1, %true_val, %false_val : tensor<32xi1>, tensor<32xf32>, tensor<32xf32> -> tensor<32xf32>
    
    return %1 : tensor<32xf32>
  }
}
