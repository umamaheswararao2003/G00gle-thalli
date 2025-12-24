// Final comprehensive test for GoogleToLinalg lowering
module {
  // Binary operations
  func.func @test_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = google.sub %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %2 = google.mul %1, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %3 = google.div %2, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %4 = google.max %3, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %5 = google.min %4, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %5 : tensor<4xf32>
  }

  // Unary operations
  func.func @test_unary(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.neg %arg0 : tensor<8xf32> -> tensor<8xf32>
    %1 = google.abs %0 : tensor<8xf32> -> tensor<8xf32>
    %2 = google.sqrt %1 : tensor<8xf32> -> tensor<8xf32>
    %3 = google.exp %2 : tensor<8xf32> -> tensor<8xf32>
    %4 = google.log %3 : tensor<8xf32> -> tensor<8xf32>
    return %4 : tensor<8xf32>
  }

  // Activations
  func.func @test_activations(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = google.relu %arg0 : tensor<16xf32> -> tensor<16xf32>
    %1 = google.tanh %0 : tensor<16xf32> -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }

  // MatMul
  func.func @test_matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32> -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }

  // Transpose
  func.func @test_transpose(%arg0: tensor<4x8x16xf32>) -> tensor<16x4x8xf32> {
    %0 = google.transpose %arg0 {perm = [2, 0, 1]} : tensor<4x8x16xf32> -> tensor<16x4x8xf32>
    return %0 : tensor<16x4x8xf32>
  }

  // Selection
  func.func @test_select(%cond: tensor<4xi1>, %true_val: tensor<4xf32>, %false_val: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.select %cond, %true_val, %false_val : tensor<4xi1>, tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // Clamp
  func.func @test_clamp(%input: tensor<8xf32>, %min: tensor<8xf32>, %max: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.clamp %input, %min, %max : tensor<8xf32>, tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
