// Test all fixed binary and unary operations
module {
  // Binary operations
  func.func @test_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = google.sub %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %2 = google.mul %1, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %3 = google.div %2, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %3 : tensor<4xf32>
  }

  // Unary operations
  func.func @test_unary(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = google.neg %arg0 : tensor<8xf32> -> tensor<8xf32>
    %1 = google.abs %0 : tensor<8xf32> -> tensor<8xf32>
    %2 = google.sqrt %1 : tensor<8xf32> -> tensor<8xf32>
    %3 = google.exp %2 : tensor<8xf32> -> tensor<8xf32>
    %4 = google.log %3 : tensor<8xf32> -> tensor<8xf32>
    %5 = google.tanh %4 : tensor<8xf32> -> tensor<8xf32>
    return %5 : tensor<8xf32>
  }
}
