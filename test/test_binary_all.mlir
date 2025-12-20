// Test all binary operations
module {
  func.func @test_all_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = google.sub %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %2 = google.mul %1, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %3 = google.div %2, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %4 = google.max %3, %arg0 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %5 = google.min %4, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %5 : tensor<4xf32>
  }
}
