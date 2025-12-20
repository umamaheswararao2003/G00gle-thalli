// Test single add operation
module {
  func.func @test_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
