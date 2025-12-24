// Test only add operations
module {
  func.func @test_add_only(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    %1 = google.add %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
