// Minimal test to debug crash
module {
  func.func @test_minimal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
