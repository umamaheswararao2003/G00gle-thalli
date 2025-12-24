// Simple MatMul test WITHOUT embedded transform
// This will test if the pipeline works without tiling

module {
  func.func @matmul_no_transform(%A: tensor<128x128xf32>, 
                                  %B: tensor<128x128xf32>, 
                                  %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}
