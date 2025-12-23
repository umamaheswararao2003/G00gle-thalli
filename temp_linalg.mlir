module {
  func.func @matmul_l2_test(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tensor.empty() : tensor<256x256xf32>
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %1 : tensor<256x256xf32>
  }
}

