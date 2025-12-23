// Test L2 Tiling with larger matrices
// Matrix size: 256x256 (to see L2 tiling benefits)

module {
  func.func @matmul_l2_test(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
