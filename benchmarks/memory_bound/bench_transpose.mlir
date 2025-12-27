// Benchmark: Transpose (Memory-Bound)
// Operation: Matrix transpose
// Arithmetic Intensity: ~0 FLOPS/byte (pure memory operation)
// Target: >300 GB/s memory bandwidth (with coalescing)

module {
  // Test 1: Square Matrix Transpose (1024x1024)
  func.func @bench_transpose_1024(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = google.transpose %arg0 {permutation = [1, 0]} : tensor<1024x1024xf32> -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }

  // Test 2: Large Square Matrix Transpose (4096x4096)
  func.func @bench_transpose_4096(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.transpose %arg0 {permutation = [1, 0]} : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: Rectangular Matrix Transpose
  func.func @bench_transpose_rect(%arg0: tensor<2048x8192xf32>) -> tensor<8192x2048xf32> {
    %0 = google.transpose %arg0 {permutation = [1, 0]} : tensor<2048x8192xf32> -> tensor<8192x2048xf32>
    return %0 : tensor<8192x2048xf32>
  }

  // Test 4: 3D Tensor Transpose
  func.func @bench_transpose_3d(%arg0: tensor<128x256x512xf32>) -> tensor<512x256x128xf32> {
    %0 = google.transpose %arg0 {permutation = [2, 1, 0]} : tensor<128x256x512xf32> -> tensor<512x256x128xf32>
    return %0 : tensor<512x256x128xf32>
  }
}
