// Benchmark: Reshape (Memory-Bound)
// Operation: Tensor reshape
// Arithmetic Intensity: ~0 FLOPS/byte (pure memory operation)
// Target: >350 GB/s memory bandwidth (sequential access)

module {
  // Test 1: 2D to 1D Reshape
  func.func @bench_reshape_2d_to_1d(%arg0: tensor<4096x4096xf32>) -> tensor<16777216xf32> {
    %0 = google.reshape %arg0 : tensor<4096x4096xf32> -> tensor<16777216xf32>
    return %0 : tensor<16777216xf32>
  }

  // Test 2: 1D to 2D Reshape
  func.func @bench_reshape_1d_to_2d(%arg0: tensor<16777216xf32>) -> tensor<4096x4096xf32> {
    %0 = google.reshape %arg0 : tensor<16777216xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: 2D to 3D Reshape
  func.func @bench_reshape_2d_to_3d(%arg0: tensor<1024x1024xf32>) -> tensor<32x32x1024xf32> {
    %0 = google.reshape %arg0 : tensor<1024x1024xf32> -> tensor<32x32x1024xf32>
    return %0 : tensor<32x32x1024xf32>
  }

  // Test 4: 3D to 4D Reshape
  func.func @bench_reshape_3d_to_4d(%arg0: tensor<128x256x512xf32>) -> tensor<16x8x256x512xf32> {
    %0 = google.reshape %arg0 : tensor<128x256x512xf32> -> tensor<16x8x256x512xf32>
    return %0 : tensor<16x8x256x512xf32>
  }

  // Test 5: Complex Reshape
  func.func @bench_reshape_complex(%arg0: tensor<64x64x64x64xf32>) -> tensor<4096x4096xf32> {
    %0 = google.reshape %arg0 : tensor<64x64x64x64xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }
}
