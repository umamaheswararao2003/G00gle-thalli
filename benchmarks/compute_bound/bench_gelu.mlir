// Benchmark: GELU Activation (Compute-Bound)
// Operation: Gaussian Error Linear Unit
// Arithmetic Intensity: ~15 FLOPS/byte (moderately compute-bound)
// Target: ~3,500 GFLOPS

module {
  // Test 1: GELU on 2D tensor
  func.func @bench_gelu_2d(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = google.gelu %arg0 : tensor<1024x1024xf32> -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }

  // Test 2: GELU on large 2D tensor
  func.func @bench_gelu_large(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.gelu %arg0 : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: GELU on 3D tensor (transformer feedforward)
  func.func @bench_gelu_3d(%arg0: tensor<128x512x2048xf32>) -> tensor<128x512x2048xf32> {
    %0 = google.gelu %arg0 : tensor<128x512x2048xf32> -> tensor<128x512x2048xf32>
    return %0 : tensor<128x512x2048xf32>
  }

  // Test 4: GELU on very large tensor
  func.func @bench_gelu_xlarge(%arg0: tensor<8192x8192xf32>) -> tensor<8192x8192xf32> {
    %0 = google.gelu %arg0 : tensor<8192x8192xf32> -> tensor<8192x8192xf32>
    return %0 : tensor<8192x8192xf32>
  }
}
