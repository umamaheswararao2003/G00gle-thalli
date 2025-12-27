// Benchmark: Softmax (Compute-Bound)
// Operation: Softmax normalization
// Arithmetic Intensity: ~20 FLOPS/byte (moderately compute-bound)
// Target: ~4,000 GFLOPS

module {
  // Test 1: Softmax on 2D tensor (batch processing)
  func.func @bench_softmax_2d(%arg0: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = google.softmax %arg0 {axis = 1 : i64} : tensor<1024x1024xf32> -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }

  // Test 2: Softmax on large 2D tensor
  func.func @bench_softmax_large(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.softmax %arg0 {axis = 1 : i64} : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: Softmax on 3D tensor (attention mechanism)
  func.func @bench_softmax_3d(%arg0: tensor<32x128x128xf32>) -> tensor<32x128x128xf32> {
    %0 = google.softmax %arg0 {axis = 2 : i64} : tensor<32x128x128xf32> -> tensor<32x128x128xf32>
    return %0 : tensor<32x128x128xf32>
  }
}
