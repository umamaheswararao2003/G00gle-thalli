// Benchmark: Broadcast (Memory-Bound)
// Operation: Tensor broadcasting
// Arithmetic Intensity: ~0 FLOPS/byte (memory write-heavy)
// Target: >320 GB/s memory bandwidth

module {
  // Test 1: Broadcast vector to matrix
  func.func @bench_broadcast_vec_to_mat(%arg0: tensor<4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.broadcast %arg0 {dimensions = [4096, 4096]} : tensor<4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 2: Broadcast scalar to matrix
  func.func @bench_broadcast_scalar_to_mat(%arg0: tensor<f32>) -> tensor<4096x4096xf32> {
    %0 = google.broadcast %arg0 {dimensions = [4096, 4096]} : tensor<f32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: Broadcast 2D to 3D
  func.func @bench_broadcast_2d_to_3d(%arg0: tensor<256x512xf32>) -> tensor<128x256x512xf32> {
    %0 = google.broadcast %arg0 {dimensions = [128, 256, 512]} : tensor<256x512xf32> -> tensor<128x256x512xf32>
    return %0 : tensor<128x256x512xf32>
  }

  // Test 4: Broadcast 1D to 3D
  func.func @bench_broadcast_1d_to_3d(%arg0: tensor<512xf32>) -> tensor<128x256x512xf32> {
    %0 = google.broadcast %arg0 {dimensions = [128, 256, 512]} : tensor<512xf32> -> tensor<128x256x512xf32>
    return %0 : tensor<128x256x512xf32>
  }

  // Test 5: Large Broadcast
  func.func @bench_broadcast_large(%arg0: tensor<1xf32>) -> tensor<8192x8192xf32> {
    %0 = google.broadcast %arg0 {dimensions = [8192, 8192]} : tensor<1xf32> -> tensor<8192x8192xf32>
    return %0 : tensor<8192x8192xf32>
  }
}
