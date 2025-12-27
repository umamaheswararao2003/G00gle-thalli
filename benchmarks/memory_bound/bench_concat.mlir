// Benchmark: Concat (Memory-Bound)
// Operation: Tensor concatenation
// Arithmetic Intensity: ~0 FLOPS/byte (pure memory copy)
// Target: >300 GB/s memory bandwidth

module {
  // Test 1: Concat along axis 0 (rows)
  func.func @bench_concat_axis0(%arg0: tensor<2048x4096xf32>, %arg1: tensor<2048x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 0 : i64} : tensor<2048x4096xf32>, tensor<2048x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 2: Concat along axis 1 (columns)
  func.func @bench_concat_axis1(%arg0: tensor<4096x2048xf32>, %arg1: tensor<4096x2048xf32>) -> tensor<4096x4096xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 1 : i64} : tensor<4096x2048xf32>, tensor<4096x2048xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: Multiple Concat (4 tensors)
  func.func @bench_concat_multiple(%arg0: tensor<1024x4096xf32>, 
                                    %arg1: tensor<1024x4096xf32>,
                                    %arg2: tensor<1024x4096xf32>,
                                    %arg3: tensor<1024x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 0 : i64} : tensor<1024x4096xf32>, tensor<1024x4096xf32> -> tensor<2048x4096xf32>
    %1 = google.concat %arg2, %arg3 {axis = 0 : i64} : tensor<1024x4096xf32>, tensor<1024x4096xf32> -> tensor<2048x4096xf32>
    %2 = google.concat %0, %1 {axis = 0 : i64} : tensor<2048x4096xf32>, tensor<2048x4096xf32> -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
  }

  // Test 4: 3D Tensor Concat
  func.func @bench_concat_3d(%arg0: tensor<64x256x512xf32>, %arg1: tensor<64x256x512xf32>) -> tensor<128x256x512xf32> {
    %0 = google.concat %arg0, %arg1 {axis = 0 : i64} : tensor<64x256x512xf32>, tensor<64x256x512xf32> -> tensor<128x256x512xf32>
    return %0 : tensor<128x256x512xf32>
  }
}
