// Benchmark: Element-wise Operations (Memory-Bound)
// Operations: add, mul, relu, sigmoid, tanh
// Arithmetic Intensity: ~2 FLOPS/byte (very low, memory-bound)
// Target: >320 GB/s memory bandwidth

module {
  // Test 1: Element-wise Add
  func.func @bench_add(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.add %arg0, %arg1 : tensor<4096x4096xf32>, tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 2: Element-wise Multiply
  func.func @bench_mul(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.mul %arg0, %arg1 : tensor<4096x4096xf32>, tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 3: ReLU Activation
  func.func @bench_relu(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.relu %arg0 : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 4: Sigmoid Activation
  func.func @bench_sigmoid(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.sigmoid %arg0 : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }

  // Test 5: Tanh Activation
  func.func @bench_tanh(%arg0: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = google.tanh %arg0 : tensor<4096x4096xf32> -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }
}
