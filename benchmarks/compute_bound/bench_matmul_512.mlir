// Benchmark: MatMul 512x512 (Compute-Bound)
// Operation: Matrix multiplication
// Arithmetic Intensity: ~170 FLOPS/byte (highly compute-bound)
// Target: ~5,000 GFLOPS

module {
  func.func @bench_matmul_512(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32> -> tensor<512x512xf32>
    return %0 : tensor<512x512xf32>
  }
  
  // L1+L2+L3 Tiling Transform (256→64→16)
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg1 
        : (!transform.any_op) -> !transform.any_op
      
      // L3 Tiling: 256x256x256
      %tiled_l3, %loops_l3:3 = transform.structured.tile_using_for %matmuls 
        tile_sizes [256, 256, 256] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      // L2 Tiling: 64x64x64
      %tiled_l2, %loops_l2:3 = transform.structured.tile_using_for %tiled_l3 
        tile_sizes [64, 64, 64] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      // L1 Tiling: 16x16x16
      %tiled_l1, %loops_l1:3 = transform.structured.tile_using_for %tiled_l2 
        tile_sizes [16, 16, 16] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      transform.yield
    }
  }
}
