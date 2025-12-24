// Test L2 Tiling with Embedded Transform
// Matrix size: 256x256
// 2-Level Tiling: L2 (64x64x64) + L1 (16x16x16)

module {
  func.func @matmul_l2_test(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = google.matmul %arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
  
  // L1+L2 Tiling Transform
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Match all linalg.matmul operations
      %matmuls = transform.structured.match ops{["linalg.matmul"]} in %arg1 
        : (!transform.any_op) -> !transform.any_op
      
      // L2 Tiling: 64x64x64 (outer loops for L2 cache)
      %tiled_l2, %loops_l2:3 = transform.structured.tile_using_for %matmuls 
        tile_sizes [64, 64, 64] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      // L1 Tiling: 16x16x16 (inner loops for L1 cache)
      %tiled_l1, %loops_l1:3 = transform.structured.tile_using_for %tiled_l2 
        tile_sizes [16, 16, 16] 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      
      transform.yield
    }
  }
}
