// L1 Cache Tiling Transform Script - CORRECT SYNTAX
// Based on MLIR's transform-op-tile.mlir test file
// Tile size: 16x16x16 for L1 cache (32KB)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // Match all linalg.matmul operations
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    
    // L1 tiling: 16x16x16
    // CORRECT SYNTAX: %loops:3 unpacks the 3 loop results (i, j, k)
    %tiled, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
