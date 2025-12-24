// Test file for L1 tiling - CORRECT SYNTAX
// Based on MLIR's own test examples

module {
  func.func @matmul_l1(%arg0: tensor<128x128xf32>, 
                       %arg1: tensor<128x128xf32>, 
                       %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                       outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}

// Transform script with CORRECT syntax
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    
    // Tile with 16x16x16 - CORRECT: %loops:3 unpacks 3 loops
    %tiled, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
