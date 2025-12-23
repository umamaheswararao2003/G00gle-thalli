// Test file for L1 tiling - SIMPLIFIED
// Simple MatMul test without embedded transform (uses pipeline's interpreter)

module {
  func.func @matmul_l1(%A: tensor<128x128xf32>, 
                       %B: tensor<128x128xf32>, 
                       %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// Transform script with corrected syntax
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %ops = transform.structured.match 
      ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    
    // Tile with 16x16x16 - returns (tiled_op, loop_handle)
    %tiled, %loops = transform.structured.tile_using_for %ops 
      tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
