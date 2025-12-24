// Test file for L1 tiling with embedded transform script
// This tests MatMul operation with L1 cache tiling

module {
  func.func @matmul_l1(%A: tensor<128x128xf32>, 
                       %B: tensor<128x128xf32>, 
                       %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// Transform script embedded in the same file
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // Match all linalg operations
    %ops = transform.structured.match 
      interface{LinalgOp} in %root
      : (!transform.any_op) -> !transform.any_op
    
    // L1 tiling: 16x16x16
    %tiled_l1, %loops_l1 = transform.structured.tile_using_for %ops 
      tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
