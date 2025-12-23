// Test file for L1 tiling - Alternative syntax
// Testing without capturing loop results

module {
  func.func @matmul_l1(%A: tensor<128x128xf32>, 
                       %B: tensor<128x128xf32>, 
                       %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %result = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// Transform script - try not capturing loops
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %ops = transform.structured.match 
      ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    
    // Try with interchange attribute instead
    %tiled:4 = transform.structured.tile_using_for %ops 
      tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
