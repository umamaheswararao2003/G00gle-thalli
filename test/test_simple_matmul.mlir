// Simplified test - just matmul, no complex types
module {
  func.func @simple_matmul(%arg0: tensor<64x64xf32>, 
                           %arg1: tensor<64x64xf32>, 
                           %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                       outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 
      : (!transform.any_op) -> !transform.any_op
    
    %tiled, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    
    transform.yield
  }
}
