// Test file for L1 tiling with MatMul+Bias+ReLU (fused operations)
// This tests that tiling works correctly on fused operations

module {
  func.func @matmul_bias_relu_l1(%A: tensor<128x128xf32>, 
                                  %B: tensor<128x128xf32>,
                                  %bias: tensor<128xf32>,
                                  %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    
    // MatMul
    %matmul = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                            outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    
    // Broadcast bias
    %bias_2d = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias : tensor<128xf32>) outs(%C : tensor<128x128xf32>) {
    ^bb0(%b: f32, %out: f32):
      linalg.yield %b : f32
    } -> tensor<128x128xf32>
    
    // Add
    %add = linalg.add ins(%matmul, %bias_2d : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
    
    // ReLU
    %relu = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%add : tensor<128x128xf32>) outs(%C : tensor<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
    } -> tensor<128x128xf32>
    
    return %relu : tensor<128x128xf32>
  }
}

// Transform script - will tile fused operations
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
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
