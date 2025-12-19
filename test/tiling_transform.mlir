func.func @test_tiling(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = google.dequant %arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["google.dequant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loop_x, %loop_y = transform.structured.tile_using_for %0 tile_sizes [16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
