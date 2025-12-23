module {
  func.func @matmul_l2_test(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = tensor.empty() : tensor<256x256xf32>
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c256_2 = arith.constant 256 : index
    %c256_3 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %c64_5 = arith.constant 64 : index
    %1 = scf.for %arg2 = %c0 to %c256 step %c64 iter_args(%arg3 = %0) -> (tensor<256x256xf32>) {
      %2 = scf.for %arg4 = %c0_0 to %c256_2 step %c64_4 iter_args(%arg5 = %arg3) -> (tensor<256x256xf32>) {
        %3 = scf.for %arg6 = %c0_1 to %c256_3 step %c64_5 iter_args(%arg7 = %arg5) -> (tensor<256x256xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %extracted_slice_6 = tensor.extract_slice %arg1[%arg6, %arg4] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %extracted_slice_7 = tensor.extract_slice %arg7[%arg2, %arg4] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %c0_8 = arith.constant 0 : index
          %c0_9 = arith.constant 0 : index
          %c0_10 = arith.constant 0 : index
          %c64_11 = arith.constant 64 : index
          %c64_12 = arith.constant 64 : index
          %c64_13 = arith.constant 64 : index
          %c16 = arith.constant 16 : index
          %c16_14 = arith.constant 16 : index
          %c16_15 = arith.constant 16 : index
          %4 = scf.for %arg8 = %c0_8 to %c64_11 step %c16 iter_args(%arg9 = %extracted_slice_7) -> (tensor<64x64xf32>) {
            %5 = scf.for %arg10 = %c0_9 to %c64_12 step %c16_14 iter_args(%arg11 = %arg9) -> (tensor<64x64xf32>) {
              %6 = scf.for %arg12 = %c0_10 to %c64_13 step %c16_15 iter_args(%arg13 = %arg11) -> (tensor<64x64xf32>) {
                %extracted_slice_16 = tensor.extract_slice %extracted_slice[%arg8, %arg12] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                %extracted_slice_17 = tensor.extract_slice %extracted_slice_6[%arg12, %arg10] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                %extracted_slice_18 = tensor.extract_slice %arg13[%arg8, %arg10] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                %7 = linalg.matmul ins(%extracted_slice_16, %extracted_slice_17 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_18 : tensor<16x16xf32>) -> tensor<16x16xf32>
                %inserted_slice_19 = tensor.insert_slice %7 into %arg13[%arg8, %arg10] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
                scf.yield %inserted_slice_19 : tensor<64x64xf32>
              }
              scf.yield %6 : tensor<64x64xf32>
            }
            scf.yield %5 : tensor<64x64xf32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg7[%arg2, %arg4] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
          scf.yield %inserted_slice : tensor<256x256xf32>
        }
        scf.yield %3 : tensor<256x256xf32>
      }
      scf.yield %2 : tensor<256x256xf32>
    }
    return %1 : tensor<256x256xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [64, 64, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_linalg_op_0, %loops_1:3 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

