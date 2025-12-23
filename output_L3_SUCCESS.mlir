module {
  func.func @matmul_l3_test(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1024_2 = arith.constant 1024 : index
    %c1024_3 = arith.constant 1024 : index
    %c256 = arith.constant 256 : index
    %c256_4 = arith.constant 256 : index
    %c256_5 = arith.constant 256 : index
    %1 = scf.for %arg2 = %c0 to %c1024 step %c256 iter_args(%arg3 = %0) -> (tensor<1024x1024xf32>) {
      %2 = scf.for %arg4 = %c0_0 to %c1024_2 step %c256_4 iter_args(%arg5 = %arg3) -> (tensor<1024x1024xf32>) {
        %3 = scf.for %arg6 = %c0_1 to %c1024_3 step %c256_5 iter_args(%arg7 = %arg5) -> (tensor<1024x1024xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
          %extracted_slice_6 = tensor.extract_slice %arg1[%arg6, %arg4] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
          %extracted_slice_7 = tensor.extract_slice %arg7[%arg2, %arg4] [256, 256] [1, 1] : tensor<1024x1024xf32> to tensor<256x256xf32>
          %c0_8 = arith.constant 0 : index
          %c0_9 = arith.constant 0 : index
          %c0_10 = arith.constant 0 : index
          %c256_11 = arith.constant 256 : index
          %c256_12 = arith.constant 256 : index
          %c256_13 = arith.constant 256 : index
          %c64 = arith.constant 64 : index
          %c64_14 = arith.constant 64 : index
          %c64_15 = arith.constant 64 : index
          %4 = scf.for %arg8 = %c0_8 to %c256_11 step %c64 iter_args(%arg9 = %extracted_slice_7) -> (tensor<256x256xf32>) {
            %5 = scf.for %arg10 = %c0_9 to %c256_12 step %c64_14 iter_args(%arg11 = %arg9) -> (tensor<256x256xf32>) {
              %6 = scf.for %arg12 = %c0_10 to %c256_13 step %c64_15 iter_args(%arg13 = %arg11) -> (tensor<256x256xf32>) {
                %extracted_slice_16 = tensor.extract_slice %extracted_slice[%arg8, %arg12] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
                %extracted_slice_17 = tensor.extract_slice %extracted_slice_6[%arg12, %arg10] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
                %extracted_slice_18 = tensor.extract_slice %arg13[%arg8, %arg10] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
                %c0_19 = arith.constant 0 : index
                %c0_20 = arith.constant 0 : index
                %c0_21 = arith.constant 0 : index
                %c64_22 = arith.constant 64 : index
                %c64_23 = arith.constant 64 : index
                %c64_24 = arith.constant 64 : index
                %c16 = arith.constant 16 : index
                %c16_25 = arith.constant 16 : index
                %c16_26 = arith.constant 16 : index
                %7 = scf.for %arg14 = %c0_19 to %c64_22 step %c16 iter_args(%arg15 = %extracted_slice_18) -> (tensor<64x64xf32>) {
                  %8 = scf.for %arg16 = %c0_20 to %c64_23 step %c16_25 iter_args(%arg17 = %arg15) -> (tensor<64x64xf32>) {
                    %9 = scf.for %arg18 = %c0_21 to %c64_24 step %c16_26 iter_args(%arg19 = %arg17) -> (tensor<64x64xf32>) {
                      %extracted_slice_28 = tensor.extract_slice %extracted_slice_16[%arg14, %arg18] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                      %extracted_slice_29 = tensor.extract_slice %extracted_slice_17[%arg18, %arg16] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                      %extracted_slice_30 = tensor.extract_slice %arg19[%arg14, %arg16] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
                      %10 = linalg.matmul ins(%extracted_slice_28, %extracted_slice_29 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_30 : tensor<16x16xf32>) -> tensor<16x16xf32>
                      %inserted_slice_31 = tensor.insert_slice %10 into %arg19[%arg14, %arg16] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
                      scf.yield %inserted_slice_31 : tensor<64x64xf32>
                    }
                    scf.yield %9 : tensor<64x64xf32>
                  }
                  scf.yield %8 : tensor<64x64xf32>
                }
                %inserted_slice_27 = tensor.insert_slice %7 into %arg13[%arg8, %arg10] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
                scf.yield %inserted_slice_27 : tensor<256x256xf32>
              }
              scf.yield %6 : tensor<256x256xf32>
            }
            scf.yield %5 : tensor<256x256xf32>
          }
          %inserted_slice = tensor.insert_slice %4 into %arg7[%arg2, %arg4] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<1024x1024xf32>
          scf.yield %inserted_slice : tensor<1024x1024xf32>
        }
        scf.yield %3 : tensor<1024x1024xf32>
      }
      scf.yield %2 : tensor<1024x1024xf32>
    }
    return %1 : tensor<1024x1024xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [256, 256, 256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_linalg_op_0, %loops_1:3 = transform.structured.tile_using_for %tiled_linalg_op tile_sizes [64, 64, 64] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_linalg_op_2, %loops_3:3 = transform.structured.tile_using_for %tiled_linalg_op_0 tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

