module {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @linear_layer_f32(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %7 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %8 = builtin.unrealized_conversion_cast %7 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(32 : index) : i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(2048 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.sub %20, %24 : i64
    %26 = llvm.add %23, %25 : i64
    %27 = llvm.urem %26, %20 : i64
    %28 = llvm.sub %26, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %22, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%39: index):  // 2 preds: ^bb0, ^bb8
    %40 = builtin.unrealized_conversion_cast %39 : index to i64
    %41 = builtin.unrealized_conversion_cast %39 : index to i64
    %42 = llvm.icmp "slt" %41, %3 : i64
    cf.cond_br %42, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%43: index):  // 2 preds: ^bb2, ^bb7
    %44 = builtin.unrealized_conversion_cast %43 : index to i64
    %45 = builtin.unrealized_conversion_cast %43 : index to i64
    %46 = llvm.icmp "slt" %45, %1 : i64
    cf.cond_br %46, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%47: index):  // 2 preds: ^bb4, ^bb6
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %49, %0 : i64
    cf.cond_br %50, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %40, %54 overflow<nsw, nuw> : i64
    %56 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %48, %56 overflow<nsw, nuw> : i64
    %58 = llvm.add %55, %57 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %48, %64 overflow<nsw, nuw> : i64
    %66 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %44, %66 overflow<nsw, nuw> : i64
    %68 = llvm.add %65, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds|nuw %63[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(64 : index) : i64
    %73 = llvm.mul %40, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %44 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %60, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(64 : index) : i64
    %81 = llvm.mul %40, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %44 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %79[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %83 : f32, !llvm.ptr
    %84 = llvm.add %49, %2 : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    cf.br ^bb5(%85 : index)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %45, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb3(%87 : index)
  ^bb8:  // pred: ^bb3
    %88 = llvm.add %41, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb1(%89 : index)
  ^bb9:  // pred: ^bb1
    %90 = llvm.mlir.constant(32 : index) : i64
    %91 = llvm.mlir.constant(64 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(2048 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.constant(64 : index) : i64
    %98 = llvm.add %96, %97 : i64
    %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.sub %97, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.urem %103, %97 : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.insertvalue %99, %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %91, %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %92, %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%116: index):  // 2 preds: ^bb9, ^bb14
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    %118 = builtin.unrealized_conversion_cast %116 : index to i64
    %119 = llvm.icmp "slt" %118, %3 : i64
    cf.cond_br %119, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%120: index):  // 2 preds: ^bb11, ^bb13
    %121 = builtin.unrealized_conversion_cast %120 : index to i64
    %122 = builtin.unrealized_conversion_cast %120 : index to i64
    %123 = llvm.icmp "slt" %122, %1 : i64
    cf.cond_br %123, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %124 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mul %121, %127 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(64 : index) : i64
    %133 = llvm.mul %117, %132 overflow<nsw, nuw> : i64
    %134 = llvm.add %133, %121 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %130, %135 : f32, !llvm.ptr
    %136 = llvm.add %122, %2 : i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    cf.br ^bb12(%137 : index)
  ^bb14:  // pred: ^bb12
    %138 = llvm.add %118, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb10(%139 : index)
  ^bb15:  // pred: ^bb10
    %140 = llvm.mlir.constant(32 : index) : i64
    %141 = llvm.mlir.constant(64 : index) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(2048 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(64 : index) : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.sub %147, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.urem %153, %147 : i64
    %155 = llvm.sub %153, %154 : i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %149, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%167: index):  // 2 preds: ^bb15, ^bb20
    %168 = builtin.unrealized_conversion_cast %167 : index to i64
    %169 = builtin.unrealized_conversion_cast %167 : index to i64
    %170 = llvm.icmp "slt" %169, %3 : i64
    cf.cond_br %170, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%171: index):  // 2 preds: ^bb17, ^bb19
    %172 = builtin.unrealized_conversion_cast %171 : index to i64
    %173 = builtin.unrealized_conversion_cast %171 : index to i64
    %174 = llvm.icmp "slt" %173, %1 : i64
    cf.cond_br %174, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %175 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(64 : index) : i64
    %177 = llvm.mul %168, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %172 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(64 : index) : i64
    %183 = llvm.mul %168, %182 overflow<nsw, nuw> : i64
    %184 = llvm.add %183, %172 overflow<nsw, nuw> : i64
    %185 = llvm.getelementptr inbounds|nuw %181[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %186 = llvm.load %185 : !llvm.ptr -> f32
    %187 = llvm.fadd %180, %186 : f32
    %188 = llvm.intr.maximum(%187, %6) : (f32, f32) -> f32
    %189 = llvm.extractvalue %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(64 : index) : i64
    %191 = llvm.mul %168, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %172 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %193 : f32, !llvm.ptr
    %194 = llvm.add %173, %2 : i64
    %195 = builtin.unrealized_conversion_cast %194 : i64 to index
    cf.br ^bb18(%195 : index)
  ^bb20:  // pred: ^bb18
    %196 = llvm.add %169, %2 : i64
    %197 = builtin.unrealized_conversion_cast %196 : i64 to index
    cf.br ^bb16(%197 : index)
  ^bb21:  // pred: ^bb16
    %198 = bufferization.to_tensor %166 : memref<32x64xf32> to tensor<32x64xf32>
    return %198 : tensor<32x64xf32>
  }
  func.func @large_linear_layer_f32(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256xf32>) -> tensor<256x256xf32> {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(256 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = bufferization.to_buffer %arg2 : tensor<256xf32> to memref<256xf32, strided<[?], offset: ?>>
    %7 = builtin.unrealized_conversion_cast %6 : memref<256xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = bufferization.to_buffer %arg1 : tensor<512x256xf32> to memref<512x256xf32, strided<[?, ?], offset: ?>>
    %9 = builtin.unrealized_conversion_cast %8 : memref<512x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = bufferization.to_buffer %arg0 : tensor<256x512xf32> to memref<256x512xf32, strided<[?, ?], offset: ?>>
    %11 = builtin.unrealized_conversion_cast %10 : memref<256x512xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(256 : index) : i64
    %13 = llvm.mlir.constant(256 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(65536 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.add %18, %19 : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.sub %19, %23 : i64
    %25 = llvm.add %22, %24 : i64
    %26 = llvm.urem %25, %19 : i64
    %27 = llvm.sub %25, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %29 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %21, %29[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.insertvalue %32, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %12, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %13, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%4 : index)
  ^bb1(%38: index):  // 2 preds: ^bb0, ^bb8
    %39 = builtin.unrealized_conversion_cast %38 : index to i64
    %40 = builtin.unrealized_conversion_cast %38 : index to i64
    %41 = llvm.icmp "slt" %40, %2 : i64
    cf.cond_br %41, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%4 : index)
  ^bb3(%42: index):  // 2 preds: ^bb2, ^bb7
    %43 = builtin.unrealized_conversion_cast %42 : index to i64
    %44 = builtin.unrealized_conversion_cast %42 : index to i64
    %45 = llvm.icmp "slt" %44, %2 : i64
    cf.cond_br %45, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%4 : index)
  ^bb5(%46: index):  // 2 preds: ^bb4, ^bb6
    %47 = builtin.unrealized_conversion_cast %46 : index to i64
    %48 = builtin.unrealized_conversion_cast %46 : index to i64
    %49 = llvm.icmp "slt" %48, %0 : i64
    cf.cond_br %49, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %50 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.getelementptr %50[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mul %39, %53 overflow<nsw, nuw> : i64
    %55 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %47, %55 overflow<nsw, nuw> : i64
    %57 = llvm.add %54, %56 overflow<nsw, nuw> : i64
    %58 = llvm.getelementptr inbounds|nuw %52[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.load %58 : !llvm.ptr -> f32
    %60 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %47, %63 overflow<nsw, nuw> : i64
    %65 = llvm.extractvalue %9[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %43, %65 overflow<nsw, nuw> : i64
    %67 = llvm.add %64, %66 overflow<nsw, nuw> : i64
    %68 = llvm.getelementptr inbounds|nuw %62[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(256 : index) : i64
    %72 = llvm.mul %39, %71 overflow<nsw, nuw> : i64
    %73 = llvm.add %72, %43 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds|nuw %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 : !llvm.ptr -> f32
    %76 = llvm.fmul %59, %69 : f32
    %77 = llvm.fadd %75, %76 : f32
    %78 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(256 : index) : i64
    %80 = llvm.mul %39, %79 overflow<nsw, nuw> : i64
    %81 = llvm.add %80, %43 overflow<nsw, nuw> : i64
    %82 = llvm.getelementptr inbounds|nuw %78[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %82 : f32, !llvm.ptr
    %83 = llvm.add %48, %1 : i64
    %84 = builtin.unrealized_conversion_cast %83 : i64 to index
    cf.br ^bb5(%84 : index)
  ^bb7:  // pred: ^bb5
    %85 = llvm.add %44, %1 : i64
    %86 = builtin.unrealized_conversion_cast %85 : i64 to index
    cf.br ^bb3(%86 : index)
  ^bb8:  // pred: ^bb3
    %87 = llvm.add %40, %1 : i64
    %88 = builtin.unrealized_conversion_cast %87 : i64 to index
    cf.br ^bb1(%88 : index)
  ^bb9:  // pred: ^bb1
    %89 = llvm.mlir.constant(256 : index) : i64
    %90 = llvm.mlir.constant(256 : index) : i64
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.mlir.constant(65536 : index) : i64
    %93 = llvm.mlir.zero : !llvm.ptr
    %94 = llvm.getelementptr %93[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.mlir.constant(64 : index) : i64
    %97 = llvm.add %95, %96 : i64
    %98 = llvm.call @malloc(%97) : (i64) -> !llvm.ptr
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.sub %96, %100 : i64
    %102 = llvm.add %99, %101 : i64
    %103 = llvm.urem %102, %96 : i64
    %104 = llvm.sub %102, %103 : i64
    %105 = llvm.inttoptr %104 : i64 to !llvm.ptr
    %106 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %107 = llvm.insertvalue %98, %106[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %105, %107[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.mlir.constant(0 : index) : i64
    %110 = llvm.insertvalue %109, %108[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %89, %110[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %90, %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%4 : index)
  ^bb10(%115: index):  // 2 preds: ^bb9, ^bb14
    %116 = builtin.unrealized_conversion_cast %115 : index to i64
    %117 = builtin.unrealized_conversion_cast %115 : index to i64
    %118 = llvm.icmp "slt" %117, %2 : i64
    cf.cond_br %118, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%4 : index)
  ^bb12(%119: index):  // 2 preds: ^bb11, ^bb13
    %120 = builtin.unrealized_conversion_cast %119 : index to i64
    %121 = builtin.unrealized_conversion_cast %119 : index to i64
    %122 = llvm.icmp "slt" %121, %2 : i64
    cf.cond_br %122, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %123 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.getelementptr %123[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.mul %120, %126 overflow<nsw, nuw> : i64
    %128 = llvm.getelementptr inbounds|nuw %125[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
    %130 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(256 : index) : i64
    %132 = llvm.mul %116, %131 overflow<nsw, nuw> : i64
    %133 = llvm.add %132, %120 overflow<nsw, nuw> : i64
    %134 = llvm.getelementptr inbounds|nuw %130[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %129, %134 : f32, !llvm.ptr
    %135 = llvm.add %121, %1 : i64
    %136 = builtin.unrealized_conversion_cast %135 : i64 to index
    cf.br ^bb12(%136 : index)
  ^bb14:  // pred: ^bb12
    %137 = llvm.add %117, %1 : i64
    %138 = builtin.unrealized_conversion_cast %137 : i64 to index
    cf.br ^bb10(%138 : index)
  ^bb15:  // pred: ^bb10
    %139 = llvm.mlir.constant(256 : index) : i64
    %140 = llvm.mlir.constant(256 : index) : i64
    %141 = llvm.mlir.constant(1 : index) : i64
    %142 = llvm.mlir.constant(65536 : index) : i64
    %143 = llvm.mlir.zero : !llvm.ptr
    %144 = llvm.getelementptr %143[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.ptrtoint %144 : !llvm.ptr to i64
    %146 = llvm.mlir.constant(64 : index) : i64
    %147 = llvm.add %145, %146 : i64
    %148 = llvm.call @malloc(%147) : (i64) -> !llvm.ptr
    %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
    %150 = llvm.mlir.constant(1 : index) : i64
    %151 = llvm.sub %146, %150 : i64
    %152 = llvm.add %149, %151 : i64
    %153 = llvm.urem %152, %146 : i64
    %154 = llvm.sub %152, %153 : i64
    %155 = llvm.inttoptr %154 : i64 to !llvm.ptr
    %156 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %157 = llvm.insertvalue %148, %156[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.mlir.constant(0 : index) : i64
    %160 = llvm.insertvalue %159, %158[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %139, %160[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %140, %162[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = builtin.unrealized_conversion_cast %164 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<256x256xf32>
    cf.br ^bb16(%4 : index)
  ^bb16(%166: index):  // 2 preds: ^bb15, ^bb20
    %167 = builtin.unrealized_conversion_cast %166 : index to i64
    %168 = builtin.unrealized_conversion_cast %166 : index to i64
    %169 = llvm.icmp "slt" %168, %2 : i64
    cf.cond_br %169, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%4 : index)
  ^bb18(%170: index):  // 2 preds: ^bb17, ^bb19
    %171 = builtin.unrealized_conversion_cast %170 : index to i64
    %172 = builtin.unrealized_conversion_cast %170 : index to i64
    %173 = llvm.icmp "slt" %172, %2 : i64
    cf.cond_br %173, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %174 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.mlir.constant(256 : index) : i64
    %176 = llvm.mul %167, %175 overflow<nsw, nuw> : i64
    %177 = llvm.add %176, %171 overflow<nsw, nuw> : i64
    %178 = llvm.getelementptr inbounds|nuw %174[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %179 = llvm.load %178 : !llvm.ptr -> f32
    %180 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.mlir.constant(256 : index) : i64
    %182 = llvm.mul %167, %181 overflow<nsw, nuw> : i64
    %183 = llvm.add %182, %171 overflow<nsw, nuw> : i64
    %184 = llvm.getelementptr inbounds|nuw %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %185 = llvm.load %184 : !llvm.ptr -> f32
    %186 = llvm.fadd %179, %185 : f32
    %187 = llvm.intr.maximum(%186, %5) : (f32, f32) -> f32
    %188 = llvm.extractvalue %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %189 = llvm.mlir.constant(256 : index) : i64
    %190 = llvm.mul %167, %189 overflow<nsw, nuw> : i64
    %191 = llvm.add %190, %171 overflow<nsw, nuw> : i64
    %192 = llvm.getelementptr inbounds|nuw %188[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %187, %192 : f32, !llvm.ptr
    %193 = llvm.add %172, %1 : i64
    %194 = builtin.unrealized_conversion_cast %193 : i64 to index
    cf.br ^bb18(%194 : index)
  ^bb20:  // pred: ^bb18
    %195 = llvm.add %168, %1 : i64
    %196 = builtin.unrealized_conversion_cast %195 : i64 to index
    cf.br ^bb16(%196 : index)
  ^bb21:  // pred: ^bb16
    %197 = bufferization.to_tensor %165 : memref<256x256xf32> to tensor<256x256xf32>
    return %197 : tensor<256x256xf32>
  }
  func.func @small_batch_linear_f32(%arg0: tensor<1x784xf32>, %arg1: tensor<784x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = llvm.mlir.constant(784 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = builtin.unrealized_conversion_cast %3 : i64 to index
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = bufferization.to_buffer %arg2 : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
    %7 = builtin.unrealized_conversion_cast %6 : memref<128xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = bufferization.to_buffer %arg1 : tensor<784x128xf32> to memref<784x128xf32, strided<[?, ?], offset: ?>>
    %9 = builtin.unrealized_conversion_cast %8 : memref<784x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = bufferization.to_buffer %arg0 : tensor<1x784xf32> to memref<1x784xf32, strided<[?, ?], offset: ?>>
    %11 = builtin.unrealized_conversion_cast %10 : memref<1x784xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(128 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(128 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.add %18, %19 : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.sub %19, %23 : i64
    %25 = llvm.add %22, %24 : i64
    %26 = llvm.urem %25, %19 : i64
    %27 = llvm.sub %25, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %29 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %21, %29[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.insertvalue %32, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %12, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %13, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%4 : index)
  ^bb1(%38: index):  // 2 preds: ^bb0, ^bb8
    %39 = builtin.unrealized_conversion_cast %38 : index to i64
    %40 = builtin.unrealized_conversion_cast %38 : index to i64
    %41 = llvm.icmp "slt" %40, %2 : i64
    cf.cond_br %41, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%4 : index)
  ^bb3(%42: index):  // 2 preds: ^bb2, ^bb7
    %43 = builtin.unrealized_conversion_cast %42 : index to i64
    %44 = builtin.unrealized_conversion_cast %42 : index to i64
    %45 = llvm.icmp "slt" %44, %1 : i64
    cf.cond_br %45, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%4 : index)
  ^bb5(%46: index):  // 2 preds: ^bb4, ^bb6
    %47 = builtin.unrealized_conversion_cast %46 : index to i64
    %48 = builtin.unrealized_conversion_cast %46 : index to i64
    %49 = llvm.icmp "slt" %48, %0 : i64
    cf.cond_br %49, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %50 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.getelementptr %50[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mul %39, %53 overflow<nsw, nuw> : i64
    %55 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %47, %55 overflow<nsw, nuw> : i64
    %57 = llvm.add %54, %56 overflow<nsw, nuw> : i64
    %58 = llvm.getelementptr inbounds|nuw %52[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.load %58 : !llvm.ptr -> f32
    %60 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %47, %63 overflow<nsw, nuw> : i64
    %65 = llvm.extractvalue %9[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %43, %65 overflow<nsw, nuw> : i64
    %67 = llvm.add %64, %66 overflow<nsw, nuw> : i64
    %68 = llvm.getelementptr inbounds|nuw %62[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(128 : index) : i64
    %72 = llvm.mul %39, %71 overflow<nsw, nuw> : i64
    %73 = llvm.add %72, %43 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds|nuw %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 : !llvm.ptr -> f32
    %76 = llvm.fmul %59, %69 : f32
    %77 = llvm.fadd %75, %76 : f32
    %78 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(128 : index) : i64
    %80 = llvm.mul %39, %79 overflow<nsw, nuw> : i64
    %81 = llvm.add %80, %43 overflow<nsw, nuw> : i64
    %82 = llvm.getelementptr inbounds|nuw %78[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %82 : f32, !llvm.ptr
    %83 = llvm.add %48, %2 : i64
    %84 = builtin.unrealized_conversion_cast %83 : i64 to index
    cf.br ^bb5(%84 : index)
  ^bb7:  // pred: ^bb5
    %85 = llvm.add %44, %2 : i64
    %86 = builtin.unrealized_conversion_cast %85 : i64 to index
    cf.br ^bb3(%86 : index)
  ^bb8:  // pred: ^bb3
    %87 = llvm.add %40, %2 : i64
    %88 = builtin.unrealized_conversion_cast %87 : i64 to index
    cf.br ^bb1(%88 : index)
  ^bb9:  // pred: ^bb1
    %89 = llvm.mlir.constant(1 : index) : i64
    %90 = llvm.mlir.constant(128 : index) : i64
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.mlir.constant(128 : index) : i64
    %93 = llvm.mlir.zero : !llvm.ptr
    %94 = llvm.getelementptr %93[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.mlir.constant(64 : index) : i64
    %97 = llvm.add %95, %96 : i64
    %98 = llvm.call @malloc(%97) : (i64) -> !llvm.ptr
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.sub %96, %100 : i64
    %102 = llvm.add %99, %101 : i64
    %103 = llvm.urem %102, %96 : i64
    %104 = llvm.sub %102, %103 : i64
    %105 = llvm.inttoptr %104 : i64 to !llvm.ptr
    %106 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %107 = llvm.insertvalue %98, %106[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %105, %107[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.mlir.constant(0 : index) : i64
    %110 = llvm.insertvalue %109, %108[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %89, %110[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %90, %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%4 : index)
  ^bb10(%115: index):  // 2 preds: ^bb9, ^bb14
    %116 = builtin.unrealized_conversion_cast %115 : index to i64
    %117 = builtin.unrealized_conversion_cast %115 : index to i64
    %118 = llvm.icmp "slt" %117, %2 : i64
    cf.cond_br %118, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%4 : index)
  ^bb12(%119: index):  // 2 preds: ^bb11, ^bb13
    %120 = builtin.unrealized_conversion_cast %119 : index to i64
    %121 = builtin.unrealized_conversion_cast %119 : index to i64
    %122 = llvm.icmp "slt" %121, %1 : i64
    cf.cond_br %122, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %123 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.getelementptr %123[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.mul %120, %126 overflow<nsw, nuw> : i64
    %128 = llvm.getelementptr inbounds|nuw %125[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
    %130 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(128 : index) : i64
    %132 = llvm.mul %116, %131 overflow<nsw, nuw> : i64
    %133 = llvm.add %132, %120 overflow<nsw, nuw> : i64
    %134 = llvm.getelementptr inbounds|nuw %130[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %129, %134 : f32, !llvm.ptr
    %135 = llvm.add %121, %2 : i64
    %136 = builtin.unrealized_conversion_cast %135 : i64 to index
    cf.br ^bb12(%136 : index)
  ^bb14:  // pred: ^bb12
    %137 = llvm.add %117, %2 : i64
    %138 = builtin.unrealized_conversion_cast %137 : i64 to index
    cf.br ^bb10(%138 : index)
  ^bb15:  // pred: ^bb10
    %139 = llvm.mlir.constant(1 : index) : i64
    %140 = llvm.mlir.constant(128 : index) : i64
    %141 = llvm.mlir.constant(1 : index) : i64
    %142 = llvm.mlir.constant(128 : index) : i64
    %143 = llvm.mlir.zero : !llvm.ptr
    %144 = llvm.getelementptr %143[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.ptrtoint %144 : !llvm.ptr to i64
    %146 = llvm.mlir.constant(64 : index) : i64
    %147 = llvm.add %145, %146 : i64
    %148 = llvm.call @malloc(%147) : (i64) -> !llvm.ptr
    %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
    %150 = llvm.mlir.constant(1 : index) : i64
    %151 = llvm.sub %146, %150 : i64
    %152 = llvm.add %149, %151 : i64
    %153 = llvm.urem %152, %146 : i64
    %154 = llvm.sub %152, %153 : i64
    %155 = llvm.inttoptr %154 : i64 to !llvm.ptr
    %156 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %157 = llvm.insertvalue %148, %156[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.mlir.constant(0 : index) : i64
    %160 = llvm.insertvalue %159, %158[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %139, %160[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %140, %162[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = builtin.unrealized_conversion_cast %164 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x128xf32>
    cf.br ^bb16(%4 : index)
  ^bb16(%166: index):  // 2 preds: ^bb15, ^bb20
    %167 = builtin.unrealized_conversion_cast %166 : index to i64
    %168 = builtin.unrealized_conversion_cast %166 : index to i64
    %169 = llvm.icmp "slt" %168, %2 : i64
    cf.cond_br %169, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%4 : index)
  ^bb18(%170: index):  // 2 preds: ^bb17, ^bb19
    %171 = builtin.unrealized_conversion_cast %170 : index to i64
    %172 = builtin.unrealized_conversion_cast %170 : index to i64
    %173 = llvm.icmp "slt" %172, %1 : i64
    cf.cond_br %173, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %174 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.mlir.constant(128 : index) : i64
    %176 = llvm.mul %167, %175 overflow<nsw, nuw> : i64
    %177 = llvm.add %176, %171 overflow<nsw, nuw> : i64
    %178 = llvm.getelementptr inbounds|nuw %174[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %179 = llvm.load %178 : !llvm.ptr -> f32
    %180 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.mlir.constant(128 : index) : i64
    %182 = llvm.mul %167, %181 overflow<nsw, nuw> : i64
    %183 = llvm.add %182, %171 overflow<nsw, nuw> : i64
    %184 = llvm.getelementptr inbounds|nuw %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %185 = llvm.load %184 : !llvm.ptr -> f32
    %186 = llvm.fadd %179, %185 : f32
    %187 = llvm.intr.maximum(%186, %5) : (f32, f32) -> f32
    %188 = llvm.extractvalue %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %189 = llvm.mlir.constant(128 : index) : i64
    %190 = llvm.mul %167, %189 overflow<nsw, nuw> : i64
    %191 = llvm.add %190, %171 overflow<nsw, nuw> : i64
    %192 = llvm.getelementptr inbounds|nuw %188[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %187, %192 : f32, !llvm.ptr
    %193 = llvm.add %172, %2 : i64
    %194 = builtin.unrealized_conversion_cast %193 : i64 to index
    cf.br ^bb18(%194 : index)
  ^bb20:  // pred: ^bb18
    %195 = llvm.add %168, %2 : i64
    %196 = builtin.unrealized_conversion_cast %195 : i64 to index
    cf.br ^bb16(%196 : index)
  ^bb21:  // pred: ^bb16
    %197 = bufferization.to_tensor %165 : memref<1x128xf32> to tensor<1x128xf32>
    return %197 : tensor<1x128xf32>
  }
  func.func @wide_linear_layer_f32(%arg0: tensor<64x256xf32>, %arg1: tensor<256x1024xf32>, %arg2: tensor<1024xf32>) -> tensor<64x1024xf32> {
    %0 = llvm.mlir.constant(256 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %7 = bufferization.to_buffer %arg2 : tensor<1024xf32> to memref<1024xf32, strided<[?], offset: ?>>
    %8 = builtin.unrealized_conversion_cast %7 : memref<1024xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = bufferization.to_buffer %arg1 : tensor<256x1024xf32> to memref<256x1024xf32, strided<[?, ?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<256x1024xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = bufferization.to_buffer %arg0 : tensor<64x256xf32> to memref<64x256xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<64x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.mlir.constant(1024 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(65536 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.sub %20, %24 : i64
    %26 = llvm.add %23, %25 : i64
    %27 = llvm.urem %26, %20 : i64
    %28 = llvm.sub %26, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %22, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%39: index):  // 2 preds: ^bb0, ^bb8
    %40 = builtin.unrealized_conversion_cast %39 : index to i64
    %41 = builtin.unrealized_conversion_cast %39 : index to i64
    %42 = llvm.icmp "slt" %41, %3 : i64
    cf.cond_br %42, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%43: index):  // 2 preds: ^bb2, ^bb7
    %44 = builtin.unrealized_conversion_cast %43 : index to i64
    %45 = builtin.unrealized_conversion_cast %43 : index to i64
    %46 = llvm.icmp "slt" %45, %1 : i64
    cf.cond_br %46, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%47: index):  // 2 preds: ^bb4, ^bb6
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %49, %0 : i64
    cf.cond_br %50, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %40, %54 overflow<nsw, nuw> : i64
    %56 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %48, %56 overflow<nsw, nuw> : i64
    %58 = llvm.add %55, %57 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %48, %64 overflow<nsw, nuw> : i64
    %66 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %44, %66 overflow<nsw, nuw> : i64
    %68 = llvm.add %65, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds|nuw %63[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(1024 : index) : i64
    %73 = llvm.mul %40, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %44 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %60, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(1024 : index) : i64
    %81 = llvm.mul %40, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %44 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %79[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %83 : f32, !llvm.ptr
    %84 = llvm.add %49, %2 : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    cf.br ^bb5(%85 : index)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %45, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb3(%87 : index)
  ^bb8:  // pred: ^bb3
    %88 = llvm.add %41, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb1(%89 : index)
  ^bb9:  // pred: ^bb1
    %90 = llvm.mlir.constant(64 : index) : i64
    %91 = llvm.mlir.constant(1024 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(65536 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.constant(64 : index) : i64
    %98 = llvm.add %96, %97 : i64
    %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.sub %97, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.urem %103, %97 : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.insertvalue %99, %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %91, %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %92, %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%116: index):  // 2 preds: ^bb9, ^bb14
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    %118 = builtin.unrealized_conversion_cast %116 : index to i64
    %119 = llvm.icmp "slt" %118, %3 : i64
    cf.cond_br %119, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%120: index):  // 2 preds: ^bb11, ^bb13
    %121 = builtin.unrealized_conversion_cast %120 : index to i64
    %122 = builtin.unrealized_conversion_cast %120 : index to i64
    %123 = llvm.icmp "slt" %122, %1 : i64
    cf.cond_br %123, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %124 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mul %121, %127 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(1024 : index) : i64
    %133 = llvm.mul %117, %132 overflow<nsw, nuw> : i64
    %134 = llvm.add %133, %121 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %130, %135 : f32, !llvm.ptr
    %136 = llvm.add %122, %2 : i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    cf.br ^bb12(%137 : index)
  ^bb14:  // pred: ^bb12
    %138 = llvm.add %118, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb10(%139 : index)
  ^bb15:  // pred: ^bb10
    %140 = llvm.mlir.constant(64 : index) : i64
    %141 = llvm.mlir.constant(1024 : index) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(65536 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(64 : index) : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.sub %147, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.urem %153, %147 : i64
    %155 = llvm.sub %153, %154 : i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %149, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x1024xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%167: index):  // 2 preds: ^bb15, ^bb20
    %168 = builtin.unrealized_conversion_cast %167 : index to i64
    %169 = builtin.unrealized_conversion_cast %167 : index to i64
    %170 = llvm.icmp "slt" %169, %3 : i64
    cf.cond_br %170, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%171: index):  // 2 preds: ^bb17, ^bb19
    %172 = builtin.unrealized_conversion_cast %171 : index to i64
    %173 = builtin.unrealized_conversion_cast %171 : index to i64
    %174 = llvm.icmp "slt" %173, %1 : i64
    cf.cond_br %174, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %175 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(1024 : index) : i64
    %177 = llvm.mul %168, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %172 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(1024 : index) : i64
    %183 = llvm.mul %168, %182 overflow<nsw, nuw> : i64
    %184 = llvm.add %183, %172 overflow<nsw, nuw> : i64
    %185 = llvm.getelementptr inbounds|nuw %181[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %186 = llvm.load %185 : !llvm.ptr -> f32
    %187 = llvm.fadd %180, %186 : f32
    %188 = llvm.intr.maximum(%187, %6) : (f32, f32) -> f32
    %189 = llvm.extractvalue %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(1024 : index) : i64
    %191 = llvm.mul %168, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %172 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %193 : f32, !llvm.ptr
    %194 = llvm.add %173, %2 : i64
    %195 = builtin.unrealized_conversion_cast %194 : i64 to index
    cf.br ^bb18(%195 : index)
  ^bb20:  // pred: ^bb18
    %196 = llvm.add %169, %2 : i64
    %197 = builtin.unrealized_conversion_cast %196 : i64 to index
    cf.br ^bb16(%197 : index)
  ^bb21:  // pred: ^bb16
    %198 = bufferization.to_tensor %166 : memref<64x1024xf32> to tensor<64x1024xf32>
    return %198 : tensor<64x1024xf32>
  }
  func.func @narrow_linear_layer_f32(%arg0: tensor<128x1024xf32>, %arg1: tensor<1024x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64xf32> {
    %0 = llvm.mlir.constant(1024 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(128 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %7 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %8 = builtin.unrealized_conversion_cast %7 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = bufferization.to_buffer %arg1 : tensor<1024x64xf32> to memref<1024x64xf32, strided<[?, ?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<1024x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = bufferization.to_buffer %arg0 : tensor<128x1024xf32> to memref<128x1024xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<128x1024xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(128 : index) : i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(8192 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.sub %20, %24 : i64
    %26 = llvm.add %23, %25 : i64
    %27 = llvm.urem %26, %20 : i64
    %28 = llvm.sub %26, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %22, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%39: index):  // 2 preds: ^bb0, ^bb8
    %40 = builtin.unrealized_conversion_cast %39 : index to i64
    %41 = builtin.unrealized_conversion_cast %39 : index to i64
    %42 = llvm.icmp "slt" %41, %3 : i64
    cf.cond_br %42, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%43: index):  // 2 preds: ^bb2, ^bb7
    %44 = builtin.unrealized_conversion_cast %43 : index to i64
    %45 = builtin.unrealized_conversion_cast %43 : index to i64
    %46 = llvm.icmp "slt" %45, %1 : i64
    cf.cond_br %46, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%47: index):  // 2 preds: ^bb4, ^bb6
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %49, %0 : i64
    cf.cond_br %50, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %40, %54 overflow<nsw, nuw> : i64
    %56 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %48, %56 overflow<nsw, nuw> : i64
    %58 = llvm.add %55, %57 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %48, %64 overflow<nsw, nuw> : i64
    %66 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %44, %66 overflow<nsw, nuw> : i64
    %68 = llvm.add %65, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds|nuw %63[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(64 : index) : i64
    %73 = llvm.mul %40, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %44 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %60, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(64 : index) : i64
    %81 = llvm.mul %40, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %44 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %79[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %83 : f32, !llvm.ptr
    %84 = llvm.add %49, %2 : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    cf.br ^bb5(%85 : index)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %45, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb3(%87 : index)
  ^bb8:  // pred: ^bb3
    %88 = llvm.add %41, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb1(%89 : index)
  ^bb9:  // pred: ^bb1
    %90 = llvm.mlir.constant(128 : index) : i64
    %91 = llvm.mlir.constant(64 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(8192 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.constant(64 : index) : i64
    %98 = llvm.add %96, %97 : i64
    %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.sub %97, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.urem %103, %97 : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.insertvalue %99, %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %91, %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %92, %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%116: index):  // 2 preds: ^bb9, ^bb14
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    %118 = builtin.unrealized_conversion_cast %116 : index to i64
    %119 = llvm.icmp "slt" %118, %3 : i64
    cf.cond_br %119, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%120: index):  // 2 preds: ^bb11, ^bb13
    %121 = builtin.unrealized_conversion_cast %120 : index to i64
    %122 = builtin.unrealized_conversion_cast %120 : index to i64
    %123 = llvm.icmp "slt" %122, %1 : i64
    cf.cond_br %123, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %124 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mul %121, %127 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(64 : index) : i64
    %133 = llvm.mul %117, %132 overflow<nsw, nuw> : i64
    %134 = llvm.add %133, %121 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %130, %135 : f32, !llvm.ptr
    %136 = llvm.add %122, %2 : i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    cf.br ^bb12(%137 : index)
  ^bb14:  // pred: ^bb12
    %138 = llvm.add %118, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb10(%139 : index)
  ^bb15:  // pred: ^bb10
    %140 = llvm.mlir.constant(128 : index) : i64
    %141 = llvm.mlir.constant(64 : index) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(8192 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(64 : index) : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.sub %147, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.urem %153, %147 : i64
    %155 = llvm.sub %153, %154 : i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %149, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<128x64xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%167: index):  // 2 preds: ^bb15, ^bb20
    %168 = builtin.unrealized_conversion_cast %167 : index to i64
    %169 = builtin.unrealized_conversion_cast %167 : index to i64
    %170 = llvm.icmp "slt" %169, %3 : i64
    cf.cond_br %170, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%171: index):  // 2 preds: ^bb17, ^bb19
    %172 = builtin.unrealized_conversion_cast %171 : index to i64
    %173 = builtin.unrealized_conversion_cast %171 : index to i64
    %174 = llvm.icmp "slt" %173, %1 : i64
    cf.cond_br %174, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %175 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(64 : index) : i64
    %177 = llvm.mul %168, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %172 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(64 : index) : i64
    %183 = llvm.mul %168, %182 overflow<nsw, nuw> : i64
    %184 = llvm.add %183, %172 overflow<nsw, nuw> : i64
    %185 = llvm.getelementptr inbounds|nuw %181[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %186 = llvm.load %185 : !llvm.ptr -> f32
    %187 = llvm.fadd %180, %186 : f32
    %188 = llvm.intr.maximum(%187, %6) : (f32, f32) -> f32
    %189 = llvm.extractvalue %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(64 : index) : i64
    %191 = llvm.mul %168, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %172 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %193 : f32, !llvm.ptr
    %194 = llvm.add %173, %2 : i64
    %195 = builtin.unrealized_conversion_cast %194 : i64 to index
    cf.br ^bb18(%195 : index)
  ^bb20:  // pred: ^bb18
    %196 = llvm.add %169, %2 : i64
    %197 = builtin.unrealized_conversion_cast %196 : i64 to index
    cf.br ^bb16(%197 : index)
  ^bb21:  // pred: ^bb16
    %198 = bufferization.to_tensor %166 : memref<128x64xf32> to tensor<128x64xf32>
    return %198 : tensor<128x64xf32>
  }
  func.func @multi_layer_network(%arg0: tensor<32x128xf32>, %arg1: tensor<128x256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256x128xf32>, %arg4: tensor<128xf32>, %arg5: tensor<128x10xf32>, %arg6: tensor<10xf32>) -> tensor<32x10xf32> {
    %0 = llvm.mlir.constant(10 : index) : i64
    %1 = llvm.mlir.constant(128 : index) : i64
    %2 = llvm.mlir.constant(256 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = builtin.unrealized_conversion_cast %5 : i64 to index
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = bufferization.to_buffer %arg6 : tensor<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    %9 = builtin.unrealized_conversion_cast %8 : memref<10xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = bufferization.to_buffer %arg5 : tensor<128x10xf32> to memref<128x10xf32, strided<[?, ?], offset: ?>>
    %11 = builtin.unrealized_conversion_cast %10 : memref<128x10xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = bufferization.to_buffer %arg4 : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
    %13 = builtin.unrealized_conversion_cast %12 : memref<128xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %14 = bufferization.to_buffer %arg3 : tensor<256x128xf32> to memref<256x128xf32, strided<[?, ?], offset: ?>>
    %15 = builtin.unrealized_conversion_cast %14 : memref<256x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = bufferization.to_buffer %arg2 : tensor<256xf32> to memref<256xf32, strided<[?], offset: ?>>
    %17 = builtin.unrealized_conversion_cast %16 : memref<256xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %18 = bufferization.to_buffer %arg1 : tensor<128x256xf32> to memref<128x256xf32, strided<[?, ?], offset: ?>>
    %19 = builtin.unrealized_conversion_cast %18 : memref<128x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %21 = builtin.unrealized_conversion_cast %20 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.mlir.constant(32 : index) : i64
    %23 = llvm.mlir.constant(256 : index) : i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(8192 : index) : i64
    %26 = llvm.mlir.zero : !llvm.ptr
    %27 = llvm.getelementptr %26[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.mlir.constant(64 : index) : i64
    %30 = llvm.add %28, %29 : i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.sub %29, %33 : i64
    %35 = llvm.add %32, %34 : i64
    %36 = llvm.urem %35, %29 : i64
    %37 = llvm.sub %35, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %31, %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %38, %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %22, %43[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %23, %44[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %23, %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %24, %46[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%6 : index)
  ^bb1(%48: index):  // 2 preds: ^bb0, ^bb8
    %49 = builtin.unrealized_conversion_cast %48 : index to i64
    %50 = builtin.unrealized_conversion_cast %48 : index to i64
    %51 = llvm.icmp "slt" %50, %4 : i64
    cf.cond_br %51, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%6 : index)
  ^bb3(%52: index):  // 2 preds: ^bb2, ^bb7
    %53 = builtin.unrealized_conversion_cast %52 : index to i64
    %54 = builtin.unrealized_conversion_cast %52 : index to i64
    %55 = llvm.icmp "slt" %54, %2 : i64
    cf.cond_br %55, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%6 : index)
  ^bb5(%56: index):  // 2 preds: ^bb4, ^bb6
    %57 = builtin.unrealized_conversion_cast %56 : index to i64
    %58 = builtin.unrealized_conversion_cast %56 : index to i64
    %59 = llvm.icmp "slt" %58, %1 : i64
    cf.cond_br %59, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %60 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %21[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.extractvalue %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %49, %63 overflow<nsw, nuw> : i64
    %65 = llvm.extractvalue %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %57, %65 overflow<nsw, nuw> : i64
    %67 = llvm.add %64, %66 overflow<nsw, nuw> : i64
    %68 = llvm.getelementptr inbounds|nuw %62[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.extractvalue %19[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.getelementptr %70[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mul %57, %73 overflow<nsw, nuw> : i64
    %75 = llvm.extractvalue %19[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mul %53, %75 overflow<nsw, nuw> : i64
    %77 = llvm.add %74, %76 overflow<nsw, nuw> : i64
    %78 = llvm.getelementptr inbounds|nuw %72[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.load %78 : !llvm.ptr -> f32
    %80 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.mlir.constant(256 : index) : i64
    %82 = llvm.mul %49, %81 overflow<nsw, nuw> : i64
    %83 = llvm.add %82, %53 overflow<nsw, nuw> : i64
    %84 = llvm.getelementptr inbounds|nuw %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 : !llvm.ptr -> f32
    %86 = llvm.fmul %69, %79 : f32
    %87 = llvm.fadd %85, %86 : f32
    %88 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.mlir.constant(256 : index) : i64
    %90 = llvm.mul %49, %89 overflow<nsw, nuw> : i64
    %91 = llvm.add %90, %53 overflow<nsw, nuw> : i64
    %92 = llvm.getelementptr inbounds|nuw %88[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %87, %92 : f32, !llvm.ptr
    %93 = llvm.add %58, %3 : i64
    %94 = builtin.unrealized_conversion_cast %93 : i64 to index
    cf.br ^bb5(%94 : index)
  ^bb7:  // pred: ^bb5
    %95 = llvm.add %54, %3 : i64
    %96 = builtin.unrealized_conversion_cast %95 : i64 to index
    cf.br ^bb3(%96 : index)
  ^bb8:  // pred: ^bb3
    %97 = llvm.add %50, %3 : i64
    %98 = builtin.unrealized_conversion_cast %97 : i64 to index
    cf.br ^bb1(%98 : index)
  ^bb9:  // pred: ^bb1
    %99 = llvm.mlir.constant(32 : index) : i64
    %100 = llvm.mlir.constant(256 : index) : i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.mlir.constant(8192 : index) : i64
    %103 = llvm.mlir.zero : !llvm.ptr
    %104 = llvm.getelementptr %103[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %105 = llvm.ptrtoint %104 : !llvm.ptr to i64
    %106 = llvm.mlir.constant(64 : index) : i64
    %107 = llvm.add %105, %106 : i64
    %108 = llvm.call @malloc(%107) : (i64) -> !llvm.ptr
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.sub %106, %110 : i64
    %112 = llvm.add %109, %111 : i64
    %113 = llvm.urem %112, %106 : i64
    %114 = llvm.sub %112, %113 : i64
    %115 = llvm.inttoptr %114 : i64 to !llvm.ptr
    %116 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %117 = llvm.insertvalue %108, %116[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.insertvalue %115, %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.mlir.constant(0 : index) : i64
    %120 = llvm.insertvalue %119, %118[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.insertvalue %99, %120[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.insertvalue %100, %121[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.insertvalue %100, %122[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.insertvalue %101, %123[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%6 : index)
  ^bb10(%125: index):  // 2 preds: ^bb9, ^bb14
    %126 = builtin.unrealized_conversion_cast %125 : index to i64
    %127 = builtin.unrealized_conversion_cast %125 : index to i64
    %128 = llvm.icmp "slt" %127, %4 : i64
    cf.cond_br %128, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%6 : index)
  ^bb12(%129: index):  // 2 preds: ^bb11, ^bb13
    %130 = builtin.unrealized_conversion_cast %129 : index to i64
    %131 = builtin.unrealized_conversion_cast %129 : index to i64
    %132 = llvm.icmp "slt" %131, %2 : i64
    cf.cond_br %132, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %133 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %134 = llvm.extractvalue %17[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %135 = llvm.getelementptr %133[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %136 = llvm.extractvalue %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %137 = llvm.mul %130, %136 overflow<nsw, nuw> : i64
    %138 = llvm.getelementptr inbounds|nuw %135[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %139 = llvm.load %138 : !llvm.ptr -> f32
    %140 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(256 : index) : i64
    %142 = llvm.mul %126, %141 overflow<nsw, nuw> : i64
    %143 = llvm.add %142, %130 overflow<nsw, nuw> : i64
    %144 = llvm.getelementptr inbounds|nuw %140[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %139, %144 : f32, !llvm.ptr
    %145 = llvm.add %131, %3 : i64
    %146 = builtin.unrealized_conversion_cast %145 : i64 to index
    cf.br ^bb12(%146 : index)
  ^bb14:  // pred: ^bb12
    %147 = llvm.add %127, %3 : i64
    %148 = builtin.unrealized_conversion_cast %147 : i64 to index
    cf.br ^bb10(%148 : index)
  ^bb15:  // pred: ^bb10
    %149 = llvm.mlir.constant(32 : index) : i64
    %150 = llvm.mlir.constant(256 : index) : i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.mlir.constant(8192 : index) : i64
    %153 = llvm.mlir.zero : !llvm.ptr
    %154 = llvm.getelementptr %153[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %155 = llvm.ptrtoint %154 : !llvm.ptr to i64
    %156 = llvm.mlir.constant(64 : index) : i64
    %157 = llvm.add %155, %156 : i64
    %158 = llvm.call @malloc(%157) : (i64) -> !llvm.ptr
    %159 = llvm.ptrtoint %158 : !llvm.ptr to i64
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.sub %156, %160 : i64
    %162 = llvm.add %159, %161 : i64
    %163 = llvm.urem %162, %156 : i64
    %164 = llvm.sub %162, %163 : i64
    %165 = llvm.inttoptr %164 : i64 to !llvm.ptr
    %166 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %167 = llvm.insertvalue %158, %166[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = llvm.insertvalue %165, %167[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.mlir.constant(0 : index) : i64
    %170 = llvm.insertvalue %169, %168[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.insertvalue %149, %170[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.insertvalue %150, %171[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %173 = llvm.insertvalue %150, %172[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %174 = llvm.insertvalue %151, %173[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb16(%6 : index)
  ^bb16(%175: index):  // 2 preds: ^bb15, ^bb20
    %176 = builtin.unrealized_conversion_cast %175 : index to i64
    %177 = builtin.unrealized_conversion_cast %175 : index to i64
    %178 = llvm.icmp "slt" %177, %4 : i64
    cf.cond_br %178, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%6 : index)
  ^bb18(%179: index):  // 2 preds: ^bb17, ^bb19
    %180 = builtin.unrealized_conversion_cast %179 : index to i64
    %181 = builtin.unrealized_conversion_cast %179 : index to i64
    %182 = llvm.icmp "slt" %181, %2 : i64
    cf.cond_br %182, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %183 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.mlir.constant(256 : index) : i64
    %185 = llvm.mul %176, %184 overflow<nsw, nuw> : i64
    %186 = llvm.add %185, %180 overflow<nsw, nuw> : i64
    %187 = llvm.getelementptr inbounds|nuw %183[%186] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %188 = llvm.load %187 : !llvm.ptr -> f32
    %189 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(256 : index) : i64
    %191 = llvm.mul %176, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %180 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %194 = llvm.load %193 : !llvm.ptr -> f32
    %195 = llvm.fadd %188, %194 : f32
    %196 = llvm.intr.maximum(%195, %7) : (f32, f32) -> f32
    %197 = llvm.extractvalue %174[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %198 = llvm.mlir.constant(256 : index) : i64
    %199 = llvm.mul %176, %198 overflow<nsw, nuw> : i64
    %200 = llvm.add %199, %180 overflow<nsw, nuw> : i64
    %201 = llvm.getelementptr inbounds|nuw %197[%200] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %196, %201 : f32, !llvm.ptr
    %202 = llvm.add %181, %3 : i64
    %203 = builtin.unrealized_conversion_cast %202 : i64 to index
    cf.br ^bb18(%203 : index)
  ^bb20:  // pred: ^bb18
    %204 = llvm.add %177, %3 : i64
    %205 = builtin.unrealized_conversion_cast %204 : i64 to index
    cf.br ^bb16(%205 : index)
  ^bb21:  // pred: ^bb16
    %206 = llvm.mlir.constant(32 : index) : i64
    %207 = llvm.mlir.constant(128 : index) : i64
    %208 = llvm.mlir.constant(1 : index) : i64
    %209 = llvm.mlir.constant(4096 : index) : i64
    %210 = llvm.mlir.zero : !llvm.ptr
    %211 = llvm.getelementptr %210[%209] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %212 = llvm.ptrtoint %211 : !llvm.ptr to i64
    %213 = llvm.mlir.constant(64 : index) : i64
    %214 = llvm.add %212, %213 : i64
    %215 = llvm.call @malloc(%214) : (i64) -> !llvm.ptr
    %216 = llvm.ptrtoint %215 : !llvm.ptr to i64
    %217 = llvm.mlir.constant(1 : index) : i64
    %218 = llvm.sub %213, %217 : i64
    %219 = llvm.add %216, %218 : i64
    %220 = llvm.urem %219, %213 : i64
    %221 = llvm.sub %219, %220 : i64
    %222 = llvm.inttoptr %221 : i64 to !llvm.ptr
    %223 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %224 = llvm.insertvalue %215, %223[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %225 = llvm.insertvalue %222, %224[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %226 = llvm.mlir.constant(0 : index) : i64
    %227 = llvm.insertvalue %226, %225[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %228 = llvm.insertvalue %206, %227[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.insertvalue %207, %228[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.insertvalue %207, %229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %231 = llvm.insertvalue %208, %230[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb22(%6 : index)
  ^bb22(%232: index):  // 2 preds: ^bb21, ^bb29
    %233 = builtin.unrealized_conversion_cast %232 : index to i64
    %234 = builtin.unrealized_conversion_cast %232 : index to i64
    %235 = llvm.icmp "slt" %234, %4 : i64
    cf.cond_br %235, ^bb23, ^bb30
  ^bb23:  // pred: ^bb22
    cf.br ^bb24(%6 : index)
  ^bb24(%236: index):  // 2 preds: ^bb23, ^bb28
    %237 = builtin.unrealized_conversion_cast %236 : index to i64
    %238 = builtin.unrealized_conversion_cast %236 : index to i64
    %239 = llvm.icmp "slt" %238, %1 : i64
    cf.cond_br %239, ^bb25, ^bb29
  ^bb25:  // pred: ^bb24
    cf.br ^bb26(%6 : index)
  ^bb26(%240: index):  // 2 preds: ^bb25, ^bb27
    %241 = builtin.unrealized_conversion_cast %240 : index to i64
    %242 = builtin.unrealized_conversion_cast %240 : index to i64
    %243 = llvm.icmp "slt" %242, %2 : i64
    cf.cond_br %243, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %244 = llvm.extractvalue %174[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.mlir.constant(256 : index) : i64
    %246 = llvm.mul %233, %245 overflow<nsw, nuw> : i64
    %247 = llvm.add %246, %241 overflow<nsw, nuw> : i64
    %248 = llvm.getelementptr inbounds|nuw %244[%247] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %249 = llvm.load %248 : !llvm.ptr -> f32
    %250 = llvm.extractvalue %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %251 = llvm.extractvalue %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %252 = llvm.getelementptr %250[%251] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %253 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.mul %241, %253 overflow<nsw, nuw> : i64
    %255 = llvm.extractvalue %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %256 = llvm.mul %237, %255 overflow<nsw, nuw> : i64
    %257 = llvm.add %254, %256 overflow<nsw, nuw> : i64
    %258 = llvm.getelementptr inbounds|nuw %252[%257] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %259 = llvm.load %258 : !llvm.ptr -> f32
    %260 = llvm.extractvalue %231[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %261 = llvm.mlir.constant(128 : index) : i64
    %262 = llvm.mul %233, %261 overflow<nsw, nuw> : i64
    %263 = llvm.add %262, %237 overflow<nsw, nuw> : i64
    %264 = llvm.getelementptr inbounds|nuw %260[%263] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %265 = llvm.load %264 : !llvm.ptr -> f32
    %266 = llvm.fmul %249, %259 : f32
    %267 = llvm.fadd %265, %266 : f32
    %268 = llvm.extractvalue %231[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.mlir.constant(128 : index) : i64
    %270 = llvm.mul %233, %269 overflow<nsw, nuw> : i64
    %271 = llvm.add %270, %237 overflow<nsw, nuw> : i64
    %272 = llvm.getelementptr inbounds|nuw %268[%271] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %267, %272 : f32, !llvm.ptr
    %273 = llvm.add %242, %3 : i64
    %274 = builtin.unrealized_conversion_cast %273 : i64 to index
    cf.br ^bb26(%274 : index)
  ^bb28:  // pred: ^bb26
    %275 = llvm.add %238, %3 : i64
    %276 = builtin.unrealized_conversion_cast %275 : i64 to index
    cf.br ^bb24(%276 : index)
  ^bb29:  // pred: ^bb24
    %277 = llvm.add %234, %3 : i64
    %278 = builtin.unrealized_conversion_cast %277 : i64 to index
    cf.br ^bb22(%278 : index)
  ^bb30:  // pred: ^bb22
    %279 = llvm.mlir.constant(32 : index) : i64
    %280 = llvm.mlir.constant(128 : index) : i64
    %281 = llvm.mlir.constant(1 : index) : i64
    %282 = llvm.mlir.constant(4096 : index) : i64
    %283 = llvm.mlir.zero : !llvm.ptr
    %284 = llvm.getelementptr %283[%282] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %285 = llvm.ptrtoint %284 : !llvm.ptr to i64
    %286 = llvm.mlir.constant(64 : index) : i64
    %287 = llvm.add %285, %286 : i64
    %288 = llvm.call @malloc(%287) : (i64) -> !llvm.ptr
    %289 = llvm.ptrtoint %288 : !llvm.ptr to i64
    %290 = llvm.mlir.constant(1 : index) : i64
    %291 = llvm.sub %286, %290 : i64
    %292 = llvm.add %289, %291 : i64
    %293 = llvm.urem %292, %286 : i64
    %294 = llvm.sub %292, %293 : i64
    %295 = llvm.inttoptr %294 : i64 to !llvm.ptr
    %296 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %297 = llvm.insertvalue %288, %296[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %298 = llvm.insertvalue %295, %297[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.mlir.constant(0 : index) : i64
    %300 = llvm.insertvalue %299, %298[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %301 = llvm.insertvalue %279, %300[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %302 = llvm.insertvalue %280, %301[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %303 = llvm.insertvalue %280, %302[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %304 = llvm.insertvalue %281, %303[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb31(%6 : index)
  ^bb31(%305: index):  // 2 preds: ^bb30, ^bb35
    %306 = builtin.unrealized_conversion_cast %305 : index to i64
    %307 = builtin.unrealized_conversion_cast %305 : index to i64
    %308 = llvm.icmp "slt" %307, %4 : i64
    cf.cond_br %308, ^bb32, ^bb36
  ^bb32:  // pred: ^bb31
    cf.br ^bb33(%6 : index)
  ^bb33(%309: index):  // 2 preds: ^bb32, ^bb34
    %310 = builtin.unrealized_conversion_cast %309 : index to i64
    %311 = builtin.unrealized_conversion_cast %309 : index to i64
    %312 = llvm.icmp "slt" %311, %1 : i64
    cf.cond_br %312, ^bb34, ^bb35
  ^bb34:  // pred: ^bb33
    %313 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %314 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %315 = llvm.getelementptr %313[%314] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %316 = llvm.extractvalue %13[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %317 = llvm.mul %310, %316 overflow<nsw, nuw> : i64
    %318 = llvm.getelementptr inbounds|nuw %315[%317] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %319 = llvm.load %318 : !llvm.ptr -> f32
    %320 = llvm.extractvalue %304[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %321 = llvm.mlir.constant(128 : index) : i64
    %322 = llvm.mul %306, %321 overflow<nsw, nuw> : i64
    %323 = llvm.add %322, %310 overflow<nsw, nuw> : i64
    %324 = llvm.getelementptr inbounds|nuw %320[%323] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %319, %324 : f32, !llvm.ptr
    %325 = llvm.add %311, %3 : i64
    %326 = builtin.unrealized_conversion_cast %325 : i64 to index
    cf.br ^bb33(%326 : index)
  ^bb35:  // pred: ^bb33
    %327 = llvm.add %307, %3 : i64
    %328 = builtin.unrealized_conversion_cast %327 : i64 to index
    cf.br ^bb31(%328 : index)
  ^bb36:  // pred: ^bb31
    %329 = llvm.mlir.constant(32 : index) : i64
    %330 = llvm.mlir.constant(128 : index) : i64
    %331 = llvm.mlir.constant(1 : index) : i64
    %332 = llvm.mlir.constant(4096 : index) : i64
    %333 = llvm.mlir.zero : !llvm.ptr
    %334 = llvm.getelementptr %333[%332] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %335 = llvm.ptrtoint %334 : !llvm.ptr to i64
    %336 = llvm.mlir.constant(64 : index) : i64
    %337 = llvm.add %335, %336 : i64
    %338 = llvm.call @malloc(%337) : (i64) -> !llvm.ptr
    %339 = llvm.ptrtoint %338 : !llvm.ptr to i64
    %340 = llvm.mlir.constant(1 : index) : i64
    %341 = llvm.sub %336, %340 : i64
    %342 = llvm.add %339, %341 : i64
    %343 = llvm.urem %342, %336 : i64
    %344 = llvm.sub %342, %343 : i64
    %345 = llvm.inttoptr %344 : i64 to !llvm.ptr
    %346 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %347 = llvm.insertvalue %338, %346[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %348 = llvm.insertvalue %345, %347[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %349 = llvm.mlir.constant(0 : index) : i64
    %350 = llvm.insertvalue %349, %348[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %351 = llvm.insertvalue %329, %350[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %352 = llvm.insertvalue %330, %351[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %353 = llvm.insertvalue %330, %352[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.insertvalue %331, %353[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb37(%6 : index)
  ^bb37(%355: index):  // 2 preds: ^bb36, ^bb41
    %356 = builtin.unrealized_conversion_cast %355 : index to i64
    %357 = builtin.unrealized_conversion_cast %355 : index to i64
    %358 = llvm.icmp "slt" %357, %4 : i64
    cf.cond_br %358, ^bb38, ^bb42
  ^bb38:  // pred: ^bb37
    cf.br ^bb39(%6 : index)
  ^bb39(%359: index):  // 2 preds: ^bb38, ^bb40
    %360 = builtin.unrealized_conversion_cast %359 : index to i64
    %361 = builtin.unrealized_conversion_cast %359 : index to i64
    %362 = llvm.icmp "slt" %361, %1 : i64
    cf.cond_br %362, ^bb40, ^bb41
  ^bb40:  // pred: ^bb39
    %363 = llvm.extractvalue %231[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %364 = llvm.mlir.constant(128 : index) : i64
    %365 = llvm.mul %356, %364 overflow<nsw, nuw> : i64
    %366 = llvm.add %365, %360 overflow<nsw, nuw> : i64
    %367 = llvm.getelementptr inbounds|nuw %363[%366] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %368 = llvm.load %367 : !llvm.ptr -> f32
    %369 = llvm.extractvalue %304[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %370 = llvm.mlir.constant(128 : index) : i64
    %371 = llvm.mul %356, %370 overflow<nsw, nuw> : i64
    %372 = llvm.add %371, %360 overflow<nsw, nuw> : i64
    %373 = llvm.getelementptr inbounds|nuw %369[%372] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %374 = llvm.load %373 : !llvm.ptr -> f32
    %375 = llvm.fadd %368, %374 : f32
    %376 = llvm.intr.maximum(%375, %7) : (f32, f32) -> f32
    %377 = llvm.extractvalue %354[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %378 = llvm.mlir.constant(128 : index) : i64
    %379 = llvm.mul %356, %378 overflow<nsw, nuw> : i64
    %380 = llvm.add %379, %360 overflow<nsw, nuw> : i64
    %381 = llvm.getelementptr inbounds|nuw %377[%380] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %376, %381 : f32, !llvm.ptr
    %382 = llvm.add %361, %3 : i64
    %383 = builtin.unrealized_conversion_cast %382 : i64 to index
    cf.br ^bb39(%383 : index)
  ^bb41:  // pred: ^bb39
    %384 = llvm.add %357, %3 : i64
    %385 = builtin.unrealized_conversion_cast %384 : i64 to index
    cf.br ^bb37(%385 : index)
  ^bb42:  // pred: ^bb37
    %386 = llvm.mlir.constant(32 : index) : i64
    %387 = llvm.mlir.constant(10 : index) : i64
    %388 = llvm.mlir.constant(1 : index) : i64
    %389 = llvm.mlir.constant(320 : index) : i64
    %390 = llvm.mlir.zero : !llvm.ptr
    %391 = llvm.getelementptr %390[%389] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %392 = llvm.ptrtoint %391 : !llvm.ptr to i64
    %393 = llvm.mlir.constant(64 : index) : i64
    %394 = llvm.add %392, %393 : i64
    %395 = llvm.call @malloc(%394) : (i64) -> !llvm.ptr
    %396 = llvm.ptrtoint %395 : !llvm.ptr to i64
    %397 = llvm.mlir.constant(1 : index) : i64
    %398 = llvm.sub %393, %397 : i64
    %399 = llvm.add %396, %398 : i64
    %400 = llvm.urem %399, %393 : i64
    %401 = llvm.sub %399, %400 : i64
    %402 = llvm.inttoptr %401 : i64 to !llvm.ptr
    %403 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %404 = llvm.insertvalue %395, %403[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %405 = llvm.insertvalue %402, %404[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %406 = llvm.mlir.constant(0 : index) : i64
    %407 = llvm.insertvalue %406, %405[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %408 = llvm.insertvalue %386, %407[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %409 = llvm.insertvalue %387, %408[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %410 = llvm.insertvalue %387, %409[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %411 = llvm.insertvalue %388, %410[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb43(%6 : index)
  ^bb43(%412: index):  // 2 preds: ^bb42, ^bb50
    %413 = builtin.unrealized_conversion_cast %412 : index to i64
    %414 = builtin.unrealized_conversion_cast %412 : index to i64
    %415 = llvm.icmp "slt" %414, %4 : i64
    cf.cond_br %415, ^bb44, ^bb51
  ^bb44:  // pred: ^bb43
    cf.br ^bb45(%6 : index)
  ^bb45(%416: index):  // 2 preds: ^bb44, ^bb49
    %417 = builtin.unrealized_conversion_cast %416 : index to i64
    %418 = builtin.unrealized_conversion_cast %416 : index to i64
    %419 = llvm.icmp "slt" %418, %0 : i64
    cf.cond_br %419, ^bb46, ^bb50
  ^bb46:  // pred: ^bb45
    cf.br ^bb47(%6 : index)
  ^bb47(%420: index):  // 2 preds: ^bb46, ^bb48
    %421 = builtin.unrealized_conversion_cast %420 : index to i64
    %422 = builtin.unrealized_conversion_cast %420 : index to i64
    %423 = llvm.icmp "slt" %422, %1 : i64
    cf.cond_br %423, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    %424 = llvm.extractvalue %354[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %425 = llvm.mlir.constant(128 : index) : i64
    %426 = llvm.mul %413, %425 overflow<nsw, nuw> : i64
    %427 = llvm.add %426, %421 overflow<nsw, nuw> : i64
    %428 = llvm.getelementptr inbounds|nuw %424[%427] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %429 = llvm.load %428 : !llvm.ptr -> f32
    %430 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %431 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %432 = llvm.getelementptr %430[%431] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %433 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %434 = llvm.mul %421, %433 overflow<nsw, nuw> : i64
    %435 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %436 = llvm.mul %417, %435 overflow<nsw, nuw> : i64
    %437 = llvm.add %434, %436 overflow<nsw, nuw> : i64
    %438 = llvm.getelementptr inbounds|nuw %432[%437] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %439 = llvm.load %438 : !llvm.ptr -> f32
    %440 = llvm.extractvalue %411[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %441 = llvm.mlir.constant(10 : index) : i64
    %442 = llvm.mul %413, %441 overflow<nsw, nuw> : i64
    %443 = llvm.add %442, %417 overflow<nsw, nuw> : i64
    %444 = llvm.getelementptr inbounds|nuw %440[%443] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %445 = llvm.load %444 : !llvm.ptr -> f32
    %446 = llvm.fmul %429, %439 : f32
    %447 = llvm.fadd %445, %446 : f32
    %448 = llvm.extractvalue %411[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %449 = llvm.mlir.constant(10 : index) : i64
    %450 = llvm.mul %413, %449 overflow<nsw, nuw> : i64
    %451 = llvm.add %450, %417 overflow<nsw, nuw> : i64
    %452 = llvm.getelementptr inbounds|nuw %448[%451] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %447, %452 : f32, !llvm.ptr
    %453 = llvm.add %422, %3 : i64
    %454 = builtin.unrealized_conversion_cast %453 : i64 to index
    cf.br ^bb47(%454 : index)
  ^bb49:  // pred: ^bb47
    %455 = llvm.add %418, %3 : i64
    %456 = builtin.unrealized_conversion_cast %455 : i64 to index
    cf.br ^bb45(%456 : index)
  ^bb50:  // pred: ^bb45
    %457 = llvm.add %414, %3 : i64
    %458 = builtin.unrealized_conversion_cast %457 : i64 to index
    cf.br ^bb43(%458 : index)
  ^bb51:  // pred: ^bb43
    %459 = llvm.mlir.constant(32 : index) : i64
    %460 = llvm.mlir.constant(10 : index) : i64
    %461 = llvm.mlir.constant(1 : index) : i64
    %462 = llvm.mlir.constant(320 : index) : i64
    %463 = llvm.mlir.zero : !llvm.ptr
    %464 = llvm.getelementptr %463[%462] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %465 = llvm.ptrtoint %464 : !llvm.ptr to i64
    %466 = llvm.mlir.constant(64 : index) : i64
    %467 = llvm.add %465, %466 : i64
    %468 = llvm.call @malloc(%467) : (i64) -> !llvm.ptr
    %469 = llvm.ptrtoint %468 : !llvm.ptr to i64
    %470 = llvm.mlir.constant(1 : index) : i64
    %471 = llvm.sub %466, %470 : i64
    %472 = llvm.add %469, %471 : i64
    %473 = llvm.urem %472, %466 : i64
    %474 = llvm.sub %472, %473 : i64
    %475 = llvm.inttoptr %474 : i64 to !llvm.ptr
    %476 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %477 = llvm.insertvalue %468, %476[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %478 = llvm.insertvalue %475, %477[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %479 = llvm.mlir.constant(0 : index) : i64
    %480 = llvm.insertvalue %479, %478[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %481 = llvm.insertvalue %459, %480[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %482 = llvm.insertvalue %460, %481[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %483 = llvm.insertvalue %460, %482[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %484 = llvm.insertvalue %461, %483[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb52(%6 : index)
  ^bb52(%485: index):  // 2 preds: ^bb51, ^bb56
    %486 = builtin.unrealized_conversion_cast %485 : index to i64
    %487 = builtin.unrealized_conversion_cast %485 : index to i64
    %488 = llvm.icmp "slt" %487, %4 : i64
    cf.cond_br %488, ^bb53, ^bb57
  ^bb53:  // pred: ^bb52
    cf.br ^bb54(%6 : index)
  ^bb54(%489: index):  // 2 preds: ^bb53, ^bb55
    %490 = builtin.unrealized_conversion_cast %489 : index to i64
    %491 = builtin.unrealized_conversion_cast %489 : index to i64
    %492 = llvm.icmp "slt" %491, %0 : i64
    cf.cond_br %492, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %493 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %494 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %495 = llvm.getelementptr %493[%494] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %496 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %497 = llvm.mul %490, %496 overflow<nsw, nuw> : i64
    %498 = llvm.getelementptr inbounds|nuw %495[%497] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %499 = llvm.load %498 : !llvm.ptr -> f32
    %500 = llvm.extractvalue %484[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %501 = llvm.mlir.constant(10 : index) : i64
    %502 = llvm.mul %486, %501 overflow<nsw, nuw> : i64
    %503 = llvm.add %502, %490 overflow<nsw, nuw> : i64
    %504 = llvm.getelementptr inbounds|nuw %500[%503] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %499, %504 : f32, !llvm.ptr
    %505 = llvm.add %491, %3 : i64
    %506 = builtin.unrealized_conversion_cast %505 : i64 to index
    cf.br ^bb54(%506 : index)
  ^bb56:  // pred: ^bb54
    %507 = llvm.add %487, %3 : i64
    %508 = builtin.unrealized_conversion_cast %507 : i64 to index
    cf.br ^bb52(%508 : index)
  ^bb57:  // pred: ^bb52
    %509 = llvm.mlir.constant(32 : index) : i64
    %510 = llvm.mlir.constant(10 : index) : i64
    %511 = llvm.mlir.constant(1 : index) : i64
    %512 = llvm.mlir.constant(320 : index) : i64
    %513 = llvm.mlir.zero : !llvm.ptr
    %514 = llvm.getelementptr %513[%512] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %515 = llvm.ptrtoint %514 : !llvm.ptr to i64
    %516 = llvm.mlir.constant(64 : index) : i64
    %517 = llvm.add %515, %516 : i64
    %518 = llvm.call @malloc(%517) : (i64) -> !llvm.ptr
    %519 = llvm.ptrtoint %518 : !llvm.ptr to i64
    %520 = llvm.mlir.constant(1 : index) : i64
    %521 = llvm.sub %516, %520 : i64
    %522 = llvm.add %519, %521 : i64
    %523 = llvm.urem %522, %516 : i64
    %524 = llvm.sub %522, %523 : i64
    %525 = llvm.inttoptr %524 : i64 to !llvm.ptr
    %526 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %527 = llvm.insertvalue %518, %526[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %528 = llvm.insertvalue %525, %527[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %529 = llvm.mlir.constant(0 : index) : i64
    %530 = llvm.insertvalue %529, %528[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %531 = llvm.insertvalue %509, %530[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %532 = llvm.insertvalue %510, %531[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %533 = llvm.insertvalue %510, %532[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %534 = llvm.insertvalue %511, %533[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %535 = builtin.unrealized_conversion_cast %534 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x10xf32>
    cf.br ^bb58(%6 : index)
  ^bb58(%536: index):  // 2 preds: ^bb57, ^bb62
    %537 = builtin.unrealized_conversion_cast %536 : index to i64
    %538 = builtin.unrealized_conversion_cast %536 : index to i64
    %539 = llvm.icmp "slt" %538, %4 : i64
    cf.cond_br %539, ^bb59, ^bb63
  ^bb59:  // pred: ^bb58
    cf.br ^bb60(%6 : index)
  ^bb60(%540: index):  // 2 preds: ^bb59, ^bb61
    %541 = builtin.unrealized_conversion_cast %540 : index to i64
    %542 = builtin.unrealized_conversion_cast %540 : index to i64
    %543 = llvm.icmp "slt" %542, %0 : i64
    cf.cond_br %543, ^bb61, ^bb62
  ^bb61:  // pred: ^bb60
    %544 = llvm.extractvalue %411[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %545 = llvm.mlir.constant(10 : index) : i64
    %546 = llvm.mul %537, %545 overflow<nsw, nuw> : i64
    %547 = llvm.add %546, %541 overflow<nsw, nuw> : i64
    %548 = llvm.getelementptr inbounds|nuw %544[%547] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %549 = llvm.load %548 : !llvm.ptr -> f32
    %550 = llvm.extractvalue %484[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %551 = llvm.mlir.constant(10 : index) : i64
    %552 = llvm.mul %537, %551 overflow<nsw, nuw> : i64
    %553 = llvm.add %552, %541 overflow<nsw, nuw> : i64
    %554 = llvm.getelementptr inbounds|nuw %550[%553] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %555 = llvm.load %554 : !llvm.ptr -> f32
    %556 = llvm.fadd %549, %555 : f32
    %557 = llvm.extractvalue %534[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %558 = llvm.mlir.constant(10 : index) : i64
    %559 = llvm.mul %537, %558 overflow<nsw, nuw> : i64
    %560 = llvm.add %559, %541 overflow<nsw, nuw> : i64
    %561 = llvm.getelementptr inbounds|nuw %557[%560] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %556, %561 : f32, !llvm.ptr
    %562 = llvm.add %542, %3 : i64
    %563 = builtin.unrealized_conversion_cast %562 : i64 to index
    cf.br ^bb60(%563 : index)
  ^bb62:  // pred: ^bb60
    %564 = llvm.add %538, %3 : i64
    %565 = builtin.unrealized_conversion_cast %564 : i64 to index
    cf.br ^bb58(%565 : index)
  ^bb63:  // pred: ^bb58
    %566 = bufferization.to_tensor %535 : memref<32x10xf32> to tensor<32x10xf32>
    return %566 : tensor<32x10xf32>
  }
  func.func @irregular_linear_layer(%arg0: tensor<17x97xf32>, %arg1: tensor<97x53xf32>, %arg2: tensor<53xf32>) -> tensor<17x53xf32> {
    %0 = llvm.mlir.constant(97 : index) : i64
    %1 = llvm.mlir.constant(53 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(17 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %7 = bufferization.to_buffer %arg2 : tensor<53xf32> to memref<53xf32, strided<[?], offset: ?>>
    %8 = builtin.unrealized_conversion_cast %7 : memref<53xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = bufferization.to_buffer %arg1 : tensor<97x53xf32> to memref<97x53xf32, strided<[?, ?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<97x53xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = bufferization.to_buffer %arg0 : tensor<17x97xf32> to memref<17x97xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<17x97xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(17 : index) : i64
    %14 = llvm.mlir.constant(53 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(901 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.sub %20, %24 : i64
    %26 = llvm.add %23, %25 : i64
    %27 = llvm.urem %26, %20 : i64
    %28 = llvm.sub %26, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %22, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%39: index):  // 2 preds: ^bb0, ^bb8
    %40 = builtin.unrealized_conversion_cast %39 : index to i64
    %41 = builtin.unrealized_conversion_cast %39 : index to i64
    %42 = llvm.icmp "slt" %41, %3 : i64
    cf.cond_br %42, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%43: index):  // 2 preds: ^bb2, ^bb7
    %44 = builtin.unrealized_conversion_cast %43 : index to i64
    %45 = builtin.unrealized_conversion_cast %43 : index to i64
    %46 = llvm.icmp "slt" %45, %1 : i64
    cf.cond_br %46, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%47: index):  // 2 preds: ^bb4, ^bb6
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %49, %0 : i64
    cf.cond_br %50, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %40, %54 overflow<nsw, nuw> : i64
    %56 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %48, %56 overflow<nsw, nuw> : i64
    %58 = llvm.add %55, %57 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %48, %64 overflow<nsw, nuw> : i64
    %66 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %44, %66 overflow<nsw, nuw> : i64
    %68 = llvm.add %65, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds|nuw %63[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(53 : index) : i64
    %73 = llvm.mul %40, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %44 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %60, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(53 : index) : i64
    %81 = llvm.mul %40, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %44 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %79[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %83 : f32, !llvm.ptr
    %84 = llvm.add %49, %2 : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    cf.br ^bb5(%85 : index)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %45, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb3(%87 : index)
  ^bb8:  // pred: ^bb3
    %88 = llvm.add %41, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb1(%89 : index)
  ^bb9:  // pred: ^bb1
    %90 = llvm.mlir.constant(17 : index) : i64
    %91 = llvm.mlir.constant(53 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(901 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.constant(64 : index) : i64
    %98 = llvm.add %96, %97 : i64
    %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.sub %97, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.urem %103, %97 : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.insertvalue %99, %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %91, %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %92, %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%116: index):  // 2 preds: ^bb9, ^bb14
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    %118 = builtin.unrealized_conversion_cast %116 : index to i64
    %119 = llvm.icmp "slt" %118, %3 : i64
    cf.cond_br %119, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%120: index):  // 2 preds: ^bb11, ^bb13
    %121 = builtin.unrealized_conversion_cast %120 : index to i64
    %122 = builtin.unrealized_conversion_cast %120 : index to i64
    %123 = llvm.icmp "slt" %122, %1 : i64
    cf.cond_br %123, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %124 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mul %121, %127 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(53 : index) : i64
    %133 = llvm.mul %117, %132 overflow<nsw, nuw> : i64
    %134 = llvm.add %133, %121 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %130, %135 : f32, !llvm.ptr
    %136 = llvm.add %122, %2 : i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    cf.br ^bb12(%137 : index)
  ^bb14:  // pred: ^bb12
    %138 = llvm.add %118, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb10(%139 : index)
  ^bb15:  // pred: ^bb10
    %140 = llvm.mlir.constant(17 : index) : i64
    %141 = llvm.mlir.constant(53 : index) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(901 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(64 : index) : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.sub %147, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.urem %153, %147 : i64
    %155 = llvm.sub %153, %154 : i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %149, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<17x53xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%167: index):  // 2 preds: ^bb15, ^bb20
    %168 = builtin.unrealized_conversion_cast %167 : index to i64
    %169 = builtin.unrealized_conversion_cast %167 : index to i64
    %170 = llvm.icmp "slt" %169, %3 : i64
    cf.cond_br %170, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%171: index):  // 2 preds: ^bb17, ^bb19
    %172 = builtin.unrealized_conversion_cast %171 : index to i64
    %173 = builtin.unrealized_conversion_cast %171 : index to i64
    %174 = llvm.icmp "slt" %173, %1 : i64
    cf.cond_br %174, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %175 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(53 : index) : i64
    %177 = llvm.mul %168, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %172 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(53 : index) : i64
    %183 = llvm.mul %168, %182 overflow<nsw, nuw> : i64
    %184 = llvm.add %183, %172 overflow<nsw, nuw> : i64
    %185 = llvm.getelementptr inbounds|nuw %181[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %186 = llvm.load %185 : !llvm.ptr -> f32
    %187 = llvm.fadd %180, %186 : f32
    %188 = llvm.intr.maximum(%187, %6) : (f32, f32) -> f32
    %189 = llvm.extractvalue %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.mlir.constant(53 : index) : i64
    %191 = llvm.mul %168, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %172 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %189[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %188, %193 : f32, !llvm.ptr
    %194 = llvm.add %173, %2 : i64
    %195 = builtin.unrealized_conversion_cast %194 : i64 to index
    cf.br ^bb18(%195 : index)
  ^bb20:  // pred: ^bb18
    %196 = llvm.add %169, %2 : i64
    %197 = builtin.unrealized_conversion_cast %196 : i64 to index
    cf.br ^bb16(%197 : index)
  ^bb21:  // pred: ^bb16
    %198 = bufferization.to_tensor %166 : memref<17x53xf32> to tensor<17x53xf32>
    return %198 : tensor<17x53xf32>
  }
  func.func @linear_with_sigmoid(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %8 = builtin.unrealized_conversion_cast %7 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %9 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.mlir.constant(32 : index) : i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(2048 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.mlir.constant(64 : index) : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.sub %20, %24 : i64
    %26 = llvm.add %23, %25 : i64
    %27 = llvm.urem %26, %20 : i64
    %28 = llvm.sub %26, %27 : i64
    %29 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %22, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %14, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %15, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%39: index):  // 2 preds: ^bb0, ^bb8
    %40 = builtin.unrealized_conversion_cast %39 : index to i64
    %41 = builtin.unrealized_conversion_cast %39 : index to i64
    %42 = llvm.icmp "slt" %41, %3 : i64
    cf.cond_br %42, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%43: index):  // 2 preds: ^bb2, ^bb7
    %44 = builtin.unrealized_conversion_cast %43 : index to i64
    %45 = builtin.unrealized_conversion_cast %43 : index to i64
    %46 = llvm.icmp "slt" %45, %1 : i64
    cf.cond_br %46, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%47: index):  // 2 preds: ^bb4, ^bb6
    %48 = builtin.unrealized_conversion_cast %47 : index to i64
    %49 = builtin.unrealized_conversion_cast %47 : index to i64
    %50 = llvm.icmp "slt" %49, %0 : i64
    cf.cond_br %50, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %40, %54 overflow<nsw, nuw> : i64
    %56 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %48, %56 overflow<nsw, nuw> : i64
    %58 = llvm.add %55, %57 overflow<nsw, nuw> : i64
    %59 = llvm.getelementptr inbounds|nuw %53[%58] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %60 = llvm.load %59 : !llvm.ptr -> f32
    %61 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.getelementptr %61[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mul %48, %64 overflow<nsw, nuw> : i64
    %66 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %44, %66 overflow<nsw, nuw> : i64
    %68 = llvm.add %65, %67 overflow<nsw, nuw> : i64
    %69 = llvm.getelementptr inbounds|nuw %63[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.load %69 : !llvm.ptr -> f32
    %71 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.mlir.constant(64 : index) : i64
    %73 = llvm.mul %40, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %44 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %60, %70 : f32
    %78 = llvm.fadd %76, %77 : f32
    %79 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(64 : index) : i64
    %81 = llvm.mul %40, %80 overflow<nsw, nuw> : i64
    %82 = llvm.add %81, %44 overflow<nsw, nuw> : i64
    %83 = llvm.getelementptr inbounds|nuw %79[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %83 : f32, !llvm.ptr
    %84 = llvm.add %49, %2 : i64
    %85 = builtin.unrealized_conversion_cast %84 : i64 to index
    cf.br ^bb5(%85 : index)
  ^bb7:  // pred: ^bb5
    %86 = llvm.add %45, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb3(%87 : index)
  ^bb8:  // pred: ^bb3
    %88 = llvm.add %41, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb1(%89 : index)
  ^bb9:  // pred: ^bb1
    %90 = llvm.mlir.constant(32 : index) : i64
    %91 = llvm.mlir.constant(64 : index) : i64
    %92 = llvm.mlir.constant(1 : index) : i64
    %93 = llvm.mlir.constant(2048 : index) : i64
    %94 = llvm.mlir.zero : !llvm.ptr
    %95 = llvm.getelementptr %94[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.mlir.constant(64 : index) : i64
    %98 = llvm.add %96, %97 : i64
    %99 = llvm.call @malloc(%98) : (i64) -> !llvm.ptr
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.sub %97, %101 : i64
    %103 = llvm.add %100, %102 : i64
    %104 = llvm.urem %103, %97 : i64
    %105 = llvm.sub %103, %104 : i64
    %106 = llvm.inttoptr %105 : i64 to !llvm.ptr
    %107 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %108 = llvm.insertvalue %99, %107[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %106, %108[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.mlir.constant(0 : index) : i64
    %111 = llvm.insertvalue %110, %109[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %91, %112[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %92, %114[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%116: index):  // 2 preds: ^bb9, ^bb14
    %117 = builtin.unrealized_conversion_cast %116 : index to i64
    %118 = builtin.unrealized_conversion_cast %116 : index to i64
    %119 = llvm.icmp "slt" %118, %3 : i64
    cf.cond_br %119, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%120: index):  // 2 preds: ^bb11, ^bb13
    %121 = builtin.unrealized_conversion_cast %120 : index to i64
    %122 = builtin.unrealized_conversion_cast %120 : index to i64
    %123 = llvm.icmp "slt" %122, %1 : i64
    cf.cond_br %123, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %124 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %126 = llvm.getelementptr %124[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.mul %121, %127 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %126[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.load %129 : !llvm.ptr -> f32
    %131 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(64 : index) : i64
    %133 = llvm.mul %117, %132 overflow<nsw, nuw> : i64
    %134 = llvm.add %133, %121 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %130, %135 : f32, !llvm.ptr
    %136 = llvm.add %122, %2 : i64
    %137 = builtin.unrealized_conversion_cast %136 : i64 to index
    cf.br ^bb12(%137 : index)
  ^bb14:  // pred: ^bb12
    %138 = llvm.add %118, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb10(%139 : index)
  ^bb15:  // pred: ^bb10
    %140 = llvm.mlir.constant(32 : index) : i64
    %141 = llvm.mlir.constant(64 : index) : i64
    %142 = llvm.mlir.constant(1 : index) : i64
    %143 = llvm.mlir.constant(2048 : index) : i64
    %144 = llvm.mlir.zero : !llvm.ptr
    %145 = llvm.getelementptr %144[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.mlir.constant(64 : index) : i64
    %148 = llvm.add %146, %147 : i64
    %149 = llvm.call @malloc(%148) : (i64) -> !llvm.ptr
    %150 = llvm.ptrtoint %149 : !llvm.ptr to i64
    %151 = llvm.mlir.constant(1 : index) : i64
    %152 = llvm.sub %147, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.urem %153, %147 : i64
    %155 = llvm.sub %153, %154 : i64
    %156 = llvm.inttoptr %155 : i64 to !llvm.ptr
    %157 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %158 = llvm.insertvalue %149, %157[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.insertvalue %156, %158[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %160 = llvm.mlir.constant(0 : index) : i64
    %161 = llvm.insertvalue %160, %159[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %141, %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %142, %164[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = builtin.unrealized_conversion_cast %165 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%167: index):  // 2 preds: ^bb15, ^bb20
    %168 = builtin.unrealized_conversion_cast %167 : index to i64
    %169 = builtin.unrealized_conversion_cast %167 : index to i64
    %170 = llvm.icmp "slt" %169, %3 : i64
    cf.cond_br %170, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%171: index):  // 2 preds: ^bb17, ^bb19
    %172 = builtin.unrealized_conversion_cast %171 : index to i64
    %173 = builtin.unrealized_conversion_cast %171 : index to i64
    %174 = llvm.icmp "slt" %173, %1 : i64
    cf.cond_br %174, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %175 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(64 : index) : i64
    %177 = llvm.mul %168, %176 overflow<nsw, nuw> : i64
    %178 = llvm.add %177, %172 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %115[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.mlir.constant(64 : index) : i64
    %183 = llvm.mul %168, %182 overflow<nsw, nuw> : i64
    %184 = llvm.add %183, %172 overflow<nsw, nuw> : i64
    %185 = llvm.getelementptr inbounds|nuw %181[%184] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %186 = llvm.load %185 : !llvm.ptr -> f32
    %187 = llvm.fadd %180, %186 : f32
    %188 = llvm.fneg %187 : f32
    %189 = math.exp %188 : f32
    %190 = llvm.fadd %189, %6 : f32
    %191 = llvm.fdiv %6, %190 : f32
    %192 = llvm.extractvalue %165[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.mlir.constant(64 : index) : i64
    %194 = llvm.mul %168, %193 overflow<nsw, nuw> : i64
    %195 = llvm.add %194, %172 overflow<nsw, nuw> : i64
    %196 = llvm.getelementptr inbounds|nuw %192[%195] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %191, %196 : f32, !llvm.ptr
    %197 = llvm.add %173, %2 : i64
    %198 = builtin.unrealized_conversion_cast %197 : i64 to index
    cf.br ^bb18(%198 : index)
  ^bb20:  // pred: ^bb18
    %199 = llvm.add %169, %2 : i64
    %200 = builtin.unrealized_conversion_cast %199 : i64 to index
    cf.br ^bb16(%200 : index)
  ^bb21:  // pred: ^bb16
    %201 = bufferization.to_tensor %166 : memref<32x64xf32> to tensor<32x64xf32>
    return %201 : tensor<32x64xf32>
  }
  func.func @linear_with_gelu(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %7 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1.41421354 : f32) : f32
    %9 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %10 = builtin.unrealized_conversion_cast %9 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %11 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %12 = builtin.unrealized_conversion_cast %11 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %14 = builtin.unrealized_conversion_cast %13 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.mlir.constant(32 : index) : i64
    %16 = llvm.mlir.constant(64 : index) : i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.mlir.constant(2048 : index) : i64
    %19 = llvm.mlir.zero : !llvm.ptr
    %20 = llvm.getelementptr %19[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.mlir.constant(64 : index) : i64
    %23 = llvm.add %21, %22 : i64
    %24 = llvm.call @malloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.sub %22, %26 : i64
    %28 = llvm.add %25, %27 : i64
    %29 = llvm.urem %28, %22 : i64
    %30 = llvm.sub %28, %29 : i64
    %31 = llvm.inttoptr %30 : i64 to !llvm.ptr
    %32 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %24, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %15, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %16, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %16, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %17, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%41: index):  // 2 preds: ^bb0, ^bb8
    %42 = builtin.unrealized_conversion_cast %41 : index to i64
    %43 = builtin.unrealized_conversion_cast %41 : index to i64
    %44 = llvm.icmp "slt" %43, %3 : i64
    cf.cond_br %44, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%45: index):  // 2 preds: ^bb2, ^bb7
    %46 = builtin.unrealized_conversion_cast %45 : index to i64
    %47 = builtin.unrealized_conversion_cast %45 : index to i64
    %48 = llvm.icmp "slt" %47, %1 : i64
    cf.cond_br %48, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%49: index):  // 2 preds: ^bb4, ^bb6
    %50 = builtin.unrealized_conversion_cast %49 : index to i64
    %51 = builtin.unrealized_conversion_cast %49 : index to i64
    %52 = llvm.icmp "slt" %51, %0 : i64
    cf.cond_br %52, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %53 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.extractvalue %14[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.getelementptr %53[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mul %42, %56 overflow<nsw, nuw> : i64
    %58 = llvm.extractvalue %14[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mul %50, %58 overflow<nsw, nuw> : i64
    %60 = llvm.add %57, %59 overflow<nsw, nuw> : i64
    %61 = llvm.getelementptr inbounds|nuw %55[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.load %61 : !llvm.ptr -> f32
    %63 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.getelementptr %63[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %66 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mul %50, %66 overflow<nsw, nuw> : i64
    %68 = llvm.extractvalue %12[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.mul %46, %68 overflow<nsw, nuw> : i64
    %70 = llvm.add %67, %69 overflow<nsw, nuw> : i64
    %71 = llvm.getelementptr inbounds|nuw %65[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.load %71 : !llvm.ptr -> f32
    %73 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(64 : index) : i64
    %75 = llvm.mul %42, %74 overflow<nsw, nuw> : i64
    %76 = llvm.add %75, %46 overflow<nsw, nuw> : i64
    %77 = llvm.getelementptr inbounds|nuw %73[%76] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %78 = llvm.load %77 : !llvm.ptr -> f32
    %79 = llvm.fmul %62, %72 : f32
    %80 = llvm.fadd %78, %79 : f32
    %81 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mlir.constant(64 : index) : i64
    %83 = llvm.mul %42, %82 overflow<nsw, nuw> : i64
    %84 = llvm.add %83, %46 overflow<nsw, nuw> : i64
    %85 = llvm.getelementptr inbounds|nuw %81[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %80, %85 : f32, !llvm.ptr
    %86 = llvm.add %51, %2 : i64
    %87 = builtin.unrealized_conversion_cast %86 : i64 to index
    cf.br ^bb5(%87 : index)
  ^bb7:  // pred: ^bb5
    %88 = llvm.add %47, %2 : i64
    %89 = builtin.unrealized_conversion_cast %88 : i64 to index
    cf.br ^bb3(%89 : index)
  ^bb8:  // pred: ^bb3
    %90 = llvm.add %43, %2 : i64
    %91 = builtin.unrealized_conversion_cast %90 : i64 to index
    cf.br ^bb1(%91 : index)
  ^bb9:  // pred: ^bb1
    %92 = llvm.mlir.constant(32 : index) : i64
    %93 = llvm.mlir.constant(64 : index) : i64
    %94 = llvm.mlir.constant(1 : index) : i64
    %95 = llvm.mlir.constant(2048 : index) : i64
    %96 = llvm.mlir.zero : !llvm.ptr
    %97 = llvm.getelementptr %96[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.ptrtoint %97 : !llvm.ptr to i64
    %99 = llvm.mlir.constant(64 : index) : i64
    %100 = llvm.add %98, %99 : i64
    %101 = llvm.call @malloc(%100) : (i64) -> !llvm.ptr
    %102 = llvm.ptrtoint %101 : !llvm.ptr to i64
    %103 = llvm.mlir.constant(1 : index) : i64
    %104 = llvm.sub %99, %103 : i64
    %105 = llvm.add %102, %104 : i64
    %106 = llvm.urem %105, %99 : i64
    %107 = llvm.sub %105, %106 : i64
    %108 = llvm.inttoptr %107 : i64 to !llvm.ptr
    %109 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %110 = llvm.insertvalue %101, %109[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %108, %110[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.mlir.constant(0 : index) : i64
    %113 = llvm.insertvalue %112, %111[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %92, %113[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %93, %114[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.insertvalue %93, %115[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.insertvalue %94, %116[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%118: index):  // 2 preds: ^bb9, ^bb14
    %119 = builtin.unrealized_conversion_cast %118 : index to i64
    %120 = builtin.unrealized_conversion_cast %118 : index to i64
    %121 = llvm.icmp "slt" %120, %3 : i64
    cf.cond_br %121, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%122: index):  // 2 preds: ^bb11, ^bb13
    %123 = builtin.unrealized_conversion_cast %122 : index to i64
    %124 = builtin.unrealized_conversion_cast %122 : index to i64
    %125 = llvm.icmp "slt" %124, %1 : i64
    cf.cond_br %125, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %126 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.getelementptr %126[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %130 = llvm.mul %123, %129 overflow<nsw, nuw> : i64
    %131 = llvm.getelementptr inbounds|nuw %128[%130] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %132 = llvm.load %131 : !llvm.ptr -> f32
    %133 = llvm.extractvalue %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.constant(64 : index) : i64
    %135 = llvm.mul %119, %134 overflow<nsw, nuw> : i64
    %136 = llvm.add %135, %123 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %133[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %132, %137 : f32, !llvm.ptr
    %138 = llvm.add %124, %2 : i64
    %139 = builtin.unrealized_conversion_cast %138 : i64 to index
    cf.br ^bb12(%139 : index)
  ^bb14:  // pred: ^bb12
    %140 = llvm.add %120, %2 : i64
    %141 = builtin.unrealized_conversion_cast %140 : i64 to index
    cf.br ^bb10(%141 : index)
  ^bb15:  // pred: ^bb10
    %142 = llvm.mlir.constant(32 : index) : i64
    %143 = llvm.mlir.constant(64 : index) : i64
    %144 = llvm.mlir.constant(1 : index) : i64
    %145 = llvm.mlir.constant(2048 : index) : i64
    %146 = llvm.mlir.zero : !llvm.ptr
    %147 = llvm.getelementptr %146[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %148 = llvm.ptrtoint %147 : !llvm.ptr to i64
    %149 = llvm.mlir.constant(64 : index) : i64
    %150 = llvm.add %148, %149 : i64
    %151 = llvm.call @malloc(%150) : (i64) -> !llvm.ptr
    %152 = llvm.ptrtoint %151 : !llvm.ptr to i64
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.sub %149, %153 : i64
    %155 = llvm.add %152, %154 : i64
    %156 = llvm.urem %155, %149 : i64
    %157 = llvm.sub %155, %156 : i64
    %158 = llvm.inttoptr %157 : i64 to !llvm.ptr
    %159 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %160 = llvm.insertvalue %151, %159[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %158, %160[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.mlir.constant(0 : index) : i64
    %163 = llvm.insertvalue %162, %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %142, %163[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = llvm.insertvalue %143, %164[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.insertvalue %143, %165[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.insertvalue %144, %166[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %168 = builtin.unrealized_conversion_cast %167 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%169: index):  // 2 preds: ^bb15, ^bb20
    %170 = builtin.unrealized_conversion_cast %169 : index to i64
    %171 = builtin.unrealized_conversion_cast %169 : index to i64
    %172 = llvm.icmp "slt" %171, %3 : i64
    cf.cond_br %172, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%173: index):  // 2 preds: ^bb17, ^bb19
    %174 = builtin.unrealized_conversion_cast %173 : index to i64
    %175 = builtin.unrealized_conversion_cast %173 : index to i64
    %176 = llvm.icmp "slt" %175, %1 : i64
    cf.cond_br %176, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %177 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = llvm.mlir.constant(64 : index) : i64
    %179 = llvm.mul %170, %178 overflow<nsw, nuw> : i64
    %180 = llvm.add %179, %174 overflow<nsw, nuw> : i64
    %181 = llvm.getelementptr inbounds|nuw %177[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %182 = llvm.load %181 : !llvm.ptr -> f32
    %183 = llvm.extractvalue %117[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %184 = llvm.mlir.constant(64 : index) : i64
    %185 = llvm.mul %170, %184 overflow<nsw, nuw> : i64
    %186 = llvm.add %185, %174 overflow<nsw, nuw> : i64
    %187 = llvm.getelementptr inbounds|nuw %183[%186] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %188 = llvm.load %187 : !llvm.ptr -> f32
    %189 = llvm.fadd %182, %188 : f32
    %190 = llvm.fdiv %189, %8 : f32
    %191 = math.erf %190 : f32
    %192 = llvm.fadd %191, %7 : f32
    %193 = llvm.fmul %192, %6 : f32
    %194 = llvm.fmul %189, %193 : f32
    %195 = llvm.extractvalue %167[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %196 = llvm.mlir.constant(64 : index) : i64
    %197 = llvm.mul %170, %196 overflow<nsw, nuw> : i64
    %198 = llvm.add %197, %174 overflow<nsw, nuw> : i64
    %199 = llvm.getelementptr inbounds|nuw %195[%198] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %194, %199 : f32, !llvm.ptr
    %200 = llvm.add %175, %2 : i64
    %201 = builtin.unrealized_conversion_cast %200 : i64 to index
    cf.br ^bb18(%201 : index)
  ^bb20:  // pred: ^bb18
    %202 = llvm.add %171, %2 : i64
    %203 = builtin.unrealized_conversion_cast %202 : i64 to index
    cf.br ^bb16(%203 : index)
  ^bb21:  // pred: ^bb16
    %204 = bufferization.to_tensor %168 : memref<32x64xf32> to tensor<32x64xf32>
    return %204 : tensor<32x64xf32>
  }
  func.func @linear_with_tanh(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %7 = builtin.unrealized_conversion_cast %6 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %9 = builtin.unrealized_conversion_cast %8 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %11 = builtin.unrealized_conversion_cast %10 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.mlir.constant(32 : index) : i64
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(2048 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.mlir.constant(64 : index) : i64
    %20 = llvm.add %18, %19 : i64
    %21 = llvm.call @malloc(%20) : (i64) -> !llvm.ptr
    %22 = llvm.ptrtoint %21 : !llvm.ptr to i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.sub %19, %23 : i64
    %25 = llvm.add %22, %24 : i64
    %26 = llvm.urem %25, %19 : i64
    %27 = llvm.sub %25, %26 : i64
    %28 = llvm.inttoptr %27 : i64 to !llvm.ptr
    %29 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %21, %29[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %28, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.mlir.constant(0 : index) : i64
    %33 = llvm.insertvalue %32, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %12, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %13, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %13, %35[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %14, %36[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb1(%5 : index)
  ^bb1(%38: index):  // 2 preds: ^bb0, ^bb8
    %39 = builtin.unrealized_conversion_cast %38 : index to i64
    %40 = builtin.unrealized_conversion_cast %38 : index to i64
    %41 = llvm.icmp "slt" %40, %3 : i64
    cf.cond_br %41, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%5 : index)
  ^bb3(%42: index):  // 2 preds: ^bb2, ^bb7
    %43 = builtin.unrealized_conversion_cast %42 : index to i64
    %44 = builtin.unrealized_conversion_cast %42 : index to i64
    %45 = llvm.icmp "slt" %44, %1 : i64
    cf.cond_br %45, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%5 : index)
  ^bb5(%46: index):  // 2 preds: ^bb4, ^bb6
    %47 = builtin.unrealized_conversion_cast %46 : index to i64
    %48 = builtin.unrealized_conversion_cast %46 : index to i64
    %49 = llvm.icmp "slt" %48, %0 : i64
    cf.cond_br %49, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %50 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.extractvalue %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.getelementptr %50[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.extractvalue %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mul %39, %53 overflow<nsw, nuw> : i64
    %55 = llvm.extractvalue %11[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mul %47, %55 overflow<nsw, nuw> : i64
    %57 = llvm.add %54, %56 overflow<nsw, nuw> : i64
    %58 = llvm.getelementptr inbounds|nuw %52[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.load %58 : !llvm.ptr -> f32
    %60 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %9[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mul %47, %63 overflow<nsw, nuw> : i64
    %65 = llvm.extractvalue %9[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mul %43, %65 overflow<nsw, nuw> : i64
    %67 = llvm.add %64, %66 overflow<nsw, nuw> : i64
    %68 = llvm.getelementptr inbounds|nuw %62[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(64 : index) : i64
    %72 = llvm.mul %39, %71 overflow<nsw, nuw> : i64
    %73 = llvm.add %72, %43 overflow<nsw, nuw> : i64
    %74 = llvm.getelementptr inbounds|nuw %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 : !llvm.ptr -> f32
    %76 = llvm.fmul %59, %69 : f32
    %77 = llvm.fadd %75, %76 : f32
    %78 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(64 : index) : i64
    %80 = llvm.mul %39, %79 overflow<nsw, nuw> : i64
    %81 = llvm.add %80, %43 overflow<nsw, nuw> : i64
    %82 = llvm.getelementptr inbounds|nuw %78[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %82 : f32, !llvm.ptr
    %83 = llvm.add %48, %2 : i64
    %84 = builtin.unrealized_conversion_cast %83 : i64 to index
    cf.br ^bb5(%84 : index)
  ^bb7:  // pred: ^bb5
    %85 = llvm.add %44, %2 : i64
    %86 = builtin.unrealized_conversion_cast %85 : i64 to index
    cf.br ^bb3(%86 : index)
  ^bb8:  // pred: ^bb3
    %87 = llvm.add %40, %2 : i64
    %88 = builtin.unrealized_conversion_cast %87 : i64 to index
    cf.br ^bb1(%88 : index)
  ^bb9:  // pred: ^bb1
    %89 = llvm.mlir.constant(32 : index) : i64
    %90 = llvm.mlir.constant(64 : index) : i64
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.mlir.constant(2048 : index) : i64
    %93 = llvm.mlir.zero : !llvm.ptr
    %94 = llvm.getelementptr %93[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.mlir.constant(64 : index) : i64
    %97 = llvm.add %95, %96 : i64
    %98 = llvm.call @malloc(%97) : (i64) -> !llvm.ptr
    %99 = llvm.ptrtoint %98 : !llvm.ptr to i64
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.sub %96, %100 : i64
    %102 = llvm.add %99, %101 : i64
    %103 = llvm.urem %102, %96 : i64
    %104 = llvm.sub %102, %103 : i64
    %105 = llvm.inttoptr %104 : i64 to !llvm.ptr
    %106 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %107 = llvm.insertvalue %98, %106[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.insertvalue %105, %107[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.mlir.constant(0 : index) : i64
    %110 = llvm.insertvalue %109, %108[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %89, %110[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %90, %111[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %90, %112[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.insertvalue %91, %113[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    cf.br ^bb10(%5 : index)
  ^bb10(%115: index):  // 2 preds: ^bb9, ^bb14
    %116 = builtin.unrealized_conversion_cast %115 : index to i64
    %117 = builtin.unrealized_conversion_cast %115 : index to i64
    %118 = llvm.icmp "slt" %117, %3 : i64
    cf.cond_br %118, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    cf.br ^bb12(%5 : index)
  ^bb12(%119: index):  // 2 preds: ^bb11, ^bb13
    %120 = builtin.unrealized_conversion_cast %119 : index to i64
    %121 = builtin.unrealized_conversion_cast %119 : index to i64
    %122 = llvm.icmp "slt" %121, %1 : i64
    cf.cond_br %122, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %123 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.getelementptr %123[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.mul %120, %126 overflow<nsw, nuw> : i64
    %128 = llvm.getelementptr inbounds|nuw %125[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
    %130 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(64 : index) : i64
    %132 = llvm.mul %116, %131 overflow<nsw, nuw> : i64
    %133 = llvm.add %132, %120 overflow<nsw, nuw> : i64
    %134 = llvm.getelementptr inbounds|nuw %130[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %129, %134 : f32, !llvm.ptr
    %135 = llvm.add %121, %2 : i64
    %136 = builtin.unrealized_conversion_cast %135 : i64 to index
    cf.br ^bb12(%136 : index)
  ^bb14:  // pred: ^bb12
    %137 = llvm.add %117, %2 : i64
    %138 = builtin.unrealized_conversion_cast %137 : i64 to index
    cf.br ^bb10(%138 : index)
  ^bb15:  // pred: ^bb10
    %139 = llvm.mlir.constant(32 : index) : i64
    %140 = llvm.mlir.constant(64 : index) : i64
    %141 = llvm.mlir.constant(1 : index) : i64
    %142 = llvm.mlir.constant(2048 : index) : i64
    %143 = llvm.mlir.zero : !llvm.ptr
    %144 = llvm.getelementptr %143[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.ptrtoint %144 : !llvm.ptr to i64
    %146 = llvm.mlir.constant(64 : index) : i64
    %147 = llvm.add %145, %146 : i64
    %148 = llvm.call @malloc(%147) : (i64) -> !llvm.ptr
    %149 = llvm.ptrtoint %148 : !llvm.ptr to i64
    %150 = llvm.mlir.constant(1 : index) : i64
    %151 = llvm.sub %146, %150 : i64
    %152 = llvm.add %149, %151 : i64
    %153 = llvm.urem %152, %146 : i64
    %154 = llvm.sub %152, %153 : i64
    %155 = llvm.inttoptr %154 : i64 to !llvm.ptr
    %156 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %157 = llvm.insertvalue %148, %156[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %158 = llvm.insertvalue %155, %157[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %159 = llvm.mlir.constant(0 : index) : i64
    %160 = llvm.insertvalue %159, %158[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.insertvalue %139, %160[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %162 = llvm.insertvalue %140, %161[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %163 = llvm.insertvalue %140, %162[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %164 = llvm.insertvalue %141, %163[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %165 = builtin.unrealized_conversion_cast %164 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb16(%5 : index)
  ^bb16(%166: index):  // 2 preds: ^bb15, ^bb20
    %167 = builtin.unrealized_conversion_cast %166 : index to i64
    %168 = builtin.unrealized_conversion_cast %166 : index to i64
    %169 = llvm.icmp "slt" %168, %3 : i64
    cf.cond_br %169, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%5 : index)
  ^bb18(%170: index):  // 2 preds: ^bb17, ^bb19
    %171 = builtin.unrealized_conversion_cast %170 : index to i64
    %172 = builtin.unrealized_conversion_cast %170 : index to i64
    %173 = llvm.icmp "slt" %172, %1 : i64
    cf.cond_br %173, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %174 = llvm.extractvalue %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.mlir.constant(64 : index) : i64
    %176 = llvm.mul %167, %175 overflow<nsw, nuw> : i64
    %177 = llvm.add %176, %171 overflow<nsw, nuw> : i64
    %178 = llvm.getelementptr inbounds|nuw %174[%177] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %179 = llvm.load %178 : !llvm.ptr -> f32
    %180 = llvm.extractvalue %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.mlir.constant(64 : index) : i64
    %182 = llvm.mul %167, %181 overflow<nsw, nuw> : i64
    %183 = llvm.add %182, %171 overflow<nsw, nuw> : i64
    %184 = llvm.getelementptr inbounds|nuw %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %185 = llvm.load %184 : !llvm.ptr -> f32
    %186 = llvm.fadd %179, %185 : f32
    %187 = math.tanh %186 : f32
    %188 = llvm.extractvalue %164[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %189 = llvm.mlir.constant(64 : index) : i64
    %190 = llvm.mul %167, %189 overflow<nsw, nuw> : i64
    %191 = llvm.add %190, %171 overflow<nsw, nuw> : i64
    %192 = llvm.getelementptr inbounds|nuw %188[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %187, %192 : f32, !llvm.ptr
    %193 = llvm.add %172, %2 : i64
    %194 = builtin.unrealized_conversion_cast %193 : i64 to index
    cf.br ^bb18(%194 : index)
  ^bb20:  // pred: ^bb18
    %195 = llvm.add %168, %2 : i64
    %196 = builtin.unrealized_conversion_cast %195 : i64 to index
    cf.br ^bb16(%196 : index)
  ^bb21:  // pred: ^bb16
    %197 = bufferization.to_tensor %165 : memref<32x64xf32> to tensor<32x64xf32>
    return %197 : tensor<32x64xf32>
  }
}

