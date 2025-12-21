module {
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @linear_layer_f32(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(32 : index) : i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(2048 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.intr.maximum(%139, %7) : (f32, f32) -> f32
    %141 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(64 : index) : i64
    %143 = llvm.mul %85, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %79 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %140, %145 : f32, !llvm.ptr
    %146 = llvm.add %74, %2 : i64
    %147 = builtin.unrealized_conversion_cast %146 : i64 to index
    cf.br ^bb1(%147 : index)
  ^bb6:  // pred: ^bb1
    %148 = bufferization.to_tensor %72 : memref<32x64xf32> to tensor<32x64xf32>
    return %148 : tensor<32x64xf32>
  }
  func.func @large_linear_layer_f32(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256xf32>) -> tensor<256x256xf32> {
    %0 = llvm.mlir.constant(512 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(65536 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(256 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<256xf32> to memref<256xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<256xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<512x256xf32> to memref<512x256xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<512x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<256x512xf32> to memref<256x512xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<256x512xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(256 : index) : i64
    %47 = llvm.mlir.constant(256 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(65536 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<256x256xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.intr.maximum(%139, %7) : (f32, f32) -> f32
    %141 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(256 : index) : i64
    %143 = llvm.mul %85, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %79 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %140, %145 : f32, !llvm.ptr
    %146 = llvm.add %74, %2 : i64
    %147 = builtin.unrealized_conversion_cast %146 : i64 to index
    cf.br ^bb1(%147 : index)
  ^bb6:  // pred: ^bb1
    %148 = bufferization.to_tensor %72 : memref<256x256xf32> to tensor<256x256xf32>
    return %148 : tensor<256x256xf32>
  }
  func.func @small_batch_linear_f32(%arg0: tensor<1x784xf32>, %arg1: tensor<784x128xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = llvm.mlir.constant(784 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(128 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = builtin.unrealized_conversion_cast %5 : i64 to index
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %7, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %8, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.zero : !llvm.ptr
    %27 = llvm.getelementptr %26[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %23, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = bufferization.to_buffer %arg2 : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<128xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = bufferization.to_buffer %arg1 : tensor<784x128xf32> to memref<784x128xf32, strided<[?, ?], offset: ?>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<784x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = bufferization.to_buffer %arg0 : tensor<1x784xf32> to memref<1x784xf32, strided<[?, ?], offset: ?>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<1x784xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(128 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.mlir.constant(128 : index) : i64
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.mlir.constant(64 : index) : i64
    %53 = llvm.add %51, %52 : i64
    %54 = llvm.call @malloc(%53) : (i64) -> !llvm.ptr
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.sub %52, %56 : i64
    %58 = llvm.add %55, %57 : i64
    %59 = llvm.urem %58, %52 : i64
    %60 = llvm.sub %58, %59 : i64
    %61 = llvm.inttoptr %60 : i64 to !llvm.ptr
    %62 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.insertvalue %54, %62[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %61, %63[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mlir.constant(0 : index) : i64
    %66 = llvm.insertvalue %65, %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %45, %66[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %46, %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = builtin.unrealized_conversion_cast %70 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x128xf32>
    cf.br ^bb1(%6 : index)
  ^bb1(%72: index):  // 2 preds: ^bb0, ^bb5
    %73 = builtin.unrealized_conversion_cast %72 : index to i64
    %74 = llvm.icmp "slt" %73, %2 : i64
    cf.cond_br %74, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %75 = llvm.srem %73, %2 : i64
    %76 = llvm.icmp "slt" %75, %5 : i64
    %77 = llvm.add %75, %2 : i64
    %78 = llvm.select %76, %77, %75 : i1, i64
    %79 = llvm.icmp "slt" %73, %5 : i64
    %80 = llvm.sub %1, %73 : i64
    %81 = llvm.select %79, %80, %73 : i1, i64
    %82 = llvm.sdiv %81, %2 : i64
    %83 = llvm.sub %1, %82 : i64
    %84 = llvm.select %79, %83, %82 : i1, i64
    cf.br ^bb3(%6 : index)
  ^bb3(%85: index):  // 2 preds: ^bb2, ^bb4
    %86 = builtin.unrealized_conversion_cast %85 : index to i64
    %87 = builtin.unrealized_conversion_cast %85 : index to i64
    %88 = llvm.icmp "slt" %87, %0 : i64
    cf.cond_br %88, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %89 = llvm.extractvalue %44[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.extractvalue %44[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.extractvalue %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mul %5, %92 overflow<nsw, nuw> : i64
    %94 = llvm.extractvalue %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mul %86, %94 overflow<nsw, nuw> : i64
    %96 = llvm.add %93, %95 overflow<nsw, nuw> : i64
    %97 = llvm.getelementptr inbounds|nuw %91[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 : !llvm.ptr -> f32
    %99 = llvm.extractvalue %42[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %42[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.getelementptr %99[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.extractvalue %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mul %86, %102 overflow<nsw, nuw> : i64
    %104 = llvm.extractvalue %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.mul %78, %104 overflow<nsw, nuw> : i64
    %106 = llvm.add %103, %105 overflow<nsw, nuw> : i64
    %107 = llvm.getelementptr inbounds|nuw %101[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.load %107 : !llvm.ptr -> f32
    %109 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %109[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.fmul %98, %108 : f32
    %114 = llvm.fadd %112, %113 : f32
    %115 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %117 = llvm.getelementptr inbounds|nuw %115[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %114, %117 : f32, !llvm.ptr
    %118 = llvm.add %87, %3 : i64
    %119 = builtin.unrealized_conversion_cast %118 : i64 to index
    cf.br ^bb3(%119 : index)
  ^bb5:  // pred: ^bb3
    %120 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.getelementptr %120[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %123 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.mul %78, %123 overflow<nsw, nuw> : i64
    %125 = llvm.getelementptr inbounds|nuw %122[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.load %125 : !llvm.ptr -> f32
    %127 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %127[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %126, %129 : f32, !llvm.ptr
    %130 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.add %84, %5 overflow<nsw, nuw> : i64
    %132 = llvm.getelementptr inbounds|nuw %130[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %133 = llvm.load %132 : !llvm.ptr -> f32
    %134 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.add %84, %5 overflow<nsw, nuw> : i64
    %136 = llvm.getelementptr inbounds|nuw %134[%135] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %137 = llvm.load %136 : !llvm.ptr -> f32
    %138 = llvm.fadd %133, %137 : f32
    %139 = llvm.intr.maximum(%138, %4) : (f32, f32) -> f32
    %140 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(128 : index) : i64
    %142 = llvm.mul %84, %141 overflow<nsw, nuw> : i64
    %143 = llvm.add %142, %78 overflow<nsw, nuw> : i64
    %144 = llvm.getelementptr inbounds|nuw %140[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %139, %144 : f32, !llvm.ptr
    %145 = llvm.add %73, %3 : i64
    %146 = builtin.unrealized_conversion_cast %145 : i64 to index
    cf.br ^bb1(%146 : index)
  ^bb6:  // pred: ^bb1
    %147 = bufferization.to_tensor %71 : memref<1x128xf32> to tensor<1x128xf32>
    return %147 : tensor<1x128xf32>
  }
  func.func @wide_linear_layer_f32(%arg0: tensor<64x256xf32>, %arg1: tensor<256x1024xf32>, %arg2: tensor<1024xf32>) -> tensor<64x1024xf32> {
    %0 = llvm.mlir.constant(256 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(65536 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(1024 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<1024xf32> to memref<1024xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<1024xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<256x1024xf32> to memref<256x1024xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<256x1024xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<64x256xf32> to memref<64x256xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<64x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(64 : index) : i64
    %47 = llvm.mlir.constant(1024 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(65536 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<64x1024xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.intr.maximum(%139, %7) : (f32, f32) -> f32
    %141 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(1024 : index) : i64
    %143 = llvm.mul %85, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %79 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %140, %145 : f32, !llvm.ptr
    %146 = llvm.add %74, %2 : i64
    %147 = builtin.unrealized_conversion_cast %146 : i64 to index
    cf.br ^bb1(%147 : index)
  ^bb6:  // pred: ^bb1
    %148 = bufferization.to_tensor %72 : memref<64x1024xf32> to tensor<64x1024xf32>
    return %148 : tensor<64x1024xf32>
  }
  func.func @narrow_linear_layer_f32(%arg0: tensor<128x1024xf32>, %arg1: tensor<1024x64xf32>, %arg2: tensor<64xf32>) -> tensor<128x64xf32> {
    %0 = llvm.mlir.constant(1024 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(8192 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<1024x64xf32> to memref<1024x64xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<1024x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<128x1024xf32> to memref<128x1024xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<128x1024xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(128 : index) : i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(8192 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<128x64xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.intr.maximum(%139, %7) : (f32, f32) -> f32
    %141 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(64 : index) : i64
    %143 = llvm.mul %85, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %79 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %140, %145 : f32, !llvm.ptr
    %146 = llvm.add %74, %2 : i64
    %147 = builtin.unrealized_conversion_cast %146 : i64 to index
    cf.br ^bb1(%147 : index)
  ^bb6:  // pred: ^bb1
    %148 = bufferization.to_tensor %72 : memref<128x64xf32> to tensor<128x64xf32>
    return %148 : tensor<128x64xf32>
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
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(128 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(128 : index) : i64
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %12[%11] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.ptrtoint %13 : !llvm.ptr to i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %15, %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %15, %17[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.insertvalue %19, %18[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %8, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %9, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %10, %23[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.zero : !llvm.ptr
    %29 = llvm.getelementptr %28[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr
    %32 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %31, %32[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.insertvalue %31, %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(0 : index) : i64
    %36 = llvm.insertvalue %35, %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %26, %37[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %27, %39[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.mlir.zero : !llvm.ptr
    %45 = llvm.getelementptr %44[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.call @malloc(%46) : (i64) -> !llvm.ptr
    %48 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %41, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %42, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %42, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %43, %55[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.mlir.constant(1 : index) : i64
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.mlir.zero : !llvm.ptr
    %61 = llvm.getelementptr %60[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.call @malloc(%62) : (i64) -> !llvm.ptr
    %64 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %65 = llvm.insertvalue %63, %64[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %63, %65[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.mlir.constant(0 : index) : i64
    %68 = llvm.insertvalue %67, %66[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %57, %68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %58, %69[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %58, %70[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %59, %71[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.constant(1 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    %75 = llvm.mlir.constant(1 : index) : i64
    %76 = llvm.mlir.zero : !llvm.ptr
    %77 = llvm.getelementptr %76[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.call @malloc(%78) : (i64) -> !llvm.ptr
    %80 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.insertvalue %79, %80[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.insertvalue %79, %81[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.mlir.constant(0 : index) : i64
    %84 = llvm.insertvalue %83, %82[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.insertvalue %73, %84[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.insertvalue %74, %85[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %74, %86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %75, %87[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.mlir.constant(1 : index) : i64
    %90 = llvm.mlir.constant(256 : index) : i64
    %91 = llvm.mlir.constant(1 : index) : i64
    %92 = llvm.mlir.constant(256 : index) : i64
    %93 = llvm.mlir.zero : !llvm.ptr
    %94 = llvm.getelementptr %93[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.ptrtoint %94 : !llvm.ptr to i64
    %96 = llvm.call @malloc(%95) : (i64) -> !llvm.ptr
    %97 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %98 = llvm.insertvalue %96, %97[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %96, %98[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.mlir.constant(0 : index) : i64
    %101 = llvm.insertvalue %100, %99[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.insertvalue %89, %101[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.insertvalue %90, %102[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.insertvalue %90, %103[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.insertvalue %91, %104[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mlir.constant(1 : index) : i64
    %107 = llvm.mlir.constant(1 : index) : i64
    %108 = llvm.mlir.constant(1 : index) : i64
    %109 = llvm.mlir.zero : !llvm.ptr
    %110 = llvm.getelementptr %109[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %111 = llvm.ptrtoint %110 : !llvm.ptr to i64
    %112 = llvm.call @malloc(%111) : (i64) -> !llvm.ptr
    %113 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %114 = llvm.insertvalue %112, %113[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %115 = llvm.insertvalue %112, %114[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.mlir.constant(0 : index) : i64
    %117 = llvm.insertvalue %116, %115[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.insertvalue %106, %117[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.insertvalue %107, %118[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = llvm.insertvalue %107, %119[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.insertvalue %108, %120[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %122 = llvm.mlir.constant(1 : index) : i64
    %123 = llvm.mlir.constant(1 : index) : i64
    %124 = llvm.mlir.constant(1 : index) : i64
    %125 = llvm.mlir.zero : !llvm.ptr
    %126 = llvm.getelementptr %125[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.ptrtoint %126 : !llvm.ptr to i64
    %128 = llvm.call @malloc(%127) : (i64) -> !llvm.ptr
    %129 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.insertvalue %128, %129[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %128, %130[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.mlir.constant(0 : index) : i64
    %133 = llvm.insertvalue %132, %131[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.insertvalue %122, %133[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.insertvalue %123, %134[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %123, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.insertvalue %124, %136[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = bufferization.to_buffer %arg6 : tensor<10xf32> to memref<10xf32, strided<[?], offset: ?>>
    %139 = builtin.unrealized_conversion_cast %138 : memref<10xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %140 = bufferization.to_buffer %arg5 : tensor<128x10xf32> to memref<128x10xf32, strided<[?, ?], offset: ?>>
    %141 = builtin.unrealized_conversion_cast %140 : memref<128x10xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %142 = bufferization.to_buffer %arg4 : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
    %143 = builtin.unrealized_conversion_cast %142 : memref<128xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %144 = bufferization.to_buffer %arg3 : tensor<256x128xf32> to memref<256x128xf32, strided<[?, ?], offset: ?>>
    %145 = builtin.unrealized_conversion_cast %144 : memref<256x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %146 = bufferization.to_buffer %arg2 : tensor<256xf32> to memref<256xf32, strided<[?], offset: ?>>
    %147 = builtin.unrealized_conversion_cast %146 : memref<256xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %148 = bufferization.to_buffer %arg1 : tensor<128x256xf32> to memref<128x256xf32, strided<[?, ?], offset: ?>>
    %149 = builtin.unrealized_conversion_cast %148 : memref<128x256xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %150 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %151 = builtin.unrealized_conversion_cast %150 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %152 = llvm.mlir.constant(32 : index) : i64
    %153 = llvm.mlir.constant(10 : index) : i64
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.mlir.constant(320 : index) : i64
    %156 = llvm.mlir.zero : !llvm.ptr
    %157 = llvm.getelementptr %156[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %158 = llvm.ptrtoint %157 : !llvm.ptr to i64
    %159 = llvm.mlir.constant(64 : index) : i64
    %160 = llvm.add %158, %159 : i64
    %161 = llvm.call @malloc(%160) : (i64) -> !llvm.ptr
    %162 = llvm.ptrtoint %161 : !llvm.ptr to i64
    %163 = llvm.mlir.constant(1 : index) : i64
    %164 = llvm.sub %159, %163 : i64
    %165 = llvm.add %162, %164 : i64
    %166 = llvm.urem %165, %159 : i64
    %167 = llvm.sub %165, %166 : i64
    %168 = llvm.inttoptr %167 : i64 to !llvm.ptr
    %169 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %170 = llvm.insertvalue %161, %169[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.insertvalue %168, %170[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %172 = llvm.mlir.constant(0 : index) : i64
    %173 = llvm.insertvalue %172, %171[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %174 = llvm.insertvalue %152, %173[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %175 = llvm.insertvalue %153, %174[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.insertvalue %153, %175[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %177 = llvm.insertvalue %154, %176[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %178 = builtin.unrealized_conversion_cast %177 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x10xf32>
    cf.br ^bb1(%6 : index)
  ^bb1(%179: index):  // 2 preds: ^bb0, ^bb20
    %180 = builtin.unrealized_conversion_cast %179 : index to i64
    %181 = builtin.unrealized_conversion_cast %179 : index to i64
    %182 = llvm.icmp "slt" %181, %4 : i64
    cf.cond_br %182, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%6 : index)
  ^bb3(%183: index):  // 2 preds: ^bb2, ^bb7
    %184 = builtin.unrealized_conversion_cast %183 : index to i64
    %185 = builtin.unrealized_conversion_cast %183 : index to i64
    %186 = llvm.icmp "slt" %185, %2 : i64
    cf.cond_br %186, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%6 : index)
  ^bb5(%187: index):  // 2 preds: ^bb4, ^bb6
    %188 = builtin.unrealized_conversion_cast %187 : index to i64
    %189 = builtin.unrealized_conversion_cast %187 : index to i64
    %190 = llvm.icmp "slt" %189, %1 : i64
    cf.cond_br %190, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %191 = llvm.extractvalue %151[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %192 = llvm.extractvalue %151[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %193 = llvm.getelementptr %191[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %194 = llvm.extractvalue %151[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %195 = llvm.mul %180, %194 overflow<nsw, nuw> : i64
    %196 = llvm.extractvalue %151[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %197 = llvm.mul %188, %196 overflow<nsw, nuw> : i64
    %198 = llvm.add %195, %197 overflow<nsw, nuw> : i64
    %199 = llvm.getelementptr inbounds|nuw %193[%198] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %200 = llvm.load %199 : !llvm.ptr -> f32
    %201 = llvm.extractvalue %149[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %202 = llvm.extractvalue %149[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %203 = llvm.getelementptr %201[%202] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %204 = llvm.extractvalue %149[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %205 = llvm.mul %188, %204 overflow<nsw, nuw> : i64
    %206 = llvm.extractvalue %149[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %207 = llvm.mul %184, %206 overflow<nsw, nuw> : i64
    %208 = llvm.add %205, %207 overflow<nsw, nuw> : i64
    %209 = llvm.getelementptr inbounds|nuw %203[%208] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %210 = llvm.load %209 : !llvm.ptr -> f32
    %211 = llvm.extractvalue %72[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %212 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %213 = llvm.getelementptr inbounds|nuw %211[%212] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %214 = llvm.load %213 : !llvm.ptr -> f32
    %215 = llvm.fmul %200, %210 : f32
    %216 = llvm.fadd %214, %215 : f32
    %217 = llvm.extractvalue %72[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %218 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %219 = llvm.getelementptr inbounds|nuw %217[%218] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %216, %219 : f32, !llvm.ptr
    %220 = llvm.add %189, %3 : i64
    %221 = builtin.unrealized_conversion_cast %220 : i64 to index
    cf.br ^bb5(%221 : index)
  ^bb7:  // pred: ^bb5
    %222 = llvm.extractvalue %147[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %223 = llvm.extractvalue %147[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %224 = llvm.getelementptr %222[%223] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %225 = llvm.extractvalue %147[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %226 = llvm.mul %184, %225 overflow<nsw, nuw> : i64
    %227 = llvm.getelementptr inbounds|nuw %224[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %228 = llvm.load %227 : !llvm.ptr -> f32
    %229 = llvm.extractvalue %88[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %231 = llvm.getelementptr inbounds|nuw %229[%230] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %228, %231 : f32, !llvm.ptr
    %232 = llvm.extractvalue %72[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %233 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %234 = llvm.getelementptr inbounds|nuw %232[%233] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %235 = llvm.load %234 : !llvm.ptr -> f32
    %236 = llvm.extractvalue %88[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %238 = llvm.getelementptr inbounds|nuw %236[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %239 = llvm.load %238 : !llvm.ptr -> f32
    %240 = llvm.fadd %235, %239 : f32
    %241 = llvm.intr.maximum(%240, %7) : (f32, f32) -> f32
    %242 = llvm.extractvalue %105[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %243 = llvm.mlir.constant(256 : index) : i64
    %244 = llvm.mul %5, %243 overflow<nsw, nuw> : i64
    %245 = llvm.add %244, %184 overflow<nsw, nuw> : i64
    %246 = llvm.getelementptr inbounds|nuw %242[%245] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %241, %246 : f32, !llvm.ptr
    %247 = llvm.add %185, %3 : i64
    %248 = builtin.unrealized_conversion_cast %247 : i64 to index
    cf.br ^bb3(%248 : index)
  ^bb8:  // pred: ^bb3
    cf.br ^bb9(%6 : index)
  ^bb9(%249: index):  // 2 preds: ^bb8, ^bb13
    %250 = builtin.unrealized_conversion_cast %249 : index to i64
    %251 = builtin.unrealized_conversion_cast %249 : index to i64
    %252 = llvm.icmp "slt" %251, %1 : i64
    cf.cond_br %252, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    cf.br ^bb11(%6 : index)
  ^bb11(%253: index):  // 2 preds: ^bb10, ^bb12
    %254 = builtin.unrealized_conversion_cast %253 : index to i64
    %255 = builtin.unrealized_conversion_cast %253 : index to i64
    %256 = llvm.icmp "slt" %255, %2 : i64
    cf.cond_br %256, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %257 = llvm.extractvalue %105[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %258 = llvm.mlir.constant(256 : index) : i64
    %259 = llvm.mul %5, %258 overflow<nsw, nuw> : i64
    %260 = llvm.add %259, %254 overflow<nsw, nuw> : i64
    %261 = llvm.getelementptr inbounds|nuw %257[%260] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %262 = llvm.load %261 : !llvm.ptr -> f32
    %263 = llvm.extractvalue %145[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %264 = llvm.extractvalue %145[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %265 = llvm.getelementptr %263[%264] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %266 = llvm.extractvalue %145[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %267 = llvm.mul %254, %266 overflow<nsw, nuw> : i64
    %268 = llvm.extractvalue %145[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %269 = llvm.mul %250, %268 overflow<nsw, nuw> : i64
    %270 = llvm.add %267, %269 overflow<nsw, nuw> : i64
    %271 = llvm.getelementptr inbounds|nuw %265[%270] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %272 = llvm.load %271 : !llvm.ptr -> f32
    %273 = llvm.extractvalue %121[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %274 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %275 = llvm.getelementptr inbounds|nuw %273[%274] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %276 = llvm.load %275 : !llvm.ptr -> f32
    %277 = llvm.fmul %262, %272 : f32
    %278 = llvm.fadd %276, %277 : f32
    %279 = llvm.extractvalue %121[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %280 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %281 = llvm.getelementptr inbounds|nuw %279[%280] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %278, %281 : f32, !llvm.ptr
    %282 = llvm.add %255, %3 : i64
    %283 = builtin.unrealized_conversion_cast %282 : i64 to index
    cf.br ^bb11(%283 : index)
  ^bb13:  // pred: ^bb11
    %284 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %285 = llvm.extractvalue %143[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %286 = llvm.getelementptr %284[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %287 = llvm.extractvalue %143[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %288 = llvm.mul %250, %287 overflow<nsw, nuw> : i64
    %289 = llvm.getelementptr inbounds|nuw %286[%288] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %290 = llvm.load %289 : !llvm.ptr -> f32
    %291 = llvm.extractvalue %137[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %292 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %293 = llvm.getelementptr inbounds|nuw %291[%292] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %290, %293 : f32, !llvm.ptr
    %294 = llvm.extractvalue %121[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %295 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %296 = llvm.getelementptr inbounds|nuw %294[%295] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %297 = llvm.load %296 : !llvm.ptr -> f32
    %298 = llvm.extractvalue %137[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %300 = llvm.getelementptr inbounds|nuw %298[%299] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %301 = llvm.load %300 : !llvm.ptr -> f32
    %302 = llvm.fadd %297, %301 : f32
    %303 = llvm.intr.maximum(%302, %7) : (f32, f32) -> f32
    %304 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %305 = llvm.mlir.constant(128 : index) : i64
    %306 = llvm.mul %5, %305 overflow<nsw, nuw> : i64
    %307 = llvm.add %306, %250 overflow<nsw, nuw> : i64
    %308 = llvm.getelementptr inbounds|nuw %304[%307] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %303, %308 : f32, !llvm.ptr
    %309 = llvm.add %251, %3 : i64
    %310 = builtin.unrealized_conversion_cast %309 : i64 to index
    cf.br ^bb9(%310 : index)
  ^bb14:  // pred: ^bb9
    cf.br ^bb15(%6 : index)
  ^bb15(%311: index):  // 2 preds: ^bb14, ^bb19
    %312 = builtin.unrealized_conversion_cast %311 : index to i64
    %313 = builtin.unrealized_conversion_cast %311 : index to i64
    %314 = llvm.icmp "slt" %313, %0 : i64
    cf.cond_br %314, ^bb16, ^bb20
  ^bb16:  // pred: ^bb15
    cf.br ^bb17(%6 : index)
  ^bb17(%315: index):  // 2 preds: ^bb16, ^bb18
    %316 = builtin.unrealized_conversion_cast %315 : index to i64
    %317 = builtin.unrealized_conversion_cast %315 : index to i64
    %318 = llvm.icmp "slt" %317, %1 : i64
    cf.cond_br %318, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %319 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %320 = llvm.mlir.constant(128 : index) : i64
    %321 = llvm.mul %5, %320 overflow<nsw, nuw> : i64
    %322 = llvm.add %321, %316 overflow<nsw, nuw> : i64
    %323 = llvm.getelementptr inbounds|nuw %319[%322] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %324 = llvm.load %323 : !llvm.ptr -> f32
    %325 = llvm.extractvalue %141[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %326 = llvm.extractvalue %141[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %327 = llvm.getelementptr %325[%326] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %328 = llvm.extractvalue %141[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %329 = llvm.mul %316, %328 overflow<nsw, nuw> : i64
    %330 = llvm.extractvalue %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %331 = llvm.mul %312, %330 overflow<nsw, nuw> : i64
    %332 = llvm.add %329, %331 overflow<nsw, nuw> : i64
    %333 = llvm.getelementptr inbounds|nuw %327[%332] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %334 = llvm.load %333 : !llvm.ptr -> f32
    %335 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %336 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %337 = llvm.getelementptr inbounds|nuw %335[%336] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %338 = llvm.load %337 : !llvm.ptr -> f32
    %339 = llvm.fmul %324, %334 : f32
    %340 = llvm.fadd %338, %339 : f32
    %341 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %342 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %343 = llvm.getelementptr inbounds|nuw %341[%342] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %340, %343 : f32, !llvm.ptr
    %344 = llvm.add %317, %3 : i64
    %345 = builtin.unrealized_conversion_cast %344 : i64 to index
    cf.br ^bb17(%345 : index)
  ^bb19:  // pred: ^bb17
    %346 = llvm.extractvalue %139[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %347 = llvm.extractvalue %139[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %348 = llvm.getelementptr %346[%347] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %349 = llvm.extractvalue %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %350 = llvm.mul %312, %349 overflow<nsw, nuw> : i64
    %351 = llvm.getelementptr inbounds|nuw %348[%350] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %352 = llvm.load %351 : !llvm.ptr -> f32
    %353 = llvm.extractvalue %56[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %354 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %355 = llvm.getelementptr inbounds|nuw %353[%354] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %352, %355 : f32, !llvm.ptr
    %356 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %357 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %358 = llvm.getelementptr inbounds|nuw %356[%357] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %359 = llvm.load %358 : !llvm.ptr -> f32
    %360 = llvm.extractvalue %56[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %361 = llvm.add %5, %5 overflow<nsw, nuw> : i64
    %362 = llvm.getelementptr inbounds|nuw %360[%361] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %363 = llvm.load %362 : !llvm.ptr -> f32
    %364 = llvm.fadd %359, %363 : f32
    %365 = llvm.extractvalue %177[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %366 = llvm.mlir.constant(10 : index) : i64
    %367 = llvm.mul %180, %366 overflow<nsw, nuw> : i64
    %368 = llvm.add %367, %312 overflow<nsw, nuw> : i64
    %369 = llvm.getelementptr inbounds|nuw %365[%368] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %364, %369 : f32, !llvm.ptr
    %370 = llvm.add %313, %3 : i64
    %371 = builtin.unrealized_conversion_cast %370 : i64 to index
    cf.br ^bb15(%371 : index)
  ^bb20:  // pred: ^bb15
    %372 = llvm.add %181, %3 : i64
    %373 = builtin.unrealized_conversion_cast %372 : i64 to index
    cf.br ^bb1(%373 : index)
  ^bb21:  // pred: ^bb1
    %374 = bufferization.to_tensor %178 : memref<32x10xf32> to tensor<32x10xf32>
    return %374 : tensor<32x10xf32>
  }
  func.func @irregular_linear_layer(%arg0: tensor<17x97xf32>, %arg1: tensor<97x53xf32>, %arg2: tensor<53xf32>) -> tensor<17x53xf32> {
    %0 = llvm.mlir.constant(97 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(901 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(53 : index) : i64
    %7 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<53xf32> to memref<53xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<53xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<97x53xf32> to memref<97x53xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<97x53xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<17x97xf32> to memref<17x97xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<17x97xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(17 : index) : i64
    %47 = llvm.mlir.constant(53 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(901 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<17x53xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.intr.maximum(%139, %7) : (f32, f32) -> f32
    %141 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(53 : index) : i64
    %143 = llvm.mul %85, %142 overflow<nsw, nuw> : i64
    %144 = llvm.add %143, %79 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %140, %145 : f32, !llvm.ptr
    %146 = llvm.add %74, %2 : i64
    %147 = builtin.unrealized_conversion_cast %146 : i64 to index
    cf.br ^bb1(%147 : index)
  ^bb6:  // pred: ^bb1
    %148 = bufferization.to_tensor %72 : memref<17x53xf32> to tensor<17x53xf32>
    return %148 : tensor<17x53xf32>
  }
  func.func @linear_with_sigmoid(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%8] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.call @malloc(%13) : (i64) -> !llvm.ptr
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %9, %20[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %10, %22[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.zero : !llvm.ptr
    %28 = llvm.getelementptr %27[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %30, %31[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %30, %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(0 : index) : i64
    %35 = llvm.insertvalue %34, %33[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %25, %36[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %26, %38[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %41 = builtin.unrealized_conversion_cast %40 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %42 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.mlir.constant(32 : index) : i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(2048 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.mlir.constant(64 : index) : i64
    %54 = llvm.add %52, %53 : i64
    %55 = llvm.call @malloc(%54) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(1 : index) : i64
    %58 = llvm.sub %53, %57 : i64
    %59 = llvm.add %56, %58 : i64
    %60 = llvm.urem %59, %53 : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %64 = llvm.insertvalue %55, %63[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %62, %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(0 : index) : i64
    %67 = llvm.insertvalue %66, %65[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %47, %68[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = builtin.unrealized_conversion_cast %71 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%73: index):  // 2 preds: ^bb0, ^bb5
    %74 = builtin.unrealized_conversion_cast %73 : index to i64
    %75 = llvm.icmp "slt" %74, %3 : i64
    cf.cond_br %75, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %76 = llvm.srem %74, %6 : i64
    %77 = llvm.icmp "slt" %76, %4 : i64
    %78 = llvm.add %76, %6 : i64
    %79 = llvm.select %77, %78, %76 : i1, i64
    %80 = llvm.icmp "slt" %74, %4 : i64
    %81 = llvm.sub %1, %74 : i64
    %82 = llvm.select %80, %81, %74 : i1, i64
    %83 = llvm.sdiv %82, %6 : i64
    %84 = llvm.sub %1, %83 : i64
    %85 = llvm.select %80, %84, %83 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%86: index):  // 2 preds: ^bb2, ^bb4
    %87 = builtin.unrealized_conversion_cast %86 : index to i64
    %88 = builtin.unrealized_conversion_cast %86 : index to i64
    %89 = llvm.icmp "slt" %88, %0 : i64
    cf.cond_br %89, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %90 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.mul %85, %93 overflow<nsw, nuw> : i64
    %95 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.add %94, %96 overflow<nsw, nuw> : i64
    %98 = llvm.getelementptr inbounds|nuw %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %102 = llvm.getelementptr %100[%101] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.mul %87, %103 overflow<nsw, nuw> : i64
    %105 = llvm.extractvalue %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %79, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %104, %106 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %102[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.load %108 : !llvm.ptr -> f32
    %110 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %112 = llvm.getelementptr inbounds|nuw %110[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fmul %99, %109 : f32
    %115 = llvm.fadd %113, %114 : f32
    %116 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %118 = llvm.getelementptr inbounds|nuw %116[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %115, %118 : f32, !llvm.ptr
    %119 = llvm.add %88, %2 : i64
    %120 = builtin.unrealized_conversion_cast %119 : i64 to index
    cf.br ^bb3(%120 : index)
  ^bb5:  // pred: ^bb3
    %121 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.extractvalue %41[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %123 = llvm.getelementptr %121[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %124 = llvm.extractvalue %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.mul %79, %124 overflow<nsw, nuw> : i64
    %126 = llvm.getelementptr inbounds|nuw %123[%125] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %130 = llvm.getelementptr inbounds|nuw %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %127, %130 : f32, !llvm.ptr
    %131 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %133 = llvm.getelementptr inbounds|nuw %131[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %134 = llvm.load %133 : !llvm.ptr -> f32
    %135 = llvm.extractvalue %39[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %137 = llvm.getelementptr inbounds|nuw %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.load %137 : !llvm.ptr -> f32
    %139 = llvm.fadd %134, %138 : f32
    %140 = llvm.fneg %139 : f32
    %141 = math.exp %140 : f32
    %142 = llvm.fadd %141, %7 : f32
    %143 = llvm.fdiv %7, %142 : f32
    %144 = llvm.extractvalue %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.mlir.constant(64 : index) : i64
    %146 = llvm.mul %85, %145 overflow<nsw, nuw> : i64
    %147 = llvm.add %146, %79 overflow<nsw, nuw> : i64
    %148 = llvm.getelementptr inbounds|nuw %144[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %143, %148 : f32, !llvm.ptr
    %149 = llvm.add %74, %2 : i64
    %150 = builtin.unrealized_conversion_cast %149 : i64 to index
    cf.br ^bb1(%150 : index)
  ^bb6:  // pred: ^bb1
    %151 = bufferization.to_tensor %72 : memref<32x64xf32> to tensor<32x64xf32>
    return %151 : tensor<32x64xf32>
  }
  func.func @linear_with_gelu(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(1.41421354 : f32) : f32
    %8 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %9 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %10 = llvm.mlir.constant(1 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.zero : !llvm.ptr
    %14 = llvm.getelementptr %13[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    %17 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.insertvalue %20, %19[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %10, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %11, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %11, %23[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %12, %24[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.mlir.zero : !llvm.ptr
    %30 = llvm.getelementptr %29[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %33 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %32, %33[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.insertvalue %36, %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %26, %37[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %27, %38[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %27, %39[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %28, %40[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %43 = builtin.unrealized_conversion_cast %42 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %44 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %45 = builtin.unrealized_conversion_cast %44 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %47 = builtin.unrealized_conversion_cast %46 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.mlir.constant(32 : index) : i64
    %49 = llvm.mlir.constant(64 : index) : i64
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.mlir.constant(2048 : index) : i64
    %52 = llvm.mlir.zero : !llvm.ptr
    %53 = llvm.getelementptr %52[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.mlir.constant(64 : index) : i64
    %56 = llvm.add %54, %55 : i64
    %57 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.sub %55, %59 : i64
    %61 = llvm.add %58, %60 : i64
    %62 = llvm.urem %61, %55 : i64
    %63 = llvm.sub %61, %62 : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr
    %65 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.insertvalue %57, %65[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %48, %69[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %49, %70[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %49, %71[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %50, %72[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = builtin.unrealized_conversion_cast %73 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%75: index):  // 2 preds: ^bb0, ^bb5
    %76 = builtin.unrealized_conversion_cast %75 : index to i64
    %77 = llvm.icmp "slt" %76, %3 : i64
    cf.cond_br %77, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %78 = llvm.srem %76, %6 : i64
    %79 = llvm.icmp "slt" %78, %4 : i64
    %80 = llvm.add %78, %6 : i64
    %81 = llvm.select %79, %80, %78 : i1, i64
    %82 = llvm.icmp "slt" %76, %4 : i64
    %83 = llvm.sub %1, %76 : i64
    %84 = llvm.select %82, %83, %76 : i1, i64
    %85 = llvm.sdiv %84, %6 : i64
    %86 = llvm.sub %1, %85 : i64
    %87 = llvm.select %82, %86, %85 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%88: index):  // 2 preds: ^bb2, ^bb4
    %89 = builtin.unrealized_conversion_cast %88 : index to i64
    %90 = builtin.unrealized_conversion_cast %88 : index to i64
    %91 = llvm.icmp "slt" %90, %0 : i64
    cf.cond_br %91, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %92 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.extractvalue %47[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.getelementptr %92[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.extractvalue %47[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mul %87, %95 overflow<nsw, nuw> : i64
    %97 = llvm.extractvalue %47[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.mul %89, %97 overflow<nsw, nuw> : i64
    %99 = llvm.add %96, %98 overflow<nsw, nuw> : i64
    %100 = llvm.getelementptr inbounds|nuw %94[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %101 = llvm.load %100 : !llvm.ptr -> f32
    %102 = llvm.extractvalue %45[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.extractvalue %45[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %104 = llvm.getelementptr %102[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %105 = llvm.extractvalue %45[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mul %89, %105 overflow<nsw, nuw> : i64
    %107 = llvm.extractvalue %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mul %81, %107 overflow<nsw, nuw> : i64
    %109 = llvm.add %106, %108 overflow<nsw, nuw> : i64
    %110 = llvm.getelementptr inbounds|nuw %104[%109] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %111 = llvm.load %110 : !llvm.ptr -> f32
    %112 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %114 = llvm.getelementptr inbounds|nuw %112[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %115 = llvm.load %114 : !llvm.ptr -> f32
    %116 = llvm.fmul %101, %111 : f32
    %117 = llvm.fadd %115, %116 : f32
    %118 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %120 = llvm.getelementptr inbounds|nuw %118[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %117, %120 : f32, !llvm.ptr
    %121 = llvm.add %90, %2 : i64
    %122 = builtin.unrealized_conversion_cast %121 : i64 to index
    cf.br ^bb3(%122 : index)
  ^bb5:  // pred: ^bb3
    %123 = llvm.extractvalue %43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.extractvalue %43[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %125 = llvm.getelementptr %123[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.extractvalue %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %127 = llvm.mul %81, %126 overflow<nsw, nuw> : i64
    %128 = llvm.getelementptr inbounds|nuw %125[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
    %130 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %132 = llvm.getelementptr inbounds|nuw %130[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %129, %132 : f32, !llvm.ptr
    %133 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %135 = llvm.getelementptr inbounds|nuw %133[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %136 = llvm.load %135 : !llvm.ptr -> f32
    %137 = llvm.extractvalue %41[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %139 = llvm.getelementptr inbounds|nuw %137[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %140 = llvm.load %139 : !llvm.ptr -> f32
    %141 = llvm.fadd %136, %140 : f32
    %142 = llvm.fdiv %141, %7 : f32
    %143 = math.erf %142 : f32
    %144 = llvm.fadd %143, %8 : f32
    %145 = llvm.fmul %144, %9 : f32
    %146 = llvm.fmul %141, %145 : f32
    %147 = llvm.extractvalue %73[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.mlir.constant(64 : index) : i64
    %149 = llvm.mul %87, %148 overflow<nsw, nuw> : i64
    %150 = llvm.add %149, %81 overflow<nsw, nuw> : i64
    %151 = llvm.getelementptr inbounds|nuw %147[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %146, %151 : f32, !llvm.ptr
    %152 = llvm.add %76, %2 : i64
    %153 = builtin.unrealized_conversion_cast %152 : i64 to index
    cf.br ^bb1(%153 : index)
  ^bb6:  // pred: ^bb1
    %154 = bufferization.to_tensor %74 : memref<32x64xf32> to tensor<32x64xf32>
    return %154 : tensor<32x64xf32>
  }
  func.func @linear_with_tanh(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<64xf32>) -> tensor<32x64xf32> {
    %0 = llvm.mlir.constant(128 : index) : i64
    %1 = llvm.mlir.constant(-1 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = builtin.unrealized_conversion_cast %4 : i64 to index
    %6 = llvm.mlir.constant(64 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr
    %14 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %13, %14[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.insertvalue %17, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %7, %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %8, %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %8, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %9, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.zero : !llvm.ptr
    %27 = llvm.getelementptr %26[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %29, %30[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %29, %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.insertvalue %33, %32[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.insertvalue %23, %34[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %36 = llvm.insertvalue %24, %35[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.insertvalue %24, %36[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %25, %37[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = bufferization.to_buffer %arg2 : tensor<64xf32> to memref<64xf32, strided<[?], offset: ?>>
    %40 = builtin.unrealized_conversion_cast %39 : memref<64xf32, strided<[?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %41 = bufferization.to_buffer %arg1 : tensor<128x64xf32> to memref<128x64xf32, strided<[?, ?], offset: ?>>
    %42 = builtin.unrealized_conversion_cast %41 : memref<128x64xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = bufferization.to_buffer %arg0 : tensor<32x128xf32> to memref<32x128xf32, strided<[?, ?], offset: ?>>
    %44 = builtin.unrealized_conversion_cast %43 : memref<32x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.mlir.constant(32 : index) : i64
    %46 = llvm.mlir.constant(64 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.mlir.constant(2048 : index) : i64
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.mlir.constant(64 : index) : i64
    %53 = llvm.add %51, %52 : i64
    %54 = llvm.call @malloc(%53) : (i64) -> !llvm.ptr
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.mlir.constant(1 : index) : i64
    %57 = llvm.sub %52, %56 : i64
    %58 = llvm.add %55, %57 : i64
    %59 = llvm.urem %58, %52 : i64
    %60 = llvm.sub %58, %59 : i64
    %61 = llvm.inttoptr %60 : i64 to !llvm.ptr
    %62 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %63 = llvm.insertvalue %54, %62[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %61, %63[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mlir.constant(0 : index) : i64
    %66 = llvm.insertvalue %65, %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %45, %66[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %46, %67[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %46, %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = builtin.unrealized_conversion_cast %70 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<32x64xf32>
    cf.br ^bb1(%5 : index)
  ^bb1(%72: index):  // 2 preds: ^bb0, ^bb5
    %73 = builtin.unrealized_conversion_cast %72 : index to i64
    %74 = llvm.icmp "slt" %73, %3 : i64
    cf.cond_br %74, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %75 = llvm.srem %73, %6 : i64
    %76 = llvm.icmp "slt" %75, %4 : i64
    %77 = llvm.add %75, %6 : i64
    %78 = llvm.select %76, %77, %75 : i1, i64
    %79 = llvm.icmp "slt" %73, %4 : i64
    %80 = llvm.sub %1, %73 : i64
    %81 = llvm.select %79, %80, %73 : i1, i64
    %82 = llvm.sdiv %81, %6 : i64
    %83 = llvm.sub %1, %82 : i64
    %84 = llvm.select %79, %83, %82 : i1, i64
    cf.br ^bb3(%5 : index)
  ^bb3(%85: index):  // 2 preds: ^bb2, ^bb4
    %86 = builtin.unrealized_conversion_cast %85 : index to i64
    %87 = builtin.unrealized_conversion_cast %85 : index to i64
    %88 = llvm.icmp "slt" %87, %0 : i64
    cf.cond_br %88, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %89 = llvm.extractvalue %44[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.extractvalue %44[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.extractvalue %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mul %84, %92 overflow<nsw, nuw> : i64
    %94 = llvm.extractvalue %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mul %86, %94 overflow<nsw, nuw> : i64
    %96 = llvm.add %93, %95 overflow<nsw, nuw> : i64
    %97 = llvm.getelementptr inbounds|nuw %91[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 : !llvm.ptr -> f32
    %99 = llvm.extractvalue %42[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.extractvalue %42[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.getelementptr %99[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.extractvalue %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %103 = llvm.mul %86, %102 overflow<nsw, nuw> : i64
    %104 = llvm.extractvalue %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %105 = llvm.mul %78, %104 overflow<nsw, nuw> : i64
    %106 = llvm.add %103, %105 overflow<nsw, nuw> : i64
    %107 = llvm.getelementptr inbounds|nuw %101[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.load %107 : !llvm.ptr -> f32
    %109 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %109[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.fmul %98, %108 : f32
    %114 = llvm.fadd %112, %113 : f32
    %115 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %117 = llvm.getelementptr inbounds|nuw %115[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %114, %117 : f32, !llvm.ptr
    %118 = llvm.add %87, %2 : i64
    %119 = builtin.unrealized_conversion_cast %118 : i64 to index
    cf.br ^bb3(%119 : index)
  ^bb5:  // pred: ^bb3
    %120 = llvm.extractvalue %40[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %121 = llvm.extractvalue %40[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %122 = llvm.getelementptr %120[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %123 = llvm.extractvalue %40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %124 = llvm.mul %78, %123 overflow<nsw, nuw> : i64
    %125 = llvm.getelementptr inbounds|nuw %122[%124] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %126 = llvm.load %125 : !llvm.ptr -> f32
    %127 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %129 = llvm.getelementptr inbounds|nuw %127[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %126, %129 : f32, !llvm.ptr
    %130 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %132 = llvm.getelementptr inbounds|nuw %130[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %133 = llvm.load %132 : !llvm.ptr -> f32
    %134 = llvm.extractvalue %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %135 = llvm.add %4, %4 overflow<nsw, nuw> : i64
    %136 = llvm.getelementptr inbounds|nuw %134[%135] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %137 = llvm.load %136 : !llvm.ptr -> f32
    %138 = llvm.fadd %133, %137 : f32
    %139 = math.tanh %138 : f32
    %140 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(64 : index) : i64
    %142 = llvm.mul %84, %141 overflow<nsw, nuw> : i64
    %143 = llvm.add %142, %78 overflow<nsw, nuw> : i64
    %144 = llvm.getelementptr inbounds|nuw %140[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %139, %144 : f32, !llvm.ptr
    %145 = llvm.add %73, %2 : i64
    %146 = builtin.unrealized_conversion_cast %145 : i64 to index
    cf.br ^bb1(%146 : index)
  ^bb6:  // pred: ^bb1
    %147 = bufferization.to_tensor %71 : memref<32x64xf32> to tensor<32x64xf32>
    return %147 : tensor<32x64xf32>
  }
}

