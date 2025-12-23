module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  func.func @matmul_no_transform(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = bufferization.to_buffer %arg1 : tensor<128x128xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
    %1 = builtin.unrealized_conversion_cast %0 : memref<128x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = bufferization.to_buffer %arg0 : tensor<128x128xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
    %3 = builtin.unrealized_conversion_cast %2 : memref<128x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = bufferization.to_buffer %arg2 : tensor<128x128xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
    %5 = builtin.unrealized_conversion_cast %4 : memref<128x128xf32, strided<[?, ?], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.mlir.constant(128 : index) : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(16384 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.add %12, %13 : i64
    %15 = llvm.call @malloc(%14) : (i64) -> !llvm.ptr
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mlir.constant(1 : index) : i64
    %18 = llvm.sub %13, %17 : i64
    %19 = llvm.add %16, %18 : i64
    %20 = llvm.urem %19, %13 : i64
    %21 = llvm.sub %19, %20 : i64
    %22 = llvm.inttoptr %21 : i64 to !llvm.ptr
    %23 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %24 = llvm.insertvalue %15, %23[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.constant(0 : index) : i64
    %27 = llvm.insertvalue %26, %25[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %6, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %7, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %7, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %8, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = builtin.unrealized_conversion_cast %31 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<128x128xf32>
    %33 = llvm.intr.stacksave : !llvm.ptr
    %34 = llvm.mlir.constant(2 : i64) : i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.alloca %35 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %5, %36 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %37 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %38 = llvm.insertvalue %34, %37[0] : !llvm.struct<(i64, ptr)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(i64, ptr)> 
    %40 = llvm.mlir.constant(2 : i64) : i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.alloca %41 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %31, %42 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %43 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %44 = llvm.insertvalue %40, %43[0] : !llvm.struct<(i64, ptr)> 
    %45 = llvm.insertvalue %42, %44[1] : !llvm.struct<(i64, ptr)> 
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.alloca %46 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %39, %47 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %48 = llvm.alloca %46 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %45, %48 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    llvm.call @memrefCopy(%51, %47, %48) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %33 : !llvm.ptr
    %52 = llvm.mlir.constant(128 : index) : i64
    %53 = llvm.mlir.constant(128 : index) : i64
    %54 = llvm.mlir.constant(16384 : index) : i64
    %55 = llvm.mlir.constant(128 : index) : i64
    %56 = llvm.mul %54, %55 overflow<nsw> : i64
    %57 = llvm.mlir.constant(0 : index) : i64
    %58 = builtin.unrealized_conversion_cast %57 : i64 to index
    %59 = llvm.mlir.constant(2097152 : index) : i64
    %60 = builtin.unrealized_conversion_cast %59 : i64 to index
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = builtin.unrealized_conversion_cast %61 : i64 to index
    scf.for %arg3 = %58 to %60 step %62 {
      %64 = builtin.unrealized_conversion_cast %arg3 : index to i64
      %65 = llvm.srem %64, %55 : i64
      %66 = llvm.mlir.constant(0 : index) : i64
      %67 = llvm.icmp "slt" %65, %66 : i64
      %68 = llvm.add %65, %55 : i64
      %69 = llvm.select %67, %68, %65 : i1, i64
      %70 = llvm.mlir.constant(0 : index) : i64
      %71 = llvm.mlir.constant(-1 : index) : i64
      %72 = llvm.icmp "slt" %64, %70 : i64
      %73 = llvm.sub %71, %64 : i64
      %74 = llvm.select %72, %73, %64 : i1, i64
      %75 = llvm.sdiv %74, %55 : i64
      %76 = llvm.sub %71, %75 : i64
      %77 = llvm.select %72, %76, %75 : i1, i64
      %78 = llvm.srem %77, %53 : i64
      %79 = llvm.mlir.constant(0 : index) : i64
      %80 = llvm.icmp "slt" %78, %79 : i64
      %81 = llvm.add %78, %53 : i64
      %82 = llvm.select %80, %81, %78 : i1, i64
      %83 = llvm.mlir.constant(0 : index) : i64
      %84 = llvm.mlir.constant(-1 : index) : i64
      %85 = llvm.icmp "slt" %77, %83 : i64
      %86 = llvm.sub %84, %77 : i64
      %87 = llvm.select %85, %86, %77 : i1, i64
      %88 = llvm.sdiv %87, %53 : i64
      %89 = llvm.sub %84, %88 : i64
      %90 = llvm.select %85, %89, %88 : i1, i64
      %91 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %92 = llvm.extractvalue %3[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %93 = llvm.getelementptr %91[%92] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %94 = llvm.extractvalue %3[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %95 = llvm.mul %90, %94 overflow<nsw, nuw> : i64
      %96 = llvm.extractvalue %3[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %97 = llvm.mul %69, %96 overflow<nsw, nuw> : i64
      %98 = llvm.add %95, %97 overflow<nsw, nuw> : i64
      %99 = llvm.getelementptr inbounds|nuw %93[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %100 = llvm.load %99 : !llvm.ptr -> f32
      %101 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %102 = llvm.extractvalue %1[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %103 = llvm.getelementptr %101[%102] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %104 = llvm.extractvalue %1[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %105 = llvm.mul %69, %104 overflow<nsw, nuw> : i64
      %106 = llvm.extractvalue %1[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %107 = llvm.mul %82, %106 overflow<nsw, nuw> : i64
      %108 = llvm.add %105, %107 overflow<nsw, nuw> : i64
      %109 = llvm.getelementptr inbounds|nuw %103[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %110 = llvm.load %109 : !llvm.ptr -> f32
      %111 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %112 = llvm.mlir.constant(128 : index) : i64
      %113 = llvm.mul %90, %112 overflow<nsw, nuw> : i64
      %114 = llvm.add %113, %82 overflow<nsw, nuw> : i64
      %115 = llvm.getelementptr inbounds|nuw %111[%114] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %116 = llvm.load %115 : !llvm.ptr -> f32
      %117 = llvm.fmul %100, %110 : f32
      %118 = llvm.fadd %116, %117 : f32
      %119 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %120 = llvm.mlir.constant(128 : index) : i64
      %121 = llvm.mul %90, %120 overflow<nsw, nuw> : i64
      %122 = llvm.add %121, %82 overflow<nsw, nuw> : i64
      %123 = llvm.getelementptr inbounds|nuw %119[%122] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %118, %123 : f32, !llvm.ptr
    }
    %63 = bufferization.to_tensor %32 : memref<128x128xf32> to tensor<128x128xf32>
    return %63 : tensor<128x128xf32>
  }
}

