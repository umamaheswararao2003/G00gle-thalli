//===- GoogleOps.td - Google dialect ops -------------------*- tablegen -*-===//
//
// Google Dialect Operations
//
//===----------------------------------------------------------------------===//

#ifndef GOOGLE_OPS
#define GOOGLE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

//===----------------------------------------------------------------------===//
// Google Dialect Definition
//===----------------------------------------------------------------------===//

def Google_Dialect : Dialect {
  let name = "google";
  let summary = "Google dialect for custom operations";
  let description = [{
    This dialect contains custom operations for Google-specific IR.
  }];
  let cppNamespace = "::mlir::google";
}

//===----------------------------------------------------------------------===//
// Reduction Kind Enum
//===----------------------------------------------------------------------===//

def Google_ReductionKind : I32EnumAttr<"ReductionKind",
    "Reduction operation kind", [
  I32EnumAttrCase<"MAX", 0, "max">,
  I32EnumAttrCase<"MIN", 1, "min">,
  I32EnumAttrCase<"SUM", 2, "sum">,
  I32EnumAttrCase<"PRODUCT", 3, "product">,
  //I32EnumAttrCase<"AND", 4, "and">,
  //I32EnumAttrCase<"OR", 5, "or">,
  //I32EnumAttrCase<"XOR", 6, "xor">,
  I32EnumAttrCase<"MEAN", 7, "mean">,
  I32EnumAttrCase<"ARGMAX", 8, "argmax">,
  I32EnumAttrCase<"ARGMIN", 9, "argmin">
]> {
  let cppNamespace = "::mlir::google";
}

//===----------------------------------------------------------------------===//
// Base Google Operation Definition
//===----------------------------------------------------------------------===//

class Google_Op<string mnemonic, list<Trait> traits = []> :
    Op<Google_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base Unary Operation
//===----------------------------------------------------------------------===//

class Google_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Google_Op<mnemonic, traits # [Pure]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

//===----------------------------------------------------------------------===//
// Base Binary Operation
//===----------------------------------------------------------------------===//

class Google_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Google_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Google Operations
//===----------------------------------------------------------------------===//

def Google_ConstantOp : Google_Op<"constant", [Pure]> {
  let summary = "Constant tensor operation";
  let description = [{
    Creates a constant tensor with a specified value.
    
    Example:
    ```mlir
    %0 = google.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : tensor<4xf32>
    ```
  }];
  
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    attr-dict `:` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$resultType)>
  ];

  let hasVerifier = 1;

}

def Google_AddOp : Google_BinaryOp<"add", [Commutative]> {
  let summary = "Element-wise addition operation";
  let description = [{
    Performs element-wise addition of two tensors.
    
    Example:
    ```mlir
    %result = google.add %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_MaxOp : Google_BinaryOp<"max", [Commutative]> {
  let summary = "Element-wise maximum operation";
  let description = [{
    Performs element-wise maximum of two tensors.
    
    Example:
    ```mlir
    %result = google.max %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_MinOp : Google_BinaryOp<"min", [Commutative]> {
  let summary = "Element-wise minimum operation";
  let description = [{
    Performs element-wise minimum of two tensors.
    
    Example:
    ```mlir
    %result = google.min %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_SubOp : Google_BinaryOp<"sub"> {
  let summary = "Element-wise subtraction operation";
  let description = [{
    Performs element-wise subtraction of two tensors.
    
    Example:
    ```mlir
    %result = google.sub %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_MulOp : Google_BinaryOp<"mul", [Commutative]> {
  let summary = "Element-wise multiplication operation";
  let description = [{
    Performs element-wise multiplication of two tensors.
    
    Example:
    ```mlir
    %result = google.mul %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_DivOp : Google_BinaryOp<"div"> {
  let summary = "Element-wise division operation";
  let description = [{
    Performs element-wise division of two tensors.
    
    Example:
    ```mlir
    %result = google.div %a, %b : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_PowOp : Google_BinaryOp<"pow"> {
  let summary = "Element-wise power operation";
  let description = [{
    Performs element-wise power operation (base^exponent).
    
    Example:
    ```mlir
    %result = google.pow %base, %exp : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_ReduceOp : Google_Op<"reduce", [Pure]> {
  let summary = "Generic reduction operation";
  let description = [{
    Performs reduction operation across specified dimensions of a tensor.
    The reduction kind is specified by the 'kind' attribute.
    
    Examples:
    ```mlir
    // Reduce all dimensions with max
    %result = google.reduce<max> %input : tensor<4x8xf32> -> tensor<f32>
    
    // Reduce along specific axes with sum
    %result = google.reduce<sum> %input axes = [0] : tensor<4x8xf32> -> tensor<8xf32>
    
    // Keep reduced dimensions with min
    %result = google.reduce<min> %input axes = [1] keepdims = true : 
              tensor<4x8xf32> -> tensor<4x1xf32>
    
    // Product reduction
    %result = google.reduce<product> %input : tensor<4x8xf32> -> tensor<f32>
    ```
  }];
  
  let arguments = (ins 
    Google_ReductionKind:$kind,
    AnyTensor:$input,
    OptionalAttr<I64ArrayAttr>:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    `<` $kind `>` $input (`axes` `=` $axes^)? (`keepdims` `=` $keepdims^)?
    attr-dict `:` type($input) `->` type($output)
  }];
  
  let builders = [
    OpBuilder<(ins "ReductionKind":$kind, "Value":$input, 
                   "ArrayRef<int64_t>":$axes, "bool":$keepdims, "Type":$resultType)>
  ];
  
  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
}

def Google_NegOp : Google_UnaryOp<"neg"> {
  let summary = "Element-wise negation operation";
  let description = [{
    Performs element-wise negation of a tensor.
    
    Example:
    ```mlir
    %result = google.neg %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_AbsOp : Google_UnaryOp<"abs"> {
  let summary = "Element-wise absolute value operation";
  let description = [{
    Performs element-wise absolute value of a tensor.
    
    Example:
    ```mlir
    %result = google.abs %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_SqrtOp : Google_UnaryOp<"sqrt"> {
  let summary = "Element-wise square root operation";
  let description = [{
    Performs element-wise square root of a tensor.
    
    Example:
    ```mlir
    %result = google.sqrt %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_RsqrtOp : Google_UnaryOp<"rsqrt"> {
  let summary = "Element-wise reciprocal square root operation";
  let description = [{
    Performs element-wise reciprocal square root (1/sqrt(x)) of a tensor.
    
    Example:
    ```mlir
    %result = google.rsqrt %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_ExpOp : Google_UnaryOp<"exp"> {
  let summary = "Element-wise exponential operation";
  let description = [{
    Performs element-wise exponential (e^x) of a tensor.
    
    Example:
    ```mlir
    %result = google.exp %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_LogOp : Google_UnaryOp<"log"> {
  let summary = "Element-wise natural logarithm operation";
  let description = [{
    Performs element-wise natural logarithm of a tensor.
    
    Example:
    ```mlir
    %result = google.log %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_CeilOp : Google_UnaryOp<"ceil"> {
  let summary = "Element-wise ceiling operation";
  let description = [{
    Performs element-wise ceiling (round up) of a tensor.
    
    Example:
    ```mlir
    %result = google.ceil %input : tensor<4xf32> -> tensor<4xf32>
    ```
  }];
}

def Google_ReluOp : Google_UnaryOp<"relu"> {
  let summary = "Rectified Linear Unit activation";
  let description = [{
    Performs element-wise ReLU activation: max(0, x).
    
    Example:
    ```mlir
    %result = google.relu %input : tensor<128x256xf32> -> tensor<128x256xf32>
    ```
  }];
}

def Google_GeluOp : Google_UnaryOp<"gelu"> {
  let summary = "Gaussian Error Linear Unit activation";
  let description = [{
    Performs element-wise GELU activation.
    Commonly used in transformer models (BERT, GPT).
    
    Example:
    ```mlir
    %result = google.gelu %input : tensor<128x768xf32> -> tensor<128x768xf32>
    ```
  }];
}

def Google_SigmoidOp : Google_UnaryOp<"sigmoid"> {
  let summary = "Sigmoid activation";
  let description = [{
    Performs element-wise sigmoid activation: 1 / (1 + exp(-x)).
    
    Example:
    ```mlir
    %result = google.sigmoid %input : tensor<32x10xf32> -> tensor<32x10xf32>
    ```
  }];
}

def Google_TanhOp : Google_UnaryOp<"tanh"> {
  let summary = "Hyperbolic tangent activation";
  let description = [{
    Performs element-wise tanh activation.
    
    Example:
    ```mlir
    %result = google.tanh %input : tensor<64x128xf32> -> tensor<64x128xf32>
    ```
  }];
}

def Google_SoftmaxOp : Google_Op<"softmax", [Pure]> {
  let summary = "Softmax activation";
  let description = [{
    Performs softmax activation along a specified axis.
    
    Example:
    ```mlir
    %result = google.softmax %input {axis = 1 : i64} : tensor<32x1000xf32> -> tensor<32x1000xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_ReshapeOp : Google_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes a tensor to a new shape while preserving total elements.
    
    Example:
    ```mlir
    %result = google.reshape %input {shape = [32, 768]} : tensor<32x12x64xf32> -> tensor<32x768xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$shape
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_TransposeOp : Google_Op<"transpose", [Pure]> {
  let summary = "Transpose operation";
  let description = [{
    Transposes a tensor by permuting its dimensions.
    
    Example:
    ```mlir
    %result = google.transpose %input {perm = [0, 2, 1]} : tensor<32x128x768xf32> -> tensor<32x768x128xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$perm
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_ConcatOp : Google_Op<"concat", [Pure]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates multiple tensors along a specified axis.
    
    Example:
    ```mlir
    %result = google.concat %a, %b, %c {axis = 1 : i64} 
      : tensor<32x64xf32>, tensor<32x128xf32>, tensor<32x256xf32> -> tensor<32x448xf32>
    ```
  }];
  
  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$axis
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_SliceOp : Google_Op<"slice", [Pure]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from a tensor.
    
    Example:
    ```mlir
    %result = google.slice %input {start = [0, 10], end = [32, 50]} 
      : tensor<32x128xf32> -> tensor<32x40xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$start,
    I64ArrayAttr:$end
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_BroadcastOp : Google_Op<"broadcast", [Pure]> {
  let summary = "Broadcast operation";
  let description = [{
    Broadcasts a tensor to a new shape following NumPy broadcasting rules.
    
    Example:
    ```mlir
    %result = google.broadcast %input {shape = [32, 128, 768]} 
      : tensor<768xf32> -> tensor<32x128x768xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$shape
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_SelectOp : Google_Op<"select", [Pure]> {
  let summary = "Conditional select operation";
  let description = [{
    Selects elements from true_val or false_val based on condition.
    
    Example:
    ```mlir
    %result = google.select %cond, %true_val, %false_val 
      : tensor<32xi1>, tensor<32xf32>, tensor<32xf32> -> tensor<32xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$condition,
    AnyTensor:$true_val,
    AnyTensor:$false_val
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $condition `,` $true_val `,` $false_val attr-dict 
    `:` type($condition) `,` type($true_val) `,` type($false_val) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_ClampOp : Google_Op<"clamp", [Pure]> {
  let summary = "Clamp operation";
  let description = [{
    Clamps values in a tensor to a specified range [min, max].
    
    Example:
    ```mlir
    %result = google.clamp %input, %min, %max 
      : tensor<32xf32>, tensor<f32>, tensor<f32> -> tensor<32xf32>
    ```
  }];
  
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$min,
    AnyTensor:$max
  );
  
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = [{
    $input `,` $min `,` $max attr-dict 
    `:` type($input) `,` type($min) `,` type($max) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

def Google_DequantOp : Google_Op<"dequant",
  [Pure,
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation",
     "generateResultTileValue",
     "getIterationDomainTileFromResultTile"]>
  ]> {
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale
  );
  let results = (outs
    AnyRankedTensor:$output
  );
  let assemblyFormat = [{
    $input `,` $scale attr-dict `:` type($input) `,` type($scale) `,` type($output)
  }];
}

def Google_MatMulOp : Google_Op<"matmul",
  [Pure,
   DeclareOpInterfaceMethods<TilingInterface,
    ["getIterationDomain",
     "getLoopIteratorTypes",
     "getResultTilePosition",
     "getTiledImplementation",
     "generateResultTileValue",
     "getIterationDomainTileFromResultTile"]>
  ]> {
  let summary = "Matrix multiplication operation";
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );
  
  let results = (outs
    AnyRankedTensor:$output
  );
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
  
  let hasVerifier = 1;
}

#endif // GOOGLE_OPS