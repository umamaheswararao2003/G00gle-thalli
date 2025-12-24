//===- GoogleOps.cpp - Google dialect ops ----------------------*- C++ -*-===//
//
// Google Dialect Operations Implementation
//
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::google;

//===----------------------------------------------------------------------===//
// Google Dialect
//===----------------------------------------------------------------------===//

#include "Google/IR/GoogleOpsDialect.cpp.inc"

void GoogleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Google/IR/GoogleOps.cpp.inc"
      >();
}

#include "Google/IR/GoogleOpsAttributes.cpp.inc"
#include "Google/IR/GoogleOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Google Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Google/IR/GoogleOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp Builder and Verification
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &builder, OperationState &state, Attribute value, Type resultType) {
  state.addAttribute("value", value);
  state.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// ConstantOp Verification

LogicalResult ConstantOp::verify() {
  auto valueAttr = getValue();
  auto outputType = getOutput().getType();
  
  // Check if value attribute type matches output type
  if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    if (denseAttr.getType() != outputType) {
      return emitOpError("value attribute type ")
             << denseAttr.getType() << " does not match output type "
             << outputType;
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Binary Operations Verification
//===----------------------------------------------------------------------===//

// Generic verifier for all binary operations
template<typename OpTy>
static LogicalResult verifyBinaryOp(OpTy op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  
  if (!lhsType || !rhsType || !resultType) {
    return op.emitOpError("operands and result must be ranked tensors");
  }
  
  // Check shapes match
  if (lhsType.getShape() != rhsType.getShape()) {
    return op.emitOpError("operand shapes must match, got ")
           << lhsType << " and " << rhsType;
  }
  
  if (lhsType.getShape() != resultType.getShape()) {
    return op.emitOpError("result shape must match operand shapes, got ")
           << resultType << " for operands " << lhsType;
  }
  
  // Check element types match
  if (lhsType.getElementType() != rhsType.getElementType()) {
    return op.emitOpError("operand element types must match, got ")
           << lhsType.getElementType() << " and " << rhsType.getElementType();
  }
  
  return success();
}

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }
LogicalResult MaxOp::verify() { return verifyBinaryOp(*this); }
LogicalResult MinOp::verify() { return verifyBinaryOp(*this); }
LogicalResult SubOp::verify() { return verifyBinaryOp(*this); }
LogicalResult MulOp::verify() { return verifyBinaryOp(*this); }
LogicalResult DivOp::verify() { return verifyBinaryOp(*this); }
LogicalResult PowOp::verify() { return verifyBinaryOp(*this); }


//===----------------------------------------------------------------------===//
// ReduceOp Builder and Verification
//===----------------------------------------------------------------------===//

void ReduceOp::build(OpBuilder &builder, OperationState &state,
                     ReductionKind kind, Value input, ArrayRef<int64_t> axes,
                     bool keepdims, Type resultType) {
  state.addOperands(input);
  state.addAttribute("kind", builder.getI32IntegerAttr(static_cast<int32_t>(kind)));
  
  if (!axes.empty()) {
    state.addAttribute("axes", builder.getI64ArrayAttr(axes));
  }
  
  if (keepdims) {
    state.addAttribute("keepdims", builder.getBoolAttr(keepdims));
  }
  
  state.addTypes(resultType);
}

LogicalResult ReduceOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }
    
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType) {
    return emitOpError("output must be a ranked tensor");
  }
  
  int64_t inputRank = inputType.getRank();
  
  // Verify axes are valid
  if (auto axesAttr = getAxesAttr()) {
    llvm::SmallVector<int64_t> axesVec;
    for (auto axis : axesAttr.getAsValueRange<IntegerAttr>()) {
      int64_t axisVal = axis.getZExtValue();
      if (axisVal < 0 || axisVal >= inputRank) {
        return emitOpError("axis ") << axisVal << " is out of range [0, " 
                                     << inputRank << ")";
      }
      axesVec.push_back(axisVal);
    }
    
    // Check for duplicate axes
    llvm::SmallSet<int64_t, 4> uniqueAxes(axesVec.begin(), axesVec.end());
    if (uniqueAxes.size() != axesVec.size()) {
      return emitOpError("duplicate axis found in axes attribute");
    }
  }
  
  // For argmax/argmin, output element type should be integer
  auto kind = getKind();
  if (kind == ReductionKind::ARGMAX || kind == ReductionKind::ARGMIN) {
    if (!outputType.getElementType().isInteger(64)) {
      return emitOpError("argmax/argmin output must have i64 element type, got ")
             << outputType.getElementType();
    }
  } else {
    // For other reductions, element types should match
    if (inputType.getElementType() != outputType.getElementType()) {
      return emitOpError("input and output element types must match for non-arg reductions, got input ")
             << inputType.getElementType() << " and output " 
             << outputType.getElementType();
    }
  }
  
  // Verify output shape matches expected reduced shape
  llvm::SmallVector<int64_t> expectedShape;
  if (auto axesAttr = getAxesAttr()) {
    llvm::SmallSet<int64_t, 4> reducedAxes;
    for (auto axis : axesAttr.getAsValueRange<IntegerAttr>()) {
      reducedAxes.insert(axis.getZExtValue());
    }
    
    for (int64_t i = 0; i < inputRank; ++i) {
      if (reducedAxes.contains(i)) {
        if (getKeepdims()) {
          expectedShape.push_back(1);
        }
      } else {
        expectedShape.push_back(inputType.getDimSize(i));
      }
    }
  } else {
    // No axes specified - reduce all dimensions
    if (getKeepdims()) {
      expectedShape.assign(inputRank, 1);
    }
    // else expectedShape is empty (scalar result)
  }
  
  auto outputShape = outputType.getShape();
  if (outputShape.size() != expectedShape.size()) {
    return emitOpError("output rank does not match expected rank, expected ")
           << expectedShape.size() << " got " << outputShape.size();
  }
  
  for (size_t i = 0; i < expectedShape.size(); ++i) {
    if (expectedShape[i] != outputShape[i] && expectedShape[i] != ShapedType::kDynamic) {
      return emitOpError("output shape mismatch at dimension ") << i
             << ", expected " << expectedShape[i] << " got " << outputShape[i];
    }
  }
  
  return success();
}

SmallVector<Range> DequantOp::getIterationDomain(OpBuilder &b) {
  int64_t rank = getInput().getType().getRank();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(b, getLoc(), getInput());

  SmallVector<Range> loopBounds(rank);
  for (auto dim : llvm::seq<int64_t>(rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = sizes[dim];
    loopBounds[dim].stride = one;
  }

  return loopBounds;
}

SmallVector<utils::IteratorType> DequantOp::getLoopIteratorTypes() {
  int64_t rank = getInput().getType().getRank();
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

LogicalResult DequantOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets = llvm::to_vector(offsets);
  resultSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();
  int64_t rank = getInput().getType().getRank();
  SmallVector<OpFoldResult> strides(rank, b.getI64IntegerAttr(1));

  auto inputTile = b.create<tensor::ExtractSliceOp>(loc, getInput(), offsets,
                                                    sizes, strides);
  auto scaleTile = b.create<tensor::ExtractSliceOp>(loc, getScale(), offsets,
                                                    sizes, strides);

  Type resultType = inputTile.getResultType();

  Operation *tiledOp =
      mlir::clone(b, getOperation(), {resultType}, {inputTile, scaleTile});

  return TilingResult{{tiledOp},
                      SmallVector<Value>(tiledOp->getResults()),
                      {inputTile, scaleTile}};
}

LogicalResult DequantOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  iterDomainOffsets = llvm::to_vector(offsets);
  iterDomainSizes = llvm::to_vector(sizes);
  return success();
}

FailureOr<TilingResult> DequantOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

//------------------------------matmul

LogicalResult MatMulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!lhsType || !rhsType || !outputType) {
    return emitOpError("operands and result must be ranked tensors");
  }
  
  // Check ranks are exactly 2
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outputType.getRank() != 2) {
    return emitOpError("all operands must be 2D tensors (matrices)");
  }
  
  int64_t M = lhsType.getDimSize(0);
  int64_t K_lhs = lhsType.getDimSize(1);
  int64_t K_rhs = rhsType.getDimSize(0);
  int64_t N = rhsType.getDimSize(1);
  
  // Check contraction dimension matches
  if (K_lhs != K_rhs && K_lhs != ShapedType::kDynamic && K_rhs != ShapedType::kDynamic) {
    return emitOpError("contraction dimension mismatch: lhs has ")
           << K_lhs << " but rhs has " << K_rhs;
  }
  
  // Check output shape
  int64_t out_M = outputType.getDimSize(0);
  int64_t out_N = outputType.getDimSize(1);
  
  if (M != out_M && M != ShapedType::kDynamic && out_M != ShapedType::kDynamic) {
    return emitOpError("output dimension 0 mismatch: expected ")
           << M << " but got " << out_M;
  }
  
  if (N != out_N && N != ShapedType::kDynamic && out_N != ShapedType::kDynamic) {
    return emitOpError("output dimension 1 mismatch: expected ")
           << N << " but got " << out_N;
  }
  
  // Check element types match
  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != outputType.getElementType()) {
    return emitOpError("element types must match across all operands");
  }
  
  return success();
}


SmallVector<Range> MatMulOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  Value lhs = getLhs();
  Value rhs = getRhs();
  
  // Get M, N, K dimensions
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  
  SmallVector<OpFoldResult> lhsSizes = tensor::getMixedSizes(b, loc, lhs);
  SmallVector<OpFoldResult> rhsSizes = tensor::getMixedSizes(b, loc, rhs);
  
  OpFoldResult M = lhsSizes[0];  // Rows of LHS
  OpFoldResult N = rhsSizes[1];  // Cols of RHS
  OpFoldResult K = lhsSizes[1];  // Contraction dimension
  
  // Return 3 ranges: [M, N, K]
  // M and N are parallel, K is reduction
  SmallVector<Range> loopBounds(3);
  loopBounds[0] = Range{zero, M, one};  // M dimension
  loopBounds[1] = Range{zero, N, one};  // N dimension
  loopBounds[2] = Range{zero, K, one};  // K dimension (reduction)
  
  return loopBounds;
}

SmallVector<utils::IteratorType> MatMulOp::getLoopIteratorTypes() {
  // M and N are parallel (can be tiled independently)
  // K is reduction (must accumulate sequentially)
  return {
    utils::IteratorType::parallel,   // M
    utils::IteratorType::parallel,   // N
    utils::IteratorType::reduction   // K
  };
}

LogicalResult MatMulOp::getIterationDomainTileFromResultTile(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
    SmallVectorImpl<OpFoldResult> &iterDomainSizes) {
  
  // Result has shape [M, N]
  // Iteration domain is [M, N, K]
  // When tiling result at [m_offset, n_offset] with size [m_size, n_size],
  // we tile iteration domain at [m_offset, n_offset, 0] with size [m_size, n_size, K]
  
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult K = tensor::getMixedSizes(b, getLoc(), getLhs())[1];
  
  iterDomainOffsets.assign({offsets[0], offsets[1], zero});
  iterDomainSizes.assign({sizes[0], sizes[1], K});
  
  return success();
}

LogicalResult MatMulOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  
  // Iteration domain is [M, N, K]
  // Result is [M, N]
  // Just drop the K dimension
  resultOffsets = {offsets[0], offsets[1]};
  resultSizes = {sizes[0], sizes[1]};
  
  return success();
}

FailureOr<TilingResult> MatMulOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  
  Location loc = getLoc();
  
  // offsets = [m_offset, n_offset, k_offset]
  // sizes = [m_size, n_size, k_size]
  
  // Extract tile from LHS: A[m_offset:m_offset+m_size, k_offset:k_offset+k_size]
  SmallVector<OpFoldResult> lhsOffsets = {offsets[0], offsets[2]};
  SmallVector<OpFoldResult> lhsSizes = {sizes[0], sizes[2]};
  SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
  
  auto lhsTile = b.create<tensor::ExtractSliceOp>(
      loc, getLhs(), lhsOffsets, lhsSizes, strides);
  
  // Extract tile from RHS: B[k_offset:k_offset+k_size, n_offset:n_offset+n_size]
  SmallVector<OpFoldResult> rhsOffsets = {offsets[2], offsets[1]};
  SmallVector<OpFoldResult> rhsSizes = {sizes[2], sizes[1]};
  
  auto rhsTile = b.create<tensor::ExtractSliceOp>(
      loc, getRhs(), rhsOffsets, rhsSizes, strides);
  
  // Compute result type for the tile
  auto lhsTileType = lhsTile.getType();
  auto rhsTileType = rhsTile.getType();
  
  SmallVector<int64_t> resultShape = {
    lhsTileType.getDimSize(0),  // m_size
    rhsTileType.getDimSize(1)   // n_size
  };
  
  auto resultTileType = RankedTensorType::get(
      resultShape, lhsTileType.getElementType());
  
  // Create tiled matmul operation
  Operation *tiledOp = mlir::clone(b, getOperation(), {resultTileType},
                                   {lhsTile, rhsTile});
  
  return TilingResult{
    {tiledOp},                                    // Tiled operations
    SmallVector<Value>(tiledOp->getResults()),   // Results
    {lhsTile, rhsTile}                            // Extracted slices
  };
}

FailureOr<TilingResult> MatMulOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  
  SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
  if (failed(getIterationDomainTileFromResultTile(
          b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes))) {
    return failure();
  }
  
  return getTiledImplementation(b, mappedOffsets, mappedSizes);
}

//===----------------------------------------------------------------------===//
// Softmax Verification
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !outputType) {
    return emitOpError("input and output must be ranked tensors");
  }
  
  // Check axis is valid
  int64_t axis = getAxis();
  int64_t rank = inputType.getRank();
  if (axis < 0 || axis >= rank) {
    return emitOpError("axis ") << axis << " is out of range [0, " << rank << ")";
  }
  
  // Check input and output shapes match
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("input and output shapes must match");
  }
  
  // Check element types match
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Reshape Verification
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !outputType) {
    return emitOpError("input and output must be ranked tensors");
  }
  
  // Calculate total elements
  int64_t inputElements = 1;
  for (int64_t dim : inputType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      inputElements = ShapedType::kDynamic;
      break;
    }
    inputElements *= dim;
  }
  
  int64_t outputElements = 1;
  for (int64_t dim : outputType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      outputElements = ShapedType::kDynamic;
      break;
    }
    outputElements *= dim;
  }
  
  // Check total elements match (if not dynamic)
  if (inputElements != ShapedType::kDynamic && outputElements != ShapedType::kDynamic) {
    if (inputElements != outputElements) {
      return emitOpError("total elements must match: input has ")
             << inputElements << " but output has " << outputElements;
    }
  }
  
  // Check element types match
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Transpose Verification
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !outputType) {
    return emitOpError("input and output must be ranked tensors");
  }
  
  int64_t rank = inputType.getRank();
  auto perm = getPerm();
  
  // Check permutation has correct size
  if (static_cast<int64_t>(perm.size()) != rank) {
    return emitOpError("permutation size ")
           << perm.size() << " does not match input rank " << rank;
  }
  
  // Check permutation is valid (all dims present, no duplicates)
  llvm::SmallSet<int64_t, 8> seen;
  for (auto p : perm.getAsValueRange<IntegerAttr>()) {
    int64_t val = p.getZExtValue();
    if (val < 0 || val >= rank) {
      return emitOpError("permutation value ")
             << val << " is out of range [0, " << rank << ")";
    }
    if (!seen.insert(val).second) {
      return emitOpError("duplicate value ") << val << " in permutation";
    }
  }
  
  // Check output shape matches permuted input shape
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int i = 0;
  for (auto p : perm.getAsValueRange<IntegerAttr>()) {
    int64_t permVal = p.getZExtValue();
    if (inputShape[permVal] != outputShape[i] && 
        inputShape[permVal] != ShapedType::kDynamic &&
        outputShape[i] != ShapedType::kDynamic) {
      return emitOpError("output shape mismatch at dimension ") << i;
    }
    i++;
  }
  
  // Check element types match
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Concat Verification
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  auto inputs = getInputs();
  if (inputs.empty()) {
    return emitOpError("requires at least one input");
  }
  
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType) {
    return emitOpError("output must be a ranked tensor");
  }
  
  int64_t axis = getAxis();
  
  // Get first input type as reference
  auto firstType = dyn_cast<RankedTensorType>(inputs[0].getType());
  if (!firstType) {
    return emitOpError("all inputs must be ranked tensors");
  }
  
  int64_t rank = firstType.getRank();
  if (axis < 0 || axis >= rank) {
    return emitOpError("axis ") << axis << " is out of range [0, " << rank << ")";
  }
  
  // Check all inputs have same rank and compatible shapes
  int64_t concatDimSize = firstType.getDimSize(axis);
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = dyn_cast<RankedTensorType>(inputs[i].getType());
    if (!inputType) {
      return emitOpError("all inputs must be ranked tensors");
    }
    
    if (inputType.getRank() != rank) {
      return emitOpError("all inputs must have the same rank");
    }
    
    // Check non-concat dimensions match
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis) continue;
      if (inputType.getDimSize(d) != firstType.getDimSize(d) &&
          inputType.getDimSize(d) != ShapedType::kDynamic &&
          firstType.getDimSize(d) != ShapedType::kDynamic) {
        return emitOpError("dimension ") << d << " mismatch between inputs";
      }
    }
    
    // Accumulate concat dimension size
    if (concatDimSize != ShapedType::kDynamic && 
        inputType.getDimSize(axis) != ShapedType::kDynamic) {
      concatDimSize += inputType.getDimSize(axis);
    } else {
      concatDimSize = ShapedType::kDynamic;
    }
    
    // Check element types match
    if (inputType.getElementType() != firstType.getElementType()) {
      return emitOpError("all inputs must have the same element type");
    }
  }
  
  // Check output shape
  auto outputShape = outputType.getShape();
  if (concatDimSize != ShapedType::kDynamic && 
      outputShape[axis] != ShapedType::kDynamic &&
      concatDimSize != outputShape[axis]) {
    return emitOpError("output concat dimension size mismatch");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Slice Verification
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !outputType) {
    return emitOpError("input and output must be ranked tensors");
  }
  
  int64_t rank = inputType.getRank();
  auto start = getStart();
  auto end = getEnd();
  
  // Check start and end have correct size
  if (static_cast<int64_t>(start.size()) != rank) {
    return emitOpError("start array size must match input rank");
  }
  if (static_cast<int64_t>(end.size()) != rank) {
    return emitOpError("end array size must match input rank");
  }
  
  // Check indices are valid
  int i = 0;
  for (auto [s, e] : llvm::zip(start.getAsValueRange<IntegerAttr>(), 
                                end.getAsValueRange<IntegerAttr>())) {
    int64_t startVal = s.getZExtValue();
    int64_t endVal = e.getZExtValue();
    int64_t dimSize = inputType.getDimSize(i);
    
    if (dimSize != ShapedType::kDynamic) {
      if (startVal < 0 || startVal > dimSize) {
        return emitOpError("start index ") << startVal 
               << " out of bounds for dimension " << i;
      }
      if (endVal < 0 || endVal > dimSize) {
        return emitOpError("end index ") << endVal 
               << " out of bounds for dimension " << i;
      }
      if (startVal > endVal) {
        return emitOpError("start index must be <= end index for dimension ") << i;
      }
    }
    i++;
  }
  
  // Check element types match
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Broadcast Verification
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !outputType) {
    return emitOpError("input and output must be ranked tensors");
  }
  
  // Check element types match
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  
  // Check broadcasting is valid (output rank >= input rank)
  if (outputType.getRank() < inputType.getRank()) {
    return emitOpError("output rank must be >= input rank for broadcasting");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Select Verification
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() {
  auto condType = dyn_cast<RankedTensorType>(getCondition().getType());
  auto trueType = dyn_cast<RankedTensorType>(getTrueVal().getType());
  auto falseType = dyn_cast<RankedTensorType>(getFalseVal().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!condType || !trueType || !falseType || !outputType) {
    return emitOpError("all operands and result must be ranked tensors");
  }
  
  // Check condition is boolean
  if (!condType.getElementType().isInteger(1)) {
    return emitOpError("condition must have i1 element type");
  }
  
  // Check true and false values have same type
  if (trueType.getShape() != falseType.getShape()) {
    return emitOpError("true and false values must have the same shape");
  }
  if (trueType.getElementType() != falseType.getElementType()) {
    return emitOpError("true and false values must have the same element type");
  }
  
  // Check output matches true/false type
  if (outputType.getShape() != trueType.getShape()) {
    return emitOpError("output shape must match true/false value shapes");
  }
  if (outputType.getElementType() != trueType.getElementType()) {
    return emitOpError("output element type must match true/false value element types");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Clamp Verification
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto minType = dyn_cast<RankedTensorType>(getMin().getType());
  auto maxType = dyn_cast<RankedTensorType>(getMax().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  
  if (!inputType || !minType || !maxType || !outputType) {
    return emitOpError("all operands and result must be ranked tensors");
  }
  
  // Check element types match
  if (inputType.getElementType() != minType.getElementType() ||
      inputType.getElementType() != maxType.getElementType() ||
      inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("all operands and result must have the same element type");
  }
  
  // Check output shape matches input
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("output shape must match input shape");
  }
  
  return success();
}
