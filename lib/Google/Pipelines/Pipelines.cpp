//===- Pipelines.cpp - Google Optimization Pipelines ---------------------===//
//
// This file implements optimization pipeline configurations for the Google
// dialect using existing MLIR passes.
//
//===----------------------------------------------------------------------===//

#include "Google/Pipelines/Pipelines.h"
#include "Google/IR/GoogleOps.h"
#include "Google/Translation/GoogleToLinalg.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace google {

//===----------------------------------------------------------------------===//
// Basic Pipeline - Fast Compilation
//===----------------------------------------------------------------------===//

void registerBasicPipeline() {
  PassPipelineRegistration<>(
    "google-basic-pipeline",
    "Basic Google optimization pipeline for fast compilation",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: Basic fusion
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 4: Lower to loops
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      
      // Phase 5: Lower to LLVM (using pass pipeline string)
      // Note: Some passes are better invoked via pipeline strings
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Optimized Pipeline - Balanced Performance
//===----------------------------------------------------------------------===//

void registerOptimizedPipeline() {
  PassPipelineRegistration<>(
    "google-optimized-pipeline",
    "Optimized Google pipeline with fusion and affine optimizations",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: Linalg optimizations
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 3: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 4: Lower to affine for better loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 5: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L1 - L1 Cache Tiling with Transform Dialect
//===----------------------------------------------------------------------===//

void registerExtremePipelineL1() {
  PassPipelineRegistration<>(
    "google-extreme-l1",
    "Extreme pipeline with L1 cache tiling (16x16x16) - MINIMAL for testing",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: TILING via Transform Dialect
      // Stop here to verify tiling works
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // STOP HERE - no bufferization, no LLVM lowering
      // This allows us to see the tiled linalg operations
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L2 - L1+L2 Cache Tiling with Transform Dialect
//===----------------------------------------------------------------------===//

void registerExtremePipelineL2() {
  PassPipelineRegistration<>(
    "google-extreme-l2",
    "Extreme pipeline with L1+L2 cache tiling (L2: 64x64x64, L1: 16x16x16)",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2 TILING via Transform Dialect
      // The transform script should be embedded in the input MLIR file
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // STOP HERE - minimal pipeline to verify 2-level tiling
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline L2 Full - Complete L1+L2 Tiling with LLVM Lowering
//===----------------------------------------------------------------------===//

void registerExtremePipelineL2Full() {
  PassPipelineRegistration<>(
    "google-extreme-l2-full",
    "Complete extreme pipeline with L1+L2 tiling and LLVM lowering",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: FUSION (critical - must happen before tiling)
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      
      // Phase 3: L1+L2 TILING via Transform Dialect
      pm.addPass(mlir::transform::createInterpreterPass());
      
      // Phase 4: Generalize named ops for further optimization
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 5: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 6: Lower to affine for loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 7: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Extreme Pipeline - Maximum Performance with Fusion and Coalescing
//===----------------------------------------------------------------------===//

void registerExtremePipeline() {
  PassPipelineRegistration<>(
    "google-extreme-pipeline",
    "Extreme optimization with aggressive fusion and coalescing",
    [](OpPassManager &pm) {
      // Phase 1: Lower Google to Linalg
      pm.addPass(createGoogleToLinalgLoweringPass());
      
      // Phase 2: Linalg optimizations
      pm.addNestedPass<func::FuncOp>(createLinalgElementwiseOpFusionPass());
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      
      // Phase 3: Bufferization
      pm.addPass(bufferization::createOneShotBufferizePass());
      
      // Phase 4: Lower to affine for loop optimization
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
      
      // Affine optimizations
      pm.addNestedPass<func::FuncOp>(affine::createLoopFusionPass());
      pm.addNestedPass<func::FuncOp>(affine::createLoopCoalescingPass());
      
      // NOTE: Tiling will be added via Transform dialect in next iteration
      // The current MLIR version doesn't expose simple tiling pass APIs
      
      // Lower affine
      pm.addPass(createLowerAffinePass());
      
      // Phase 5: Lower to LLVM
      pm.addPass(createConvertFuncToLLVMPass());
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
}

//===----------------------------------------------------------------------===//
// Pipeline Registration
//===----------------------------------------------------------------------===//

void registerGooglePipelines() {
  registerBasicPipeline();
  registerOptimizedPipeline();
  registerExtremePipeline();
  registerExtremePipelineL1();
  registerExtremePipelineL2();
  registerExtremePipelineL2Full();
}

} // namespace google
} // namespace mlir
