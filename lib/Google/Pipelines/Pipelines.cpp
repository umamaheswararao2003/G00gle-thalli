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
// Pipeline Registration
//===----------------------------------------------------------------------===//

void registerGooglePipelines() {
  registerBasicPipeline();
  registerOptimizedPipeline();
}

} // namespace google
} // namespace mlir
