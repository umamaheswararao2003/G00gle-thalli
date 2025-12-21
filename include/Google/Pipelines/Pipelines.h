//===- Pipelines.h - Google Optimization Pipelines -------------*- C++ -*-===//
//
// This file declares optimization pipeline configurations for the Google
// dialect, leveraging existing MLIR optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef GOOGLE_PIPELINES_PIPELINES_H
#define GOOGLE_PIPELINES_PIPELINES_H

namespace mlir {
namespace google {

/// Register all Google optimization pipelines
void registerGooglePipelines();

/// Register basic pipeline (fast compilation)
void registerBasicPipeline();

/// Register optimized pipeline (balanced performance)
void registerOptimizedPipeline();

/// Register extreme pipeline (maximum performance)
void registerExtremePipeline();

} // namespace google
} // namespace mlir

#endif // GOOGLE_PIPELINES_PIPELINES_H
