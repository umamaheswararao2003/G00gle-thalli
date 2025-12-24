//===- GoogleToLinalg.h - Google to Linalg conversion ----------*- C++ -*-===//
//
// This file declares the pass for lowering Google dialect operations to
// Linalg, Tensor, Arith, and Math dialects.
//
//===----------------------------------------------------------------------===//

#ifndef GOOGLE_TRANSLATION_GOOGLETOLINALG_H
#define GOOGLE_TRANSLATION_GOOGLETOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class RewritePatternSet;

namespace google {

/// Populate patterns for lowering Google dialect ops to Linalg + Arith/Math/Tensor
void populateGoogleToLinalgConversionPatterns(RewritePatternSet &patterns);

/// Create the Google to Linalg lowering pass
std::unique_ptr<Pass> createGoogleToLinalgLoweringPass();

/// Register the pass
void registerGoogleToLinalgLoweringPass();

} // namespace google
} // namespace mlir

#endif // GOOGLE_TRANSLATION_GOOGLETOLINALG_H
