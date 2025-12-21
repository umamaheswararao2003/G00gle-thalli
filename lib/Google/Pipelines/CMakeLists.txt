add_mlir_library(MLIRGooglePipelines
  Pipelines.cpp

  LINK_LIBS PUBLIC
  MLIRGoogleOps
  MLIRGoogleToArithTransforms
  MLIRAffineTransforms
  MLIRArithTransforms
  MLIRBufferizationTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRSCFTransforms
  MLIRPass
  MLIRTransforms
)
