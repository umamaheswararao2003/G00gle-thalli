
add_mlir_library(MLIRGoogleToArithTransforms
  GoogleToArith.cpp
  GoogleToArith.h
  GoogletoTosa.cpp
  GoogletoTosa.h
  GoogleToLinalg.cpp
  GoogleToLinalg.h

  DEPENDS
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)

target_include_directories(MLIRGoogleToArithTransforms PRIVATE
  ${PROJECT_SOURCE_DIR}/include/Google/Translation
)
