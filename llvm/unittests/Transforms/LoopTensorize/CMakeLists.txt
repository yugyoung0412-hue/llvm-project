set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  TransformUtils
  Vectorize
  )

add_llvm_unittest(LoopTensorizeTests
  TensorISAInfoTest.cpp
  LoopNestAnalyzerTest.cpp
  PatternClassifierTest.cpp
  TransformSpaceTest.cpp
  )
