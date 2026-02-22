set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  Vectorize
  )

add_llvm_unittest(LoopTensorizeTests
  TensorISAInfoTest.cpp
  )
