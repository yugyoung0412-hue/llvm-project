add_llvm_component_library(LLVMTensorize
  LoopTensorizationLegality.cpp
  LoopTensorize.cpp
  LoopTensorizeCostModel.cpp
  TPattern.cpp
  TPlan.cpp
  TPlanAnalysis.cpp
  TPlanner.cpp
  TPlanRecipes.cpp
  TPlanDecisionLogic.cpp
  TPlanSpace.cpp
  TPlanTransforms.cpp
  TPlanVerifier.cpp
  TPRecipeBuilder.cpp
  TPRecipeMatcher.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tensorize

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  )
