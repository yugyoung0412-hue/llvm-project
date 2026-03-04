//===- TensorPatternClassifier.h - Tensor pattern classification ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORPATTERNCLASSIFIER_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORPATTERNCLASSIFIER_H

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"

namespace llvm {

class TPlan;

enum class PatternKind {
  GEMM,
  Conv2D,
  Elementwise,
  Reduction,
  Generic
};

struct PatternHint {
  PatternKind Kind = PatternKind::Generic;
  /// Index into TTI.getSupportedTensorOps() to try first; -1 = no hint.
  int PreferredOpIdx = -1;
  /// True when col_matrix fits in L2 and im2col -> GEMM lowering is preferred.
  /// Only meaningful when Kind == Conv2D.
  bool UseIm2Col = false;
};

/// Classifies a LoopNestInfo into a PatternHint.
PatternHint classifyPattern(const LoopNestInfo &Info);

/// Classifies a TPlan (via its recipes) into a PatternHint.
/// Also sets TPComputeRecipe::Pattern in-place.
PatternHint classifyPattern(TPlan &Plan);

} // namespace llvm
#endif
