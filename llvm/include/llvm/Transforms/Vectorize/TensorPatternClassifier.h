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
};

/// Classifies a LoopNestInfo into a PatternHint.
PatternHint classifyPattern(const LoopNestInfo &Info);

// Forward-declare TPlan to avoid circular include (TPlan.h includes this file).
class TPlan;

/// Classifies a TPlan's recipe structure into a PatternHint.
/// This is the primary interface once a TPlan has been built.
PatternHint classifyPattern(const TPlan &Plan);

} // namespace llvm
#endif
