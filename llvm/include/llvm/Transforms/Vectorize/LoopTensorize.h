//===- LoopTensorize.h - Loop Tensorization Pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPTENSORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPTENSORIZE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LoopTensorizeOptions {
  unsigned BeamWidth = 8;
  bool     Enabled   = true;
};

class LoopTensorizePass : public PassInfoMixin<LoopTensorizePass> {
  LoopTensorizeOptions Opts;

public:
  explicit LoopTensorizePass(LoopTensorizeOptions Opts = {}) : Opts(Opts) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm
#endif
