//===- LoopTensorize.cpp - Loop Tensorization Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopTensorize.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

PreservedAnalyses LoopTensorizePass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  if (!Opts.Enabled)
    return PreservedAnalyses::all();

  // TODO: implement
  return PreservedAnalyses::all();
}
