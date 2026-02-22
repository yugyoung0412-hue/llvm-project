//===- LoopTensorize.cpp - Loop Tensorization Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopTensorize.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"

#define DEBUG_TYPE "loop-tensorize"

using namespace llvm;

PreservedAnalyses LoopTensorizePass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  if (!Opts.Enabled)
    return PreservedAnalyses::all();

  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DI = FAM.getResult<DependenceAnalysis>(F);

  for (auto &RawNest : collectLoopNests(LI)) {
    auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
    if (!InfoOpt)
      continue;
    PatternHint Hint = classifyPattern(*InfoOpt);
    LLVM_DEBUG(dbgs() << "PatternHint: "
      << (Hint.Kind == PatternKind::GEMM        ? "GEMM"
        : Hint.Kind == PatternKind::Elementwise ? "Elementwise"
        :                                         "Generic")
      << "\n");
  }

  return PreservedAnalyses::all();
}
