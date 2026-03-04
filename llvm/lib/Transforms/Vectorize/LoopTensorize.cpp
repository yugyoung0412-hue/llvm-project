//===- LoopTensorize.cpp - Loop Tensorization Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopTensorize.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorCodeGen.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

#define DEBUG_TYPE "loop-tensorize"

using namespace llvm;

PreservedAnalyses LoopTensorizePass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  if (!Opts.Enabled)
    return PreservedAnalyses::all();

  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DI = FAM.getResult<DependenceAnalysis>(F);
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);

  bool Changed = false;
  for (auto &RawNest : collectLoopNests(LI)) {
    auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
    if (!InfoOpt)
      continue;

    // Build a TPlan from the loop nest analysis; this becomes the primary IR.
    auto Plan = TPlanBuilder_build(*InfoOpt, PatternHint{}, TensorOpDesc{}, {},
                                   F.getContext());

    // Classify based on the TPlan when available; fall back to legacy path.
    PatternHint Hint = Plan ? classifyPattern(*Plan) : classifyPattern(*InfoOpt);
    LLVM_DEBUG(dbgs() << "PatternHint: "
      << (Hint.Kind == PatternKind::GEMM        ? "GEMM"
        : Hint.Kind == PatternKind::Elementwise ? "Elementwise"
        :                                         "Generic")
      << "\n");

    // Pick element type from the first memory access, default to float.
    Type *ElemTy = nullptr;
    if (!InfoOpt->Accesses.empty())
      ElemTy = InfoOpt->Accesses[0].ElemType;
    if (!ElemTy)
      ElemTy = Type::getFloatTy(F.getContext());

    SmallVector<TensorOpDesc> SupportedOps = TTI.getSupportedTensorOps();
    TensorCostModelParams Params = buildCostParams(TTI, ElemTy);

    // If no hardware tensor ops are available but pattern is GEMM, synthesize
    // a generic MatMul descriptor using llvm.matrix.multiply.
    SmallVector<TensorOpDesc> EffectiveOps(SupportedOps);
    if (EffectiveOps.empty() && Hint.Kind == PatternKind::GEMM) {
      TensorOpDesc Synthetic;
      Synthetic.OpKind = TensorOpDesc::Kind::MatMul;
      Synthetic.IntrinsicID = Intrinsic::matrix_multiply;
      EffectiveOps.push_back(Synthetic);
      Params.PeakTensorFLOPS = Params.PeakVectorFLOPS * 2.0f;
    }

    SearchState Initial;
    Initial.Current = *InfoOpt;
    Initial.Plan    = Plan.get(); // may be null; Plan keeps ownership
    Initial.Cost = std::numeric_limits<float>::infinity();
    Initial.IsTerminal = false;

    SearchState Best =
        runBeamSearch(Initial, EffectiveOps, Params, Opts.BeamWidth);

    LLVM_DEBUG(dbgs() << "Best cost: " << Best.Cost
                      << " transforms: " << Best.Applied.size() << "\n");

    Changed |= applyPlan(Best, Hint, EffectiveOps, F, LI, SE, DT);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
