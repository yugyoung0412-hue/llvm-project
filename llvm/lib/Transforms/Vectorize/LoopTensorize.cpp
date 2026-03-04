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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
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

    // Pick element type from the first memory access, default to float.
    Type *ElemTy = nullptr;
    if (!InfoOpt->Accesses.empty())
      ElemTy = InfoOpt->Accesses[0].ElemType;
    if (!ElemTy)
      ElemTy = Type::getFloatTy(F.getContext());

    SmallVector<TensorOpDesc> SupportedOps = TTI.getSupportedTensorOps();
    TensorCostModelParams Params = buildCostParams(TTI, ElemTy);

    // --- TPlan path (new) ---
    {
      TPlan Plan = TPlan::buildInitial(*InfoOpt);
      PatternHint TPlanHint = classifyPattern(Plan);
      LLVM_DEBUG(dbgs() << "TPlan: classifyPattern: "
        << (TPlanHint.Kind == PatternKind::GEMM        ? "GEMM"
          : TPlanHint.Kind == PatternKind::Conv2D      ? "Conv2D"
          : TPlanHint.Kind == PatternKind::Elementwise ? "Elementwise"
          :                                              "Generic")
        << "\n");

      // Build EffectiveOps for TPlan search (same logic as legacy path).
      SmallVector<TensorOpDesc> TPlanEffectiveOps(SupportedOps);
      if (TPlanEffectiveOps.empty() &&
          TPlanHint.Kind == PatternKind::GEMM) {
        TensorOpDesc Synthetic;
        Synthetic.OpKind = TensorOpDesc::Kind::MatMul;
        Synthetic.IntrinsicID = Intrinsic::matrix_multiply;
        TPlanEffectiveOps.push_back(Synthetic);
        Params.PeakTensorFLOPS = Params.PeakVectorFLOPS * 2.0f;
      }

      TPlan BestTPlan =
          searchTPlan(std::move(Plan), TPlanEffectiveOps, Params, Opts.BeamWidth);
      if (applyTPlan(BestTPlan, F, LI, SE, DT)) {
        Changed = true;
        continue; // skip legacy path
      }
    }

    // --- Legacy path (fallback) ---
    PatternHint Hint = classifyPattern(*InfoOpt);
    LLVM_DEBUG(dbgs() << "PatternHint: "
      << (Hint.Kind == PatternKind::GEMM        ? "GEMM"
        : Hint.Kind == PatternKind::Conv2D      ? "Conv2D"
        : Hint.Kind == PatternKind::Elementwise ? "Elementwise"
        :                                         "Generic")
      << "\n");

      // Conv2D lowering decision: estimate col_matrix size and choose
      // im2col -> GEMM when it fits in L2 cache, direct tile otherwise.
      if (Hint.Kind == PatternKind::Conv2D) {
        uint64_t ColMatrixBytes = 1;
        bool AllConstant = true;
        for (const auto &IV : InfoOpt->IVs) {
          if (const auto *SC = dyn_cast<SCEVConstant>(IV.TripCount))
            ColMatrixBytes *= SC->getValue()->getZExtValue() + 1;
          else { AllConstant = false; break; }
        }
        Type *TmpElemTy = InfoOpt->Accesses.empty()
                              ? Type::getFloatTy(F.getContext())
                              : InfoOpt->Accesses[0].ElemType;
        if (!TmpElemTy) TmpElemTy = Type::getFloatTy(F.getContext());
        uint64_t ElemBytes = TmpElemTy->getPrimitiveSizeInBits() / 8;
        if (ElemBytes == 0) ElemBytes = 4;
        ColMatrixBytes *= ElemBytes;

        uint64_t L2Size = 262144; // 256 KiB fallback
        if (auto MaybeL2 = TTI.getCacheSize(TargetTransformInfo::CacheLevel::L2D))
          L2Size = *MaybeL2;

        Hint.UseIm2Col = AllConstant && (ColMatrixBytes <= L2Size);

        LLVM_DEBUG(dbgs() << "Conv2D: col_matrix_bytes=" << ColMatrixBytes
                          << " L2=" << L2Size
                          << " use_im2col=" << Hint.UseIm2Col << "\n");
      }

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
