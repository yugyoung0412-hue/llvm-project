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
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TensorCodeGen.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"

#define DEBUG_TYPE "loop-tensorize"

using namespace llvm;

static cl::opt<unsigned> OverridePF(
    "loop-tensorize-pf",
    cl::desc("Override parallel factor (tile size) for all dimensions. "
             "0 = use default (256). Intended for testing only."),
    cl::init(0));

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

    // Build and print the initial TPlan.
    TPlan Plan = TPlan::buildInitial(*InfoOpt);
    LLVM_DEBUG(Plan.print(dbgs()));

    // Lower to IR.  TPlanWidener_widen() (called inside lower()) seeds
    // Plan.DimPFMap with the default PF=256 for each IV dimension.
    // Override specific dims here if a target-specific tile size is needed.

    // Apply command-line PF override for testing.
    // hasDimPF() in the widener will preserve these values.
    if (OverridePF > 0)
      for (unsigned D = 0; D < InfoOpt->Depth; ++D)
        Plan.setDimPF(D, OverridePF);

    TPlanLowering_lower(Plan, F, LI, SE, DT);

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
    Initial.Cost = std::numeric_limits<float>::infinity();
    Initial.IsTerminal = false;

    SearchState Best =
        runBeamSearch(Initial, EffectiveOps, Params, Opts.BeamWidth);

    LLVM_DEBUG(dbgs() << "Best cost: " << Best.Cost
                      << " transforms: " << Best.Applied.size() << "\n");

    bool PlanChanged = applyPlan(Best, Hint, EffectiveOps, F, LI, SE, DT);
    Changed |= PlanChanged;
    LLVM_DEBUG(F.print(dbgs()));
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
