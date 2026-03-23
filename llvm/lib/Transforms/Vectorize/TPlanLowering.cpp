//===- TPlanLowering.cpp - Lower TPlan to LLVM IR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements execute() for every TPlan recipe kind and the
/// TPlanLowering_lower() entry point.
///
/// Pipeline inside TPlanLowering_lower():
///   1. TPlanWidener_widen()           — propagate DimSets via union BFS
///   2. TPRecipePatternMatcher_match() — classify every recipe
///   3. Execute recipes in program order (depth-first region walk)
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-lower"

//===----------------------------------------------------------------------===//
// Helper: emit @llvm.matrix.multiply for a Contraction reduction
//===----------------------------------------------------------------------===//

static Value *emitContraction(const TPRecipeBase *FusedMul,
                               const TPRecipeBase *ReductionUpdate,
                               TPTransformState &State) {
  if (!FusedMul || FusedMul->operands().size() < 2)
    return nullptr;

  auto *LHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(0));
  auto *RHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(1));
  if (!LHSDR || !RHSDR)
    return nullptr;

  // Dimension safety: require exactly 2D operands.
  if (LHSDR->DimSet.count() != 2 || RHSDR->DimSet.count() != 2) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction requires 2D operands, "
                         "falling back to scalar clone\n");
    return nullptr;
  }

  Value *LHS = State.getValue(LHSDR);
  Value *RHS = State.getValue(RHSDR);
  if (!LHS || !RHS) return nullptr;

  SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDR, State.Plan);
  SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDR, State.Plan);

  int ContractDim = State.getContractDim(ReductionUpdate);

  // Find position of ContractDim in each operand's sorted DimSet.
  auto findPos = [](const SmallBitVector &DS, int Dim) -> unsigned {
    unsigned Pos = 0;
    for (int D = DS.find_first(); D >= 0; D = DS.find_next(D), ++Pos)
      if (D == Dim) return Pos;
    return 0;
  };
  unsigned LHSPos = findPos(LHSDR->DimSet, ContractDim);
  unsigned RHSPos = findPos(RHSDR->DimSet, ContractDim);
  unsigned M = LHSShape[1 - LHSPos];
  unsigned K = LHSShape[LHSPos];
  unsigned N = RHSShape[1 - RHSPos];

  // Ensure operands are flat FixedVectorType as required by the intrinsic.
  Type *ElemTy = LHS->getType()->getScalarType();
  auto ensureFlat = [&](Value *V, unsigned Elems) -> Value * {
    Type *FlatTy = FixedVectorType::get(ElemTy, Elems);
    if (V->getType() != FlatTy)
      return State.Builder.CreateBitCast(V, FlatTy);
    return V;
  };
  LHS = ensureFlat(LHS, M * K);
  RHS = ensureFlat(RHS, K * N);

  // Emit @llvm.matrix.multiply(LHS, RHS, M, K, N)
  Type *ResTy = FixedVectorType::get(ElemTy, M * N);
  Function *MatMulFn = Intrinsic::getOrInsertDeclaration(
      State.Builder.GetInsertBlock()->getModule(),
      Intrinsic::matrix_multiply,
      {ResTy, LHS->getType(), RHS->getType()});
  return State.Builder.CreateCall(
      MatMulFn,
      {LHS, RHS,
       State.Builder.getInt32(M),
       State.Builder.getInt32(K),
       State.Builder.getInt32(N)});
}

//===----------------------------------------------------------------------===//
// execute() implementations per recipe kind
//===----------------------------------------------------------------------===//

void TPCanonicalIVRecipe::execute(TPTransformState &) const {
  // Canonical IV is handled by the loop structure — no direct IR emission.
}

void TPCanonicalIVIncrRecipe::execute(TPTransformState &) const {
  // Same as above — no direct IR emission.
}

void TPCanonicalIVExitCmpRecipe::execute(TPTransformState &) const {
  // Same as above — no direct IR emission.
}

void TPWidenInductionRecipe::execute(TPTransformState &State) const {
  // IV values are loop PHIs already present in IR; register them in ValueMap.
  State.setValue(this, IVPhi);
}

void TPReductionPHIRecipe::execute(TPTransformState &State) const {
  State.setValue(this, RedPhi);
}

void TPWidenCastRecipe::execute(TPTransformState &State) const {
  auto *SrcDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
  Value *Src = SrcDR ? State.getValue(SrcDR) : nullptr;
  if (!Src) return;
  Value *Result = State.Builder.Insert(CastInst->clone());
  State.setValue(this, Result);
}

void TPWidenGEPRecipe::execute(TPTransformState &State) const {
  Value *Result = State.Builder.Insert(GEPInst->clone());
  State.setValue(this, Result);
}

void TPWidenLoadRecipe::execute(TPTransformState &State) const {
  Value *Result = State.Builder.Insert(LoadInst->clone());
  State.setValue(this, Result);
}

void TPWidenStoreRecipe::execute(TPTransformState &State) const {
  State.Builder.Insert(StoreInst->clone());
}

void TPWidenRecipe::execute(TPTransformState &State) const {
  TensorOpKind Kind = State.getKind(this);

  switch (Kind) {
  case TensorOpKind::Contraction: {
    // For the fused mul (fmul): deferred to its reduction consumer — no-op.
    // For the reduction update (fadd): emit @llvm.matrix.multiply.
    TPRecipeBase *FusedMul = State.getFusedMulRecipe(this);
    if (FusedMul) {
      Value *Result = emitContraction(FusedMul, this, State);
      if (Result)
        State.setValue(this, Result);
    }
    // else: this is the fmul itself (FusedMulRecipe==nullptr here) — no-op.
    return;
  }

  case TensorOpKind::ElementWise:
  case TensorOpKind::Scalar: {
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::BroadcastBinary: {
    // TODO: emit broadcast intrinsic. For now, clone scalar op.
    LLVM_DEBUG(dbgs() << "TPlanLowering: BroadcastBinary not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::OuterProduct: {
    // TODO: emit outer product intrinsic. For now, clone scalar op.
    LLVM_DEBUG(dbgs() << "TPlanLowering: OuterProduct not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::PlainReduction: {
    // Reduction update with no fuseable mul-like producer — clone as scalar.
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(this, Result);
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Region walker
//===----------------------------------------------------------------------===//

static void lowerRegion(const TPLoopRegion *Region, TPTransformState &State) {
  if (!Region)
    return;
  for (const TPRecipeBase &R : Region->getRecipes())
    R.execute(State);
  lowerRegion(Region->getChild(), State);
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);

  // 2. Classify every recipe by DimSet patterns.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM);

  // 3. Lower: emit IR at the function entry block.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;

  lowerRegion(Plan.getRootRegion(), State);
  return true;
}
