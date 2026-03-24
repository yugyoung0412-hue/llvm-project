//===- TPlanLowering.cpp - Lower TPlan to LLVM IR -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-lower"

//===----------------------------------------------------------------------===//
// execute() implementations per recipe kind
//===----------------------------------------------------------------------===//

void TPCanonicalIVRecipe::execute(TPTransformState &) const {
  // Canonical IV is handled by the loop structure — no direct IR emission.
}

void TPCanonicalIVIncrRecipe::execute(TPTransformState &) const {
  // No direct emission needed — canonical IV increment is loop structure.
}

void TPCanonicalIVExitCmpRecipe::execute(TPTransformState &) const {
  // No direct emission needed — exit compare is loop structure.
}

void TPWidenInductionRecipe::execute(TPTransformState &State) const {
  // IV phi already exists in IR; register it so consumers can look it up.
  if (DefVal)
    State.setValue(DefVal.get(), IVPhi);
}

void TPReductionPHIRecipe::execute(TPTransformState &State) const {
  if (DefVal)
    State.setValue(DefVal.get(), RedPhi);
}

void TPWidenCastRecipe::execute(TPTransformState &State) const {
  auto *SrcDV = dyn_cast<TPDefVal>(getOperand(0));
  Value *Src = SrcDV ? State.getValue(SrcDV) : nullptr;
  if (!Src || !DefVal) return;
  Value *Result = State.Builder.Insert(CastInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenGEPRecipe::execute(TPTransformState &State) const {
  if (!DefVal) return;
  Value *Result = State.Builder.Insert(GEPInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenLoadRecipe::execute(TPTransformState &State) const {
  if (!DefVal) return;
  Value *Result = State.Builder.Insert(LoadInst->clone());
  State.setValue(DefVal.get(), Result);
}

void TPWidenStoreRecipe::execute(TPTransformState &State) const {
  State.Builder.Insert(StoreInst->clone());
}

//===----------------------------------------------------------------------===//
// Contraction emission helper
//===----------------------------------------------------------------------===//

static Value *emitContraction(const TPRecipeBase *FusedMul,
                               const TPRecipeBase *ReductionUpdate,
                               TPTransformState &State) {
  if (!FusedMul || FusedMul->operands().size() < 2)
    return nullptr;

  auto *LHSDefVal = dyn_cast<TPDefVal>(FusedMul->getOperand(0));
  auto *RHSDefVal = dyn_cast<TPDefVal>(FusedMul->getOperand(1));
  if (!LHSDefVal || !RHSDefVal)
    return nullptr;

  // Require exactly 2D operands.
  if (LHSDefVal->DimSet.count() != 2 || RHSDefVal->DimSet.count() != 2) {
    State.Builder.GetInsertBlock()->getContext().diagnose(
        DiagnosticInfoUnsupported(
            *State.Builder.GetInsertBlock()->getParent(),
            "TPlanLowering: Contraction requires 2D operands"));
    return nullptr;
  }

  Value *LHS = State.getValue(LHSDefVal);
  Value *RHS = State.getValue(RHSDefVal);
  if (!LHS || !RHS) return nullptr;

  SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDefVal, State.Plan);
  SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDefVal, State.Plan);

  int ContractDim = State.getContractDim(ReductionUpdate);

  // Find position of ContractDim in each operand's sorted DimSet.
  auto findPos = [](const SmallBitVector &DS, int Dim) -> unsigned {
    unsigned Pos = 0;
    for (int D = DS.find_first(); D >= 0; D = DS.find_next(D), ++Pos)
      if (D == Dim) return Pos;
    return 0;
  };
  unsigned LHSPos = findPos(LHSDefVal->DimSet, ContractDim);
  unsigned RHSPos = findPos(RHSDefVal->DimSet, ContractDim);
  unsigned M = LHSShape[1 - LHSPos];
  unsigned K = LHSShape[LHSPos];
  unsigned N = RHSShape[1 - RHSPos];

  // Ensure LHS/RHS are flat FixedVectorType.
  Type *ElemTy = LHS->getType()->getScalarType();
  auto ensureFlat = [&](Value *V, unsigned Elems) -> Value * {
    Type *FlatTy = FixedVectorType::get(ElemTy, Elems);
    if (V->getType() != FlatTy)
      return State.Builder.CreateBitCast(V, FlatTy);
    return V;
  };
  LHS = ensureFlat(LHS, M * K);
  RHS = ensureFlat(RHS, K * N);

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
// TPWidenRecipe::execute — dispatches on TensorOpKind
//===----------------------------------------------------------------------===//

void TPWidenRecipe::execute(TPTransformState &State) const {
  TensorOpKind Kind = State.getKind(this);

  // Contraction fmul: deferred to the reduction update recipe.
  if (Kind == TensorOpKind::Contraction && !isReductionUpdateRecipe()) {
    return;
  }

  // Contraction fadd (reduction update): emit matmul.
  if (Kind == TensorOpKind::Contraction && isReductionUpdateRecipe()) {
    if (!DefVal) return;
    TPRecipeBase *FusedMul = State.getFusedMulRecipe(this);
    Value *Result = emitContraction(FusedMul, this, State);
    if (Result)
      State.setValue(DefVal.get(), Result);
    return;
  }

  switch (Kind) {
  case TensorOpKind::PlainReduction:
  case TensorOpKind::ElementWise:
  case TensorOpKind::Scalar: {
    if (!DefVal) return;
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  case TensorOpKind::BroadcastBinary: {
    if (!DefVal) return;
    LLVM_DEBUG(dbgs() << "TPlanLowering: BroadcastBinary not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  case TensorOpKind::OuterProduct: {
    if (!DefVal) return;
    LLVM_DEBUG(dbgs() << "TPlanLowering: OuterProduct not yet implemented, "
                         "falling back to scalar clone\n");
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
    return;
  }

  default:
    if (!DefVal) return;
    Value *Result = State.Builder.Insert(Inst->clone());
    State.setValue(DefVal.get(), Result);
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
  // 1. Propagate DimSets.
  TPlanWidener_widen(Plan);

  // 2. Classify recipes.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM);

  // 3. Lower: emit IR at the entry block.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;

  lowerRegion(Plan.getRootRegion(), State);
  return true;
}
