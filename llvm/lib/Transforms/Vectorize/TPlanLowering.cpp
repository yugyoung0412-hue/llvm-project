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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-lower"

//===----------------------------------------------------------------------===//
// Intrinsic declaration helpers
//===----------------------------------------------------------------------===//

/// Returns (creating if needed) @llvm.tensor.matmul.<type>.
/// Signature: void(ptr C, i64 M, i64 N, i64 ldc,
///                 ptr A, i64 M, i64 K, i64 lda,
///                 ptr B, i64 K, i64 N, i64 ldb)
static FunctionCallee getTensorMatmulFn(Module &M, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  std::string Name = "llvm.tensor.matmul.";
  Name += ElemTy->isFloatTy() ? "f32" : "f64";
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(Ctx),
      {PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty},
      /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

/// Returns (creating if needed) @llvm.tensor.elementwise.<op>.<rank>d.<type>.
/// Signature: void( (ptr, i64×Rank) × 3 tensors,  i64×Rank dims )
static FunctionCallee getTensorElementwiseFn(Module &M, StringRef OpName,
                                              unsigned Rank, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  std::string Name = "llvm.tensor.elementwise.";
  Name += OpName.str();
  Name += "." + std::to_string(Rank) + "d.";
  Name += ElemTy->isFloatTy() ? "f32" : "f64";
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  for (unsigned T = 0; T < 3; ++T) {
    Params.push_back(PtrTy);
    for (unsigned R = 0; R < Rank; ++R)
      Params.push_back(I64Ty);
  }
  for (unsigned R = 0; R < Rank; ++R)
    Params.push_back(I64Ty);
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  return M.getOrInsertFunction(Name, FT);
}

//===----------------------------------------------------------------------===//
// Stage-print helpers (unconditional; not gated by LLVM_DEBUG)
//===----------------------------------------------------------------------===//

static StringRef kindToStr(TensorOpKind K) {
  switch (K) {
  case TensorOpKind::Scalar:          return "Scalar";
  case TensorOpKind::ElementWise:     return "ElementWise";
  case TensorOpKind::BroadcastBinary: return "BroadcastBinary";
  case TensorOpKind::OuterProduct:    return "OuterProduct";
  case TensorOpKind::Contraction:     return "Contraction";
  case TensorOpKind::PlainReduction:  return "PlainReduction";
  }
  return "Unknown";
}

static void collectAllBBs(TPBlockBase *Start,
                            SmallVectorImpl<TPBasicBlock *> &Out,
                            SmallPtrSet<TPBlockBase *, 32> &Visited) {
  if (!Start || !Visited.insert(Start).second)
    return;
  if (auto *BB = dyn_cast<TPBasicBlock>(Start))
    Out.push_back(BB);
  if (auto *R = dyn_cast<TPRegionBlock>(Start))
    if (R->getEntry())
      collectAllBBs(R->getEntry(), Out, Visited);
  for (TPBlockBase *Succ : Start->getSuccessors())
    collectAllBBs(Succ, Out, Visited);
}

static void printClassificationSummary(const TPlan &Plan,
                                        const RecipeClassMap &CM,
                                        raw_ostream &OS) {
  OS << "Recipe classifications (non-Scalar):\n";
  SmallVector<TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectAllBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);
  bool Any = false;
  for (TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      auto It = CM.find(&R);
      if (It == CM.end())
        continue;
      const auto &C = It->second;
      if (C.Kind == TensorOpKind::Scalar)
        continue;
      OS << "  [" << BB->getName() << "] " << kindToStr(C.Kind);
      if (C.ContractDim >= 0)
        OS << " (contractDim=" << C.ContractDim << ")";
      OS << "\n";
      Any = true;
    }
  }
  if (!Any)
    OS << "  (none)\n";
}

//===----------------------------------------------------------------------===//
// Helper: emit @llvm.tensor.matmul for a Contraction reduction
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
  if (LHSDR->DimSet.count() != 2 || RHSDR->DimSet.count() != 2) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction requires 2D operands\n");
    return nullptr;
  }

  // The LHS/RHS recipes are load recipes; we need their pointer operands for
  // the @llvm.tensor.matmul intrinsic (ptr-based API).
  auto *LHSLoad = dyn_cast<TPWidenLoadRecipe>(LHSDR);
  auto *RHSLoad = dyn_cast<TPWidenLoadRecipe>(RHSDR);
  if (!LHSLoad || !RHSLoad)
    return nullptr;

  auto *LHSPtrDR = dyn_cast<TPSingleDefRecipe>(LHSLoad->getOperand(0));
  auto *RHSPtrDR = dyn_cast<TPSingleDefRecipe>(RHSLoad->getOperand(0));
  if (!LHSPtrDR || !RHSPtrDR)
    return nullptr;

  Value *LHSPtr = State.getValue(LHSPtrDR);
  Value *RHSPtr = State.getValue(RHSPtrDR);
  if (!LHSPtr || !RHSPtr)
    return nullptr;

  // Determine element type from the load instruction. Use getScalarType() to
  // handle both scalar (load float) and vector (load <N x float>) loads.
  Type *ElemTy = LHSLoad->getInstruction()->getType()->getScalarType();
  if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy())
    return nullptr;

  SmallVector<unsigned>  LHSShape   = getTPValueShape(*LHSDR, State.Plan);
  SmallVector<unsigned>  RHSShape   = getTPValueShape(*RHSDR, State.Plan);
  SmallVector<const SCEV *> LHSStrides =
      getTPValueStrides(*LHSDR, State.Plan, *State.SE);
  SmallVector<const SCEV *> RHSStrides =
      getTPValueStrides(*RHSDR, State.Plan, *State.SE);

  int ContractDim = State.getContractDim(ReductionUpdate);
  if (ContractDim < 0 ||
      !LHSDR->DimSet.test(static_cast<unsigned>(ContractDim)) ||
      !RHSDR->DimSet.test(static_cast<unsigned>(ContractDim))) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction dim not in operand DimSets\n");
    return nullptr;
  }
  // findPos: returns position of Dim in DimSet iteration order (innermost-first).
  // The guards above guarantee Dim is present, so the fallback 0 is never reached.
  auto findPos = [](const SmallBitVector &DS, int Dim) -> unsigned {
    unsigned Pos = 0;
    for (int D = DS.find_first(); D >= 0; D = DS.find_next(D), ++Pos)
      if (D == Dim) return Pos;
    return 0; // unreachable given caller's DimSet membership guard
  };
  unsigned LHSPos = findPos(LHSDR->DimSet, ContractDim);
  unsigned RHSPos = findPos(RHSDR->DimSet, ContractDim);

  uint64_t M = LHSShape[1 - LHSPos];
  uint64_t K = LHSShape[LHSPos];
  uint64_t N = RHSShape[1 - RHSPos];
  // Locate the C accumulator pointer from the store recipe that consumes
  // the reduction result.
  //
  // Primary: walk recipe-level users of the ReductionUpdate defined value.
  // This works when the store is in the same TPBasicBlock.
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
        if (PtrDR)
          CPtr = State.getValue(PtrDR);
        break;
      }
    }
  }
  // Fallback: walk IR-level uses of the reduction result instruction.
  // The store to C may be in a different block (e.g., the outer loop latch)
  // and reference %sum as an IR live-in rather than a recipe TPUser.
  // We trace through any GEP to the base pointer so the result always
  // dominates the current insertion point.
  if (!CPtr) {
    if (auto *WR = dyn_cast<TPWidenRecipe>(ReductionUpdate)) {
      for (User *U : WR->getInstruction()->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          Value *Ptr = SI->getPointerOperand();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
            CPtr = GEP->getPointerOperand(); // base ptr (e.g. %C arg)
          else
            CPtr = Ptr;
          break;
        }
      }
    }
  }
  if (!CPtr) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction cannot find C pointer\n");
    return nullptr;
  }

  Module *Mod = State.Builder.GetInsertBlock()->getModule();
  auto MatmulFn = getTensorMatmulFn(*Mod, ElemTy);
  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
  };
  unsigned LHSLastDim = static_cast<unsigned>(LHSDR->DimSet.find_last());
  unsigned RHSLastDim = static_cast<unsigned>(RHSDR->DimSet.find_last());
  Value *VDA = expandStride(LHSStrides.back(), LHSLastDim);
  Value *VDB = expandStride(RHSStrides[RHSPos], static_cast<unsigned>(ContractDim));
  Value *VDC;
  if (CStoreRecipe && State.SE) {
    int CLastDim = CStoreRecipe->DimSet.find_last();
    const SCEV *LDC_SCEV = CLastDim >= 0
        ? CStoreRecipe->getMemStride(static_cast<unsigned>(CLastDim),
                                      State.Plan, *State.SE)
        : State.SE->getConstant(APInt(64, 1));
    VDC = expandStride(LDC_SCEV, CLastDim >= 0 ? static_cast<unsigned>(CLastDim) : 0);
  } else {
    VDC = I64(State.Plan.getDenseStrideForDim(LHSLastDim + 1));
  }

  return B.CreateCall(MatmulFn,
      {CPtr,   I64(M), I64(N), VDC,
       LHSPtr, I64(M), I64(K), VDA,
       RHSPtr, I64(K), I64(N), VDB});
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

void TPWidenIntOrFpInductionRecipe::execute(TPTransformState &State) const {
  // IV values are loop PHIs already present in IR; register them in ValueMap.
  State.setValue(this, IVPhi);
}

void TPWidenPointerInductionRecipe::execute(TPTransformState &State) const {
  State.setValue(this, IVPhi);
}

void TPReductionPHIRecipe::execute(TPTransformState &State) const {
  State.setValue(this, RedPhi);
}

void TPFirstOrderRecurrencePHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPActiveLaneMaskPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPEVLBasedIVPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPWidenPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPPredInstPHIRecipe::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}
void TPPhi::execute(TPTransformState &) const {
  // TODO: implement when lowering supports this recipe
}

void TPWidenCastRecipe::execute(TPTransformState &State) const {
  auto *SrcDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
  Value *Src = SrcDR ? State.getValue(SrcDR) : nullptr;
  if (!Src) return;
  auto *Clone = CastInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[CastInst] = Result;
  State.setValue(this, Result);
}

void TPWidenGEPRecipe::execute(TPTransformState &State) const {
  auto *Clone = GEPInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[GEPInst] = Result;
  State.setValue(this, Result);
}

void TPWidenLoadRecipe::execute(TPTransformState &State) const {
  auto *Clone = LoadInst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  State.EmittedMap[LoadInst] = Result;
  State.setValue(this, Result);
}

void TPWidenStoreRecipe::execute(TPTransformState &State) const {
  auto *Clone = StoreInst->clone();
  State.remapClone(Clone);
  State.Builder.Insert(Clone);
  State.EmittedMap[StoreInst] = Clone;
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

  case TensorOpKind::ElementWise: {
    auto tryVectorize = [&]() -> bool {
      auto *ADR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      auto *BDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
      if (!ADR || !BDR) return false;

      unsigned Rank = ADR->DimSet.count();
      if (Rank < 1 || Rank > 3) return false; // TODO: support rank > 3

      // Determine op name from the BinaryOperator opcode.
      StringRef OpName;
      if (auto *BO = dyn_cast<BinaryOperator>(Inst)) {
        switch (BO->getOpcode()) {
        case Instruction::FAdd: OpName = "fadd"; break;
        case Instruction::FSub: OpName = "fsub"; break;
        case Instruction::FMul: OpName = "fmul"; break;
        default: return false;
        }
      } else {
        return false;
      }

      // Get pointer operands from load recipes.
      auto *ALoad = dyn_cast<TPWidenLoadRecipe>(ADR);
      auto *BLoad = dyn_cast<TPWidenLoadRecipe>(BDR);
      if (!ALoad || !BLoad) return false;
      auto *APtrDR = dyn_cast<TPSingleDefRecipe>(ALoad->getOperand(0));
      auto *BPtrDR = dyn_cast<TPSingleDefRecipe>(BLoad->getOperand(0));
      if (!APtrDR || !BPtrDR) return false;
      Value *APtr = State.getValue(APtrDR);
      Value *BPtr = State.getValue(BPtrDR);
      if (!APtr || !BPtr) return false;

      // Find C pointer from the store recipe that uses this recipe's result.
      // Store operand 0 is the pointer (PtrOp), operand 1 is the value (ValOp).
      Value *CPtr = nullptr;
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          auto *RB = dyn_cast<TPRecipeBase>(U);
          if (!RB) continue;
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
            auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
            if (PtrDR) CPtr = State.getValue(PtrDR);
            break;
          }
        }
      }
      if (!CPtr) return false;

      SmallVector<unsigned>  Shape    = getTPValueShape(*ADR, State.Plan);
      SmallVector<const SCEV *> AStrides =
          getTPValueStrides(*ADR, State.Plan, *State.SE);
      SmallVector<const SCEV *> BStrides =
          getTPValueStrides(*BDR, State.Plan, *State.SE);
      // Derive C strides from the store recipe's MemStrides.
      SmallVector<const SCEV *> CStrides;
      TPWidenStoreRecipe *CStoreRecipe = nullptr;
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          auto *RB = dyn_cast<TPRecipeBase>(U);
          if (!RB) continue;
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
            CStoreRecipe = SR;
            for (int D = SR->DimSet.find_first(); D >= 0;
                 D = SR->DimSet.find_next(D))
              CStrides.push_back(
                  SR->getMemStride(static_cast<unsigned>(D), State.Plan, *State.SE));
            break;
          }
        }
      }
      if (CStrides.empty())
        CStrides = AStrides; // fallback: use A's strides

      Type *ElemTy = ALoad->getInstruction()->getType()->getScalarType();
      if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy()) return false;
      Module *Mod  = State.Builder.GetInsertBlock()->getModule();
      auto EltFn   = getTensorElementwiseFn(*Mod, OpName, Rank, ElemTy);
      IRBuilder<> &B = State.Builder;
      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
      auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
        if (State.Expander && State.Expander->isSafeToExpand(S))
          return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                               &*B.GetInsertPoint());
        return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
      };

      SmallVector<Value *> Args;
      auto appendStrideArgs = [&](SmallVector<const SCEV *> &Strides,
                                   const SmallBitVector &DimBV) {
        int D = DimBV.find_first();
        for (const SCEV *S : Strides) {
          Args.push_back(expandStride(S, D >= 0 ? static_cast<unsigned>(D) : 0));
          if (D >= 0) D = DimBV.find_next(D);
        }
      };
      Args.push_back(CPtr);
      appendStrideArgs(CStrides, CStoreRecipe ? CStoreRecipe->DimSet : ADR->DimSet);
      Args.push_back(APtr);
      appendStrideArgs(AStrides, ADR->DimSet);
      Args.push_back(BPtr);
      appendStrideArgs(BStrides, BDR->DimSet);
      for (unsigned D : Shape) Args.push_back(I64(D));
      B.CreateCall(EltFn, Args);
      return true;
    };

    if (tryVectorize()) return;

    // Scalar fallback.
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::Scalar: {
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::BroadcastBinary: {
    LLVM_DEBUG({
      auto *DR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      if (DR) {
        auto Shape   = getTPValueShape(*DR, State.Plan);
        auto Strides = State.SE
            ? getTPValueStrides(*DR, State.Plan, *State.SE)
            : SmallVector<const SCEV *>{};
        dbgs() << "BroadcastBinary shape=[";
        for (unsigned D : Shape) dbgs() << D << " ";
        dbgs() << "] strides=[";
        for (const SCEV *S : Strides) {
          if (S) S->print(dbgs()); else dbgs() << "?";
          dbgs() << " ";
        }
        dbgs() << "] — scalar fallback (TODO: broadcast intrinsic)\n";
      }
    });
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::OuterProduct: {
    auto tryVectorize = [&]() -> bool {
      auto *LHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      auto *RHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
      if (!LHSDR || !RHSDR) return false;

      auto *LHSLoad = dyn_cast<TPWidenLoadRecipe>(LHSDR);
      auto *RHSLoad = dyn_cast<TPWidenLoadRecipe>(RHSDR);
      if (!LHSLoad || !RHSLoad) return false;
      auto *LHSPtrDR = dyn_cast<TPSingleDefRecipe>(LHSLoad->getOperand(0));
      auto *RHSPtrDR = dyn_cast<TPSingleDefRecipe>(RHSLoad->getOperand(0));
      if (!LHSPtrDR || !RHSPtrDR) return false;
      Value *LHSPtr = State.getValue(LHSPtrDR);
      Value *RHSPtr = State.getValue(RHSPtrDR);
      if (!LHSPtr || !RHSPtr) return false;

      SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDR, State.Plan);
      SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDR, State.Plan);
      if (LHSShape.empty() || RHSShape.empty()) return false;

      uint64_t M = 1, N = 1;
      for (unsigned D : LHSShape) M *= D;
      for (unsigned D : RHSShape) N *= D;

      // Store operand 0 is pointer, operand 1 is value.
      // Primary: recipe-level user walk. Fallback: IR-level store users
      // (same pattern as emitContraction, for stores in outer-loop latches).
      Value *CPtr = nullptr;
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          auto *RB = dyn_cast<TPRecipeBase>(U);
          if (!RB) continue;
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
            auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
            if (PtrDR) CPtr = State.getValue(PtrDR);
            break;
          }
        }
      }
      if (!CPtr) {
        for (User *U : LHSLoad->getInstruction()->users()) {
          if (auto *SI = dyn_cast<StoreInst>(U)) {
            Value *Ptr = SI->getPointerOperand();
            if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
              CPtr = GEP->getPointerOperand();
            else
              CPtr = Ptr;
            break;
          }
        }
      }
      if (!CPtr) return false;

      Type *ElemTy = LHSLoad->getInstruction()->getType()->getScalarType();
      if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy()) return false;
      Module *Mod  = State.Builder.GetInsertBlock()->getModule();
      auto MatmulFn = getTensorMatmulFn(*Mod, ElemTy);
      IRBuilder<> &B = State.Builder;
      auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };

      // OuterProduct = matmul(col-vec[M×1], row-vec[1×N]) → C[M×N]
      B.CreateCall(MatmulFn,
          {CPtr,    I64(M), I64(N), I64(N),    // C, M, N, ldc=N (dense)
           LHSPtr,  I64(M), I64(1), I64(1),    // A: M×1 col-vector, lda=1
           RHSPtr,  I64(1), I64(N), I64(N)});  // B: 1×N row-vector, ldb=N
      return true;
    };

    if (tryVectorize()) return;

    // Scalar fallback.
    LLVM_DEBUG(dbgs() << "TPlanLowering: OuterProduct vectorize failed, "
                         "falling back to scalar clone\n");
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }

  case TensorOpKind::PlainReduction: {
    // Reduction update with no fuseable mul-like producer — clone as scalar.
    LLVM_DEBUG({
      auto *DR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
      if (DR) {
        auto Shape   = getTPValueShape(*DR, State.Plan);
        auto Strides = State.SE
            ? getTPValueStrides(*DR, State.Plan, *State.SE)
            : SmallVector<const SCEV *>{};
        dbgs() << "PlainReduction shape=[";
        for (unsigned D : Shape) dbgs() << D << " ";
        dbgs() << "] strides=[";
        for (const SCEV *S : Strides) {
          if (S) S->print(dbgs()); else dbgs() << "?";
          dbgs() << " ";
        }
        dbgs() << "] — scalar fallback\n";
      }
    });
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);
  LLVM_DEBUG({
    dbgs() << "\n=== Stage 2: After Widening (DimSets propagated) ===\n";
    Plan.print(dbgs());
  });
  errs() << "\n=== Stage 2: After Widening (DimSets propagated) ===\n";
  Plan.print(errs());

  // 2. Classify every recipe by DimSet patterns.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM, SE, LI);
  LLVM_DEBUG({
    dbgs() << "\n=== Stage 3: After Pattern Matching (recipe classifications) ===\n";
    Plan.print(dbgs());
    printClassificationSummary(Plan, CM, dbgs());
  });
  errs() << "\n=== Stage 3: After Pattern Matching (recipe classifications) ===\n";
  Plan.print(errs());
  printClassificationSummary(Plan, CM, errs());

  // 3. Lower: walk block CFG in construction order.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;
  SCEVExpander Expander(SE, "tplan.stride");
  State.SE = &SE;
  State.Expander = &Expander;

  if (Plan.getEntry()) {
    // Collect and execute all blocks in top-level construction order.
    // TPRegionBlock::execute() recurses into its interior.
    for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
      B->execute(State);
  }
  return true;
}
