//===- TPRecipeMatcher.cpp - Pattern matching for TPlan recipes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

#define DEBUG_TYPE "tplan-matcher"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the DimSet of \p V, or an empty bitset for live-ins/synthetics.
static const SmallBitVector &getDimSet(const TPValue *V) {
  static const SmallBitVector Empty;
  if (const auto *SDR = dyn_cast<TPSingleDefRecipe>(V))
    return SDR->DimSet;
  return Empty;
}

/// Walk past WidenCast and scalar unary (single-operand Widen) recipes.
/// Returns the first recipe that is NOT an intermediate, or nullptr if
/// the chain bottoms out at a live-in / synthetic value.
/// Null-safe: returns nullptr when given nullptr.
static TPRecipeBase *skipIntermediateRecipes(TPValue *V) {
  while (V) {
    auto *SDR = dyn_cast<TPSingleDefRecipe>(V);
    if (!SDR)
      return nullptr;
    // SDR IS the recipe — no getDefiningRecipe() needed
    if (SDR->getTPRecipeID() == TPRecipeBase::TPWidenCastSC) {
      V = SDR->getOperand(0); // skip cast, follow source
      continue;
    }
    if (SDR->getTPRecipeID() == TPRecipeBase::TPWidenSC &&
        SDR->operands().size() == 1) {
      // Single-operand Widen = unary op; skip it.
      if (isa<UnaryOperator>(cast<TPWidenRecipe>(SDR)->getInstruction())) {
        V = SDR->getOperand(0);
        continue;
      }
    }
    return SDR; // Not an intermediate — stop here.
  }
  return nullptr;
}

/// True if \p R is an fmul-like recipe.
static bool isMulLike(const TPRecipeBase *R) {
  if (!R || R->getTPRecipeID() != TPRecipeBase::TPWidenSC)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  return WR->getInstruction()->getOpcode() == Instruction::FMul;
}

/// True if \p R is a reduction update recipe:
/// a Widen recipe (fadd/fsub etc.) whose one operand is defined by a
/// TPReductionPHIRecipe.
static bool isReductionUpdate(const TPRecipeBase *R) {
  if (!R || R->getTPRecipeID() != TPRecipeBase::TPWidenSC)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  if (!isa<BinaryOperator>(WR->getInstruction()))
    return false;
  for (TPValue *Op : R->operands()) {
    auto *RV = dyn_cast<TPRecipeValue>(Op);
    if (!RV) continue;
    if (isa<TPReductionPHIRecipe>(RV->getDefiningRecipe()))
      return true;
  }
  return false;
}

/// Returns the non-PHI operand of a reduction update recipe.
/// Precondition: isReductionUpdate(R) == true.
static TPValue *getReductionInput(const TPRecipeBase *R) {
  for (TPValue *Op : R->operands()) {
    auto *RV = dyn_cast<TPRecipeValue>(Op);
    if (!RV || !isa<TPReductionPHIRecipe>(RV->getDefiningRecipe()))
      return Op;
  }
  return nullptr;
}

/// Classify a single reduction-update recipe.
static RecipeClassification classifyReduction(const TPRecipeBase &R,
                                               const TPlan &Plan) {
  TPValue *Input = getReductionInput(&R);
  TPRecipeBase *Producer = skipIntermediateRecipes(Input);

  if (Producer && isMulLike(Producer)) {
    const SmallBitVector &D0 = getDimSet(Producer->getOperand(0));
    const SmallBitVector &D1 = getDimSet(Producer->getOperand(1));
    // Resize to common size for bitwise ops.
    unsigned N = std::max({D0.size(), D1.size(),
                           Plan.getReductionDims().size()});
    SmallBitVector SharedFinal = D0;
    SharedFinal.resize(N);
    SmallBitVector D1r = D1;
    D1r.resize(N);
    SharedFinal &= D1r; // Shared = D0 & D1

    SmallBitVector RD = Plan.getReductionDims();
    RD.resize(N);
    SharedFinal &= RD; // Shared & ReductionDims

    if (SharedFinal.any()) {
      int ContractDim = SharedFinal.find_first();
      return {TensorOpKind::Contraction, ContractDim,
              const_cast<TPRecipeBase *>(Producer)};
    }
  }
  return {TensorOpKind::PlainReduction, -1, nullptr};
}

/// Classify a binary op recipe (non-reduction).
static TensorOpKind classifyBinaryOp(const TPRecipeBase &R) {
  const SmallBitVector &D0 = getDimSet(R.getOperand(0));
  const SmallBitVector &D1 = getDimSet(R.getOperand(1));

  if (D0.none() && D1.none())
    return TensorOpKind::Scalar;

  // Resize to equal length for comparison.
  unsigned N = std::max(D0.size(), D1.size());
  SmallBitVector A = D0, B = D1;
  A.resize(N); B.resize(N);

  if (A == B)
    return TensorOpKind::BinaryOp;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BinaryOp; // A ⊆ B (broadcast)
  if (Intersection == B) return TensorOpKind::BinaryOp; // B ⊆ A (broadcast)

  // Disjoint check.
  if (Intersection.none())
    return TensorOpKind::OuterProduct;

  // Partial overlap: this binary op may be the fused mul of a contraction.
  // Return Scalar conservatively; the second pass in TPRecipePatternMatcher_match
  // will correct fused muls to Contraction.
  LLVM_DEBUG(llvm::dbgs()
             << "TPRecipeMatcher: partial-overlap binary op classified as "
                "Scalar (may be corrected by second pass)\n");
  return TensorOpKind::Scalar;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

SmallVector<unsigned> llvm::getTPValueShape(const TPSingleDefRecipe &V,
                                             const TPlan &Plan) {
  SmallVector<unsigned> Shape;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Shape.push_back(Plan.getPFForDim(static_cast<unsigned>(D)));
  return Shape;
}

SmallVector<const SCEV *> llvm::getTPValueStrides(const TPSingleDefRecipe &V,
                                                   const TPlan &Plan,
                                                   ScalarEvolution &SE) {
  SmallVector<const SCEV *> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan, SE));
  return Strides;
}

/// Build a map from TPlan dimension index → Loop* by scanning IV recipes
/// in the plan's block structure.
static DenseMap<unsigned, Loop *>
buildDimToLoop(TPlan &Plan, LoopInfo &LI) {
  DenseMap<unsigned, Loop *> DimToLoop;
  if (!Plan.getEntry())
    return DimToLoop;
  SmallVector<TPBlockBase *, 8> Worklist;
  SmallPtrSet<TPBlockBase *, 8> Seen;
  Worklist.push_back(const_cast<TPBlockBase *>(Plan.getEntry()));
  while (!Worklist.empty()) {
    TPBlockBase *Blk = Worklist.pop_back_val();
    if (!Seen.insert(Blk).second)
      continue;
    if (auto *BB = dyn_cast<TPBasicBlock>(Blk)) {
      for (TPRecipeBase &R : *BB) {
        if (auto *IV = dyn_cast<TPWidenInductionRecipe>(&R)) {
          auto *Phi = IV->getIVPhi();
          if (Loop *L = LI.getLoopFor(Phi->getParent()))
            DimToLoop[IV->getDimIndex()] = L;
        }
      }
    }
    if (auto *Reg = dyn_cast<TPRegionBlock>(Blk))
      if (Reg->getEntry())
        Worklist.push_back(Reg->getEntry());
    for (TPBlockBase *Succ : Blk->getSuccessors())
      Worklist.push_back(Succ);
  }
  return DimToLoop;
}

/// Recursively collect per-loop step values from a SCEV expression into
/// \p LoopStep.  Handles two IR patterns produced by TPlan canonicalization:
///
///  1. Nested AddRec chain  — {{base,+,s0}_L0,+,s1}_L1,+,s2}_L2
///     Generated when a single GEP carries a multi-loop flat index.
///     The original while-loop already handles this case, but with canonical
///     IVs (start == 0) it exits after the first iteration because
///     AR->getStart() returns the scalar zero.  We keep the while-loop to
///     handle any remaining nesting below the outermost AddRec.
///
///  2. SCEVAdd of per-loop AddRecs — Add({0,+,s0}_L0, {0,+,s1}_L1, ...)
///     Generated when the index is a sum of separate canonical-IV products
///     (e.g. i*stride_i + k*stride_k).  Each operand is an independent
///     AddRec, so we recurse into every SCEVAdd operand.
///
/// Entries already present in \p LoopStep are left unchanged (try_emplace),
/// so the innermost call wins when the same loop appears at multiple levels.
static void extractAddRecSteps(const SCEV *S, ScalarEvolution &SE,
                                DenseMap<const Loop *, const SCEV *> &LoopStep) {
  if (!S || isa<SCEVCouldNotCompute>(S))
    return;
  // Case 2: flat sum — recurse into each operand independently.
  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (const SCEV *Op : Add->operands())
      extractAddRecSteps(Op, SE, LoopStep);
    return;
  }
  // Case 1: nested AddRec chain — walk until start is no longer an AddRec.
  while (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    LoopStep.try_emplace(AR->getLoop(), AR->getStepRecurrence(SE));
    S = AR->getStart();
  }
}

/// Extract per-dimension element-count strides from \p GEPIdx (the flat index
/// expression of a single-index GEP) and accumulate results into \p MemStrides.
///
/// Delegates SCEV decomposition to extractAddRecSteps so that both nested
/// AddRec chains and SCEVAdd-of-AddRecs are handled correctly.
static void populateSCEVStridesFromIndex(
    DenseMap<unsigned, const SCEV *> &MemStrides,
    const SmallBitVector &DimSet,
    Value *GEPIdx,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    ScalarEvolution &SE) {
  const SCEV *IdxSCEV = SE.getSCEV(GEPIdx);
  if (isa<SCEVCouldNotCompute>(IdxSCEV))
    return;
  DenseMap<const Loop *, const SCEV *> LoopStep;
  extractAddRecSteps(IdxSCEV, SE, LoopStep);
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    auto DIt = DimToLoop.find(static_cast<unsigned>(D));
    if (DIt == DimToLoop.end())
      continue;
    auto SIt = LoopStep.find(DIt->second);
    if (SIt != LoopStep.end())
      MemStrides.try_emplace(static_cast<unsigned>(D), SIt->second);
  }
}

/// Walk the GEP pointer chain starting from \p Ptr and call
/// populateSCEVStridesFromIndex on every single-index GEP encountered.
///
/// TPlan's canonical-IV lowering produces a chain of GEPs where each link
/// adds exactly one loop dimension's byte offset:
///
///   store_ptr = GEP(GEP(GEP(base, batch*nb3), i*nb2), k*nb0)
///                          ^outermost              ^innermost
///
/// A single call to populateSCEVStridesFromIndex on the innermost GEP only
/// captures the innermost stride.  Walking the full chain collects all
/// affine dimensions; non-affine steps (e.g. srem-based batch broadcasting)
/// are silently skipped because their SCEV is SCEVCouldNotCompute.
static void walkGEPChainForStrides(
    DenseMap<unsigned, const SCEV *> &MemStrides,
    const SmallBitVector &DimSet,
    Value *Ptr,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    ScalarEvolution &SE) {
  // Bound the walk to avoid pathological IR (e.g. pointer cycles).
  unsigned MaxDepth = 32;
  while (Ptr && MaxDepth-- > 0) {
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    if (!GEP)
      break;
    if (GEP->getNumIndices() == 1)
      populateSCEVStridesFromIndex(MemStrides, DimSet,
                                   GEP->getOperand(1), DimToLoop, SE);
    Ptr = GEP->getPointerOperand();
  }
}

/// Populate MemStrides on a load recipe by walking the full GEP chain of
/// its load instruction's pointer operand.
static void populateSCEVStrides(TPWidenLoadRecipe &LR,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  auto *Load = cast<LoadInst>(LR.getInstruction());
  walkGEPChainForStrides(LR.MemStrides, LR.DimSet,
                          Load->getPointerOperand(), DimToLoop, SE);
}

/// Populate DimSet + MemStrides on a store recipe. DimSet is copied from
/// the stored-value operand's DimSet; strides come from walking the full
/// GEP chain of the store's pointer operand.
static void populateSCEVStrides(TPWidenStoreRecipe &SR,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  if (auto *ValDR = dyn_cast<TPSingleDefRecipe>(SR.getOperand(1)))
    SR.DimSet = ValDR->DimSet;
  if (SR.DimSet.none())
    return;
  auto *SI = cast<StoreInst>(SR.getInstruction());
  walkGEPChainForStrides(SR.MemStrides, SR.DimSet,
                          SI->getPointerOperand(), DimToLoop, SE);
}

/// Collect all TPBasicBlock instances by recursively walking block successors
/// and descending into TPRegionBlock interiors.
static void collectBBs(TPBlockBase *Start,
                        SmallVectorImpl<const TPBasicBlock *> &Out,
                        SmallPtrSetImpl<TPBlockBase *> &Visited) {
  if (!Start || !Visited.insert(Start).second)
    return;
  if (auto *BB = dyn_cast<TPBasicBlock>(Start))
    Out.push_back(BB);
  if (auto *R = dyn_cast<TPRegionBlock>(Start))
    if (R->getEntry())
      collectBBs(R->getEntry(), Out, Visited);
  for (TPBlockBase *Succ : Start->getSuccessors())
    collectBBs(Succ, Out, Visited);
}

void llvm::TPRecipePatternMatcher_match(TPlan &Plan, RecipeClassMap &Out,
                                         ScalarEvolution &SE, LoopInfo &LI) {
  // Build dim→Loop mapping from IV recipes before classifying.
  DenseMap<unsigned, Loop *> DimToLoop = buildDimToLoop(Plan, LI);

  SmallVector<const TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);

  for (const TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      // Populate SCEV strides on load/store recipes before classification.
      if (auto *LR = dyn_cast<TPWidenLoadRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenLoadRecipe *>(LR), DimToLoop, SE);
      else if (auto *SR = dyn_cast<TPWidenStoreRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenStoreRecipe *>(SR), DimToLoop, SE);

      RecipeClassification C;
      if (isReductionUpdate(&R)) {
        C = classifyReduction(R, Plan);
      } else if (R.getTPRecipeID() == TPRecipeBase::TPWidenSC &&
                 R.operands().size() == 2) {
        auto *Inst = cast<TPWidenRecipe>(R).getInstruction();
        if (isa<BinaryOperator>(Inst) || isa<CmpInst>(Inst))
          C.Kind = classifyBinaryOp(R);
      }
      // else: load, store, cast, PHI, canonical IV → default Scalar
      Out[&R] = C;
    }
  }

  // Second pass: mark each FusedMulRecipe of a Contraction as Contraction too.
  SmallVector<std::pair<TPRecipeBase *, int>> FusedMuls;
  for (auto &[R, C] : Out) {
    if (C.Kind == TensorOpKind::Contraction && C.FusedMulRecipe)
      FusedMuls.push_back({C.FusedMulRecipe, C.ContractDim});
  }
  for (auto [MulR, Dim] : FusedMuls) {
    Out[MulR].Kind = TensorOpKind::Contraction;
    Out[MulR].ContractDim = Dim;
  }
}
