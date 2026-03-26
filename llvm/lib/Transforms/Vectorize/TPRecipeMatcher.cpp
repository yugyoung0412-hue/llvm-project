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
    return TensorOpKind::ElementWise;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BroadcastBinary; // A ⊆ B
  if (Intersection == B) return TensorOpKind::BroadcastBinary; // B ⊆ A

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

SmallVector<uint64_t> llvm::getTPValueStrides(const TPSingleDefRecipe &V,
                                               const TPlan &Plan) {
  SmallVector<uint64_t> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan));
  return Strides;
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

void llvm::TPRecipePatternMatcher_match(const TPlan &Plan,
                                         RecipeClassMap &Out) {
  SmallVector<const TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);

  for (const TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      RecipeClassification C;
      if (isReductionUpdate(&R)) {
        C = classifyReduction(R, Plan);
      } else if (R.getTPRecipeID() == TPRecipeBase::TPWidenSC &&
                 isa<BinaryOperator>(
                     cast<TPWidenRecipe>(R).getInstruction()) &&
                 R.operands().size() == 2) {
        C.Kind = classifyBinaryOp(R);
      }
      // else: load, store, cast, PHI, canonical IV → default Scalar
      Out[&R] = C;
    }
  }

  // Second pass: mark each FusedMulRecipe of a Contraction as Contraction too.
  // This ensures the fmul's execute() is a no-op (deferred to its consumer).
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
