//===- TPRecipeMatcher.cpp - Pattern matching for TPlan recipes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the DimSet of \p V, or an empty bitset for live-ins/synthetics.
static const SmallBitVector &getDimSet(const TPValue *V) {
  static const SmallBitVector Empty;
  if (const auto *DV = dyn_cast<TPDefVal>(V))
    return DV->DimSet;
  return Empty;
}

/// Walk past WidenCast and scalar unary (single-operand Widen) recipes.
/// Returns the first recipe not in that intermediate set, or nullptr.
/// Null-safe: returns nullptr when given nullptr.
static TPRecipeBase *skipIntermediateRecipes(TPValue *V) {
  while (V) {
    auto *DV = dyn_cast<TPDefVal>(V);
    if (!DV)
      return nullptr;
    TPRecipeBase *R = DV->getDefiningRecipe();
    if (!R)
      return nullptr;
    if (R->getKind() == TPRecipeBase::RecipeKind::WidenCast) {
      V = R->getOperand(0); // skip cast, follow source
      continue;
    }
    if (R->getKind() == TPRecipeBase::RecipeKind::Widen &&
        R->operands().size() == 1) {
      auto *WR = cast<TPWidenRecipe>(R);
      if (isa<UnaryOperator>(WR->getInstruction())) {
        V = R->getOperand(0);
        continue;
      }
    }
    return R;
  }
  return nullptr;
}

/// True if \p R is an fmul-like recipe.
static bool isMulLike(const TPRecipeBase *R) {
  if (!R || R->getKind() != TPRecipeBase::RecipeKind::Widen)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  return WR->getInstruction()->getOpcode() == Instruction::FMul;
}

/// True if \p R is a reduction update recipe:
/// a Widen recipe (fadd/fmul) whose one operand is defined by a ReductionPHI.
static bool isReductionUpdate(const TPRecipeBase *R) {
  if (!R || R->getKind() != TPRecipeBase::RecipeKind::Widen)
    return false;
  auto *WR = cast<TPWidenRecipe>(R);
  if (!isa<BinaryOperator>(WR->getInstruction()))
    return false;
  for (TPValue *Op : R->operands()) {
    auto *DV = dyn_cast<TPDefVal>(Op);
    if (DV && dyn_cast<TPReductionPHIRecipe>(DV->getDefiningRecipe()))
      return true;
  }
  return false;
}

/// Returns the non-PHI operand of a reduction update recipe.
static TPValue *getReductionInput(const TPRecipeBase *R) {
  for (TPValue *Op : R->operands()) {
    auto *DV = dyn_cast<TPDefVal>(Op);
    if (!DV || !dyn_cast<TPReductionPHIRecipe>(DV->getDefiningRecipe()))
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
    unsigned N = std::max({D0.size(), D1.size(),
                           Plan.getReductionDims().size()});

    SmallBitVector SharedFinal = D0;
    SharedFinal.resize(N);
    SmallBitVector D1r = D1;
    D1r.resize(N);
    SharedFinal &= D1r;

    SmallBitVector RD = Plan.getReductionDims();
    RD.resize(N);
    SharedFinal &= RD;

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

  unsigned N = std::max(D0.size(), D1.size());
  SmallBitVector A = D0, B = D1;
  A.resize(N); B.resize(N);

  if (A == B)
    return TensorOpKind::ElementWise;

  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BroadcastBinary; // A ⊆ B
  if (Intersection == B) return TensorOpKind::BroadcastBinary; // B ⊆ A

  if (Intersection.none())
    return TensorOpKind::OuterProduct;

  // Partial overlap: invalid TPlan program.
  llvm_unreachable("TPRecipeMatcher: binary op with partially-overlapping "
                   "DimSets is not a valid TPlan program");
}

/// Walk all recipes in \p Region and its children.
static void matchRegion(const TPLoopRegion *Region, const TPlan &Plan,
                         RecipeClassMap &Out) {
  if (!Region)
    return;
  for (const TPRecipeBase &R : Region->getRecipes()) {
    RecipeClassification C;
    if (isReductionUpdate(&R)) {
      C = classifyReduction(R, Plan);
    } else if (R.getKind() == TPRecipeBase::RecipeKind::Widen &&
               isa<BinaryOperator>(
                   cast<TPWidenRecipe>(R).getInstruction()) &&
               R.operands().size() == 2) {
      C.Kind = classifyBinaryOp(R);
    }
    // else: load, store, cast, PHI, canonical IV → default Scalar
    Out[&R] = C;
  }
  matchRegion(Region->getChild(), Plan, Out);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

SmallVector<unsigned> llvm::getTPValueShape(const TPDefVal &V,
                                             const TPlan &Plan) {
  SmallVector<unsigned> Shape;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Shape.push_back(Plan.getPFForDim(static_cast<unsigned>(D)));
  return Shape;
}

void llvm::TPRecipePatternMatcher_match(const TPlan &Plan,
                                         RecipeClassMap &Out) {
  matchRegion(Plan.getRootRegion(), Plan, Out);
}
