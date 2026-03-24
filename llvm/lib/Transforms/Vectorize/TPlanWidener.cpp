//===- TPlanWidener.cpp - DimSet BFS propagation for TPlan ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implements TPlanWidener_widen(): propagates DimSets from induction-variable
/// recipes through the def-use graph using BFS with union rule.
///
/// Phase 1: Seed — each TPWidenInductionRecipe's defined value gets
///          DimSet = {recipe.getDimIndex()}.
/// Phase 2: BFS — for every TPDefVal V with non-empty DimSet, for every
///          TPUser U that is a TPRecipeBase defining a TPDefVal DV:
///          DV.DimSet |= V.DimSet
///
/// Reduction accumulator PHIs (TPReductionPHIRecipe) are intentionally
/// seeded with empty DimSet — they carry scalar accumulated values.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

/// Walk all recipes in \p Region and its children recursively.
template <typename Fn>
static void walkRecipes(TPLoopRegion *Region, Fn &&F) {
  if (!Region)
    return;
  for (TPRecipeBase &R : Region->getRecipes())
    F(R);
  walkRecipes(Region->getChild(), F);
}

void llvm::TPlanWidener_widen(TPlan &Plan) {
  SmallVector<TPDefVal *, 32> Worklist;
  SmallPtrSet<TPDefVal *, 32> Visited;

  // Phase 1: Seed from TPWidenInductionRecipe.
  walkRecipes(Plan.getRootRegion(), [&](TPRecipeBase &R) {
    if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
      TPDefVal *DV = WI->getDefinedValue();
      if (!DV)
        return;
      unsigned Dim = WI->getDimIndex();
      DV->DimSet.resize(std::max(DV->DimSet.size(), Dim + 1));
      DV->DimSet.set(Dim);
      if (Visited.insert(DV).second)
        Worklist.push_back(DV);
    }
  });

  // Phase 2: BFS union propagation.
  while (!Worklist.empty()) {
    TPDefVal *V = Worklist.pop_back_val();

    for (TPUser *U : V->users()) {
      auto *Recipe = dyn_cast<TPRecipeBase>(U);
      if (!Recipe)
        continue;

      TPDefVal *DV = Recipe->getDefinedValue(); // null for stores/branches
      if (!DV)
        continue;

      // Resize to accommodate all bit indices.
      unsigned NeedSize = V->DimSet.size();
      if (DV->DimSet.size() < NeedSize)
        DV->DimSet.resize(NeedSize);

      // Union: if new bits added, enqueue.
      SmallBitVector Before = DV->DimSet;
      DV->DimSet |= V->DimSet;
      if (DV->DimSet != Before && Visited.insert(DV).second)
        Worklist.push_back(DV);
    }
  }
}
