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
/// Phase 1 (Seed): each TPWidenInductionRecipe's defined value gets
///          DimSet = {recipe.getDimIndex()}.
/// Phase 2 (BFS): for every TPSingleDefRecipe V with non-empty DimSet, for
///          every TPUser U that is a TPRecipeBase defining a TPSingleDefRecipe
///          DV: DV.DimSet |= V.DimSet
///
/// Reduction accumulator PHIs (TPReductionPHIRecipe) are intentionally
/// seeded with empty DimSet — they carry scalar accumulated values.
///
/// NOTE: Each DV may be enqueued multiple times (at most Depth times total),
/// since DimSet is monotonically growing (|= only adds bits) and bounded by
/// the loop nest depth. This guarantees termination at the fixpoint.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

/// Collect all TPBasicBlock instances in the plan by recursively walking
/// block successors and descending into TPRegionBlock interiors.
static void collectBasicBlocks(TPBlockBase *Start,
                                SmallVectorImpl<TPBasicBlock *> &Out,
                                SmallPtrSetImpl<TPBlockBase *> &Visited) {
  if (!Start || !Visited.insert(Start).second)
    return;
  if (auto *BB = dyn_cast<TPBasicBlock>(Start))
    Out.push_back(BB);
  if (auto *R = dyn_cast<TPRegionBlock>(Start))
    if (R->getEntry())
      collectBasicBlocks(R->getEntry(), Out, Visited);
  for (TPBlockBase *Succ : Start->getSuccessors())
    collectBasicBlocks(Succ, Out, Visited);
}

void llvm::TPlanWidener_widen(TPlan &Plan) {
  SmallVector<TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBasicBlocks(Plan.getEntry(), AllBBs, Visited);

  SmallVector<TPSingleDefRecipe *, 32> Worklist;

  // Phase 1: Seed from TPWidenInductionRecipe.
  for (TPBasicBlock *BB : AllBBs) {
    for (TPRecipeBase &R : *BB) {
      if (auto *WI = dyn_cast<TPWidenInductionRecipe>(&R)) {
        unsigned Dim = WI->getDimIndex();
        WI->DimSet.resize(std::max(WI->DimSet.size(), (size_t)(Dim + 1)));
        WI->DimSet.set(Dim);
        Worklist.push_back(WI);
      }
    }
  }

  // Phase 2: BFS union propagation to fixpoint.
  // A DV may be re-enqueued whenever its DimSet gains new bits.  Termination
  // is guaranteed because DimSet is monotonically non-decreasing and bounded
  // by the loop nest depth (at most Depth bits per DV).
  while (!Worklist.empty()) {
    TPSingleDefRecipe *V = Worklist.pop_back_val();
    for (TPUser *U : V->users()) {
      auto *Recipe = dyn_cast<TPRecipeBase>(U);
      if (!Recipe)
        continue;
      TPSingleDefRecipe *DV = Recipe->getDefinedValue();
      if (!DV)
        continue;
      unsigned NeedSize = V->DimSet.size();
      if (DV->DimSet.size() < NeedSize)
        DV->DimSet.resize(NeedSize);
      SmallBitVector Before = DV->DimSet;
      DV->DimSet |= V->DimSet;
      if (DV->DimSet != Before)
        Worklist.push_back(DV);
    }
  }
}
