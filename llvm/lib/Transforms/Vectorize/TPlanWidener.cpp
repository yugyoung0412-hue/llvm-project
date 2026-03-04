//===- TPlanWidener.cpp - PF propagation for TPlan ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implements TPlanWidener_widen(): propagates parallel factors (PF) from
/// induction-variable header PHI recipes through the def-use graph using BFS.
///
/// Algorithm:
///   1. Seed the worklist with all TPHeaderPHIRecipe nodes in the VectorBody
///      header; set their PF from the plan's PFMap.
///   2. BFS: for each TPValue with PF > 1, visit all TPUser users that are
///      TPRecipeBase; propagate max(currentPF, existingPF) to each defined
///      value.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

void llvm::TPlanWidener_widen(TPlan &Plan) {
  TPBasicBlock *Header = Plan.getVectorBody()->getHeader();
  if (!Header)
    return;

  // Worklist of TPValues that have been given a PF > 1.
  SmallVector<TPValue *, 16> Worklist;
  SmallPtrSet<TPValue *, 16> Visited;

  // Phase 1: seed from header PHIs.
  for (TPRecipeBase &R : *Header) {
    if (R.getKind() != RecipeKind::HeaderPHI)
      continue;
    auto &PHI = static_cast<TPHeaderPHIRecipe &>(R);
    unsigned PF = Plan.getPF(PHI.getDimIndex());
    if (PF <= 1)
      continue;
    PHI.setParallelFactor(PF);
    if (Visited.insert(&PHI).second)
      Worklist.push_back(&PHI);
  }

  // Phase 2: BFS propagation.
  while (!Worklist.empty()) {
    TPValue *V = Worklist.pop_back_val();
    unsigned PF = V->getParallelFactor();

    for (TPUser *U : V->users()) {
      // All TPUser instances in this IR are TPRecipeBase nodes (there are no
      // pure TPUser subclasses). Use static_cast to avoid RTTI dependency.
      auto *R = static_cast<TPRecipeBase *>(U);
      if (!R)
        continue;

      // Propagate the max PF to all values defined by this recipe.
      for (unsigned I = 0, E = R->getNumDefinedValues(); I < E; ++I) {
        TPValue *DV = R->getDefinedValue(I);
        if (!DV)
          continue;
        unsigned OldPF = DV->getParallelFactor();
        unsigned NewPF = std::max(OldPF, PF);
        if (NewPF != OldPF) {
          DV->setParallelFactor(NewPF);
          if (Visited.insert(DV).second)
            Worklist.push_back(DV);
        }
      }
    }
  }
}
