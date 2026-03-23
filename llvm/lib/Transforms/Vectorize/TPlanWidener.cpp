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

#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

void llvm::TPlanWidener_widen(TPlan &Plan) {
  // TODO: rewire in commit 2 — walk block CFG via constructionOrder().
  (void)Plan;
}
