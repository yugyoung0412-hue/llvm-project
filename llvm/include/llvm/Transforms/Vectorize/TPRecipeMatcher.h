//===- TPRecipeMatcher.h - Pattern matching for TPlan recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Declares TPRecipePatternMatcher_match(), getTPValueShape(), and
/// getTPValueStrides(). Requires TPlanWidener_widen() to have been called first.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
#define LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"

namespace llvm {
class LoopInfo;
class SCEV;
class ScalarEvolution;
class TPlan;
class TPSingleDefRecipe;

/// Returns the tensor shape of \p V: { Plan.getPFForDim(d) for d in V.DimSet }.
/// Returns {} for scalar (empty DimSet) values.
/// Requires TPlanWidener_widen() to have been called first.
SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                       const TPlan &Plan);

/// Returns the effective memory stride for each dim in V.DimSet (innermost
/// first) as SCEV expressions in element units. Each entry is
/// V.getMemStride(D, Plan, SE): a SCEV from MemStrides if populated by
/// TPRecipePatternMatcher_match(), else a dense-default SCEV constant.
SmallVector<const SCEV *> getTPValueStrides(const TPSingleDefRecipe &V,
                                             const TPlan &Plan,
                                             ScalarEvolution &SE);

/// Classify every recipe in \p Plan into a TensorOpKind, and populate
/// MemStrides on load/store recipes using SCEV GEP-index analysis.
/// Requires TPlanWidener_widen() to have been called first.
/// Results are written into \p Out (existing entries are overwritten).
void TPRecipePatternMatcher_match(TPlan &Plan, RecipeClassMap &Out,
                                   ScalarEvolution &SE, LoopInfo &LI);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
