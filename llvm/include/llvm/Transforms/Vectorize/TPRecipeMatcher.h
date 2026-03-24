//===- TPRecipeMatcher.h - Pattern matching for TPlan recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
#define LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"

namespace llvm {
class TPlan;
class TPDefVal;

/// Returns the tensor shape of \p V: { Plan.getPFForDim(d) for d in V.DimSet }.
/// Returns {} for scalar (empty DimSet) values.
/// Requires TPlanWidener_widen() to have been called first.
SmallVector<unsigned> getTPValueShape(const TPDefVal &V, const TPlan &Plan);

/// Classify every recipe in \p Plan into a TensorOpKind.
/// Requires TPlanWidener_widen() to have been called first.
/// Results are written into \p Out (existing entries are overwritten).
void TPRecipePatternMatcher_match(const TPlan &Plan, RecipeClassMap &Out);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
