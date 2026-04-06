//===- TPlanTypes.h - Shared types for TPlan lowering ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Defines TensorOpKind, RecipeClassification, and RecipeClassMap.
/// No includes from TPlan.h or TPRecipe.h to avoid circular dependencies.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {
class TPRecipeBase;

/// Classification of a recipe's tensor operation semantics.
enum class TensorOpKind {
  Scalar,           ///< DimSet empty — scalar op, no tensor parallelism
  ElementWise,      ///< Binary op, both operand DimSets equal (kept for legacy)
  BroadcastBinary,  ///< Binary op, one DimSet is strict subset (kept for legacy)
  BinaryOp,         ///< Binary op — unified element-wise + broadcast path
  OuterProduct,     ///< Binary op, operand DimSets are disjoint
  Contraction,      ///< Reduction update of mul-like op sharing a reduction dim
  PlainReduction,   ///< Reduction update with no fuseable mul-like producer
};

struct RecipeClassification {
  TensorOpKind  Kind           = TensorOpKind::Scalar;
  int           ContractDim    = -1;         ///< Loop-dim index; Contraction only
  TPRecipeBase *FusedMulRecipe = nullptr;    ///< Pre-resolved mul recipe; Contraction only
};

/// Maps every recipe in a TPlan to its classification.
/// Produced by TPRecipePatternMatcher_match(), consumed by TPlanLowering_lower().
using RecipeClassMap = DenseMap<const TPRecipeBase *, RecipeClassification>;

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLANTYPES_H
