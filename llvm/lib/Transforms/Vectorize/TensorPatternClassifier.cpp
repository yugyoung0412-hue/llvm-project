//===- TensorPatternClassifier.cpp - Tensor pattern classification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

// GEMM detection: depth >= 3, exactly 3 distinct base pointers,
// exactly 2 reads and 1 write.
static bool isGEMM(const LoopNestInfo &Info) {
  if (Info.Depth < 3)
    return false;

  SmallPtrSet<Value *, 4> Bases;
  unsigned Reads = 0, Writes = 0;
  for (const auto &MA : Info.Accesses) {
    Bases.insert(MA.BasePtr);
    if (MA.Kind == AccessKind::Read)
      ++Reads;
    else if (MA.Kind == AccessKind::Write)
      ++Writes;
  }
  return Bases.size() == 3 && Reads == 2 && Writes == 1;
}

PatternHint llvm::classifyPattern(const LoopNestInfo &Info) {
  PatternHint Hint;

  if (!Info.IsAffine || !Info.IsPerfectNest)
    return Hint; // Generic

  if (isGEMM(Info)) {
    Hint.Kind = PatternKind::GEMM;
    return Hint;
  }

  // Elementwise: depth == 1, single read + single write (or just write)
  if (Info.Depth == 1 && !Info.Accesses.empty()) {
    bool AllSameIVCount = true;
    for (const auto &MA : Info.Accesses)
      if (MA.IndexExprs.size() != Info.Depth)
        AllSameIVCount = false;
    if (AllSameIVCount) {
      Hint.Kind = PatternKind::Elementwise;
      return Hint;
    }
  }

  return Hint; // Generic
}

//===----------------------------------------------------------------------===//
// classifyPattern(const TPlan &) — TPlan-based classification
//===----------------------------------------------------------------------===//

PatternHint llvm::classifyPattern(const TPlan &Plan) {
  PatternHint Hint;

  const TPBasicBlock *Header = Plan.getVectorBody()->getHeader();
  if (!Header)
    return Hint;

  unsigned PHICount = 0;
  unsigned LoadCount = 0;
  unsigned StoreCount = 0;
  bool HasMatMul = false;
  bool HasConv = false;
  bool HasReductionPHI = false;

  for (const TPRecipeBase &R : *Header) {
    switch (R.getKind()) {
    case RecipeKind::HeaderPHI:
    case RecipeKind::ScalarHeaderPHI:
      ++PHICount;
      break;
    case RecipeKind::ReductionPHI:
      ++PHICount;
      HasReductionPHI = true;
      break;
    case RecipeKind::WidenLoad:
    case RecipeKind::TensorLoad:
      ++LoadCount;
      break;
    case RecipeKind::WidenStore:
    case RecipeKind::TensorStore:
      ++StoreCount;
      break;
    case RecipeKind::MatMul:
      HasMatMul = true;
      break;
    case RecipeKind::Conv:
      HasConv = true;
      break;
    default:
      break;
    }
  }

  if (HasMatMul || (PHICount >= 3 && LoadCount >= 2 && StoreCount >= 1)) {
    Hint.Kind = PatternKind::GEMM;
  } else if (HasConv || PHICount >= 5) {
    Hint.Kind = PatternKind::Conv2D;
  } else if (HasReductionPHI) {
    Hint.Kind = PatternKind::Reduction;
  } else if (PHICount == 1 && !HasReductionPHI) {
    Hint.Kind = PatternKind::Elementwise;
  } else {
    Hint.Kind = PatternKind::Generic;
  }

  return Hint;
}
