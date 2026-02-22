//===- TensorPatternClassifier.cpp - Tensor pattern classification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/ADT/SmallPtrSet.h"

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
