//===- TensorPatternClassifier.cpp - Tensor pattern classification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Vectorize/TPlan.h"

using namespace llvm;

/// Recursively walk a nested SCEVAddRecExpr chain and collect all linear
/// AddRec nodes grouped by their step operand.  The SCEV representation for
/// a multi-loop affine subscript is a chain such as
///   {{{base,+,s0}L0,+,s1}L1,+,s2}L2
/// so we must descend into the start operand to find inner AddRecs.
static void collectTermsByStep(
    const SCEV *S,
    DenseMap<const SCEV *, SmallVector<const SCEVAddRecExpr *, 4>> &Out) {
  if (!S)
    return;
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AR->isAffine()) {
      Out[AR->getOperand(1)].push_back(AR); // operand(1) is the step
    }
    // Recurse into the start operand to find inner AddRecs.
    collectTermsByStep(AR->getStart(), Out);
  }
  // Also handle a flat SCEVAddExpr whose operands may be AddRecs.
  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (const SCEV *Op : Add->operands())
      collectTermsByStep(Op, Out);
  }
}

/// Detect Conv2D sliding-window pattern:
///   depth >= 4, perfect+affine, exactly 3 distinct base pointers,
///   and at least one read's pointer SCEV has two SCEVAddRecExpr nodes
///   sharing the same step value (e.g. oh and kh both step by W*elemSize).
/// The accumulation pattern (output[oh,ow] += ...) produces 3 reads + 1 write
/// because the output is both loaded and stored, so we allow Reads >= 2.
static bool isConv2D(const LoopNestInfo &Info) {
  if (Info.Depth < 4)
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
  // 3 distinct base arrays (input, kernel, output), at least 2 reads (input
  // + output-read-for-accumulate), exactly 1 write.
  if (Bases.size() != 3 || Reads < 2 || Writes != 1)
    return false;

  // For each read access, inspect the pointer SCEV for the sliding-window
  // signature: two or more AddRec nodes sharing the same step value.
  for (const auto &MA : Info.Accesses) {
    if (MA.Kind != AccessKind::Read || MA.IndexExprs.empty())
      continue;
    DenseMap<const SCEV *, SmallVector<const SCEVAddRecExpr *, 4>> StepMap;
    collectTermsByStep(MA.IndexExprs[0], StepMap);
    for (const auto &[Step, ARs] : StepMap)
      if (ARs.size() >= 2)
        return true; // Two IVs share the same stride -> sliding window
  }
  return false;
}

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

  // Check Conv2D first: stricter than GEMM (depth >= 4 + step equality).
  // A 4-deep nest with 3 base pointers would otherwise also match isGEMM.
  if (isConv2D(Info)) {
    Hint.Kind = PatternKind::Conv2D;
    return Hint;
  }

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

PatternHint llvm::classifyPattern(TPlan &Plan) {
  PatternHint Hint;

  // 1. Count induction recipes → depth.
  unsigned Depth = 0;
  for (const auto &R : Plan.recipes())
    if (isa<TPInductionRecipe>(R))
      ++Depth;

  // 2. Count mem recipes and collect distinct base pointers.
  SmallPtrSet<Value *, 4> Bases;
  unsigned Reads = 0, Writes = 0;
  for (const auto &R : Plan.recipes()) {
    if (const auto *MR = dyn_cast<TPMemRecipe>(&R)) {
      if (MR->MA.BasePtr)
        Bases.insert(MR->MA.BasePtr);
      if (MR->IsWrite)
        ++Writes;
      else
        ++Reads;
    }
  }

  // 3. Conv2D: depth>=4, 3 distinct bases, reads>=2, writes==1,
  //    and at least one read MA has the sliding-window SCEV signature.
  PatternKind Kind = PatternKind::Generic;
  if (Depth >= 4 && Bases.size() == 3 && Reads >= 2 && Writes == 1) {
    for (const auto &R : Plan.recipes()) {
      if (const auto *MR = dyn_cast<TPMemRecipe>(&R)) {
        if (!MR->IsWrite && !MR->MA.IndexExprs.empty()) {
          DenseMap<const SCEV *, SmallVector<const SCEVAddRecExpr *, 4>> StepMap;
          collectTermsByStep(MR->MA.IndexExprs[0], StepMap);
          for (const auto &[Step, ARs] : StepMap) {
            if (ARs.size() >= 2) {
              Kind = PatternKind::Conv2D;
              break;
            }
          }
        }
        if (Kind == PatternKind::Conv2D)
          break;
      }
    }
  }

  // 4. GEMM: depth>=3, 3 distinct bases, reads==2, writes==1.
  if (Kind == PatternKind::Generic &&
      Depth >= 3 && Bases.size() == 3 && Reads == 2 && Writes == 1)
    Kind = PatternKind::GEMM;

  // 5. Set Pattern on the TPComputeRecipe.
  for (auto &R : Plan.recipes()) {
    if (auto *CR = dyn_cast<TPComputeRecipe>(&R)) {
      CR->Pattern = Kind;
      break;
    }
  }

  Hint.Kind = Kind;
  return Hint;
}
