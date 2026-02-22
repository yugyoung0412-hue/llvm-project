//===- LoopNestAnalyzer.h - Loop nest analysis for LoopTensorize ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPNESTANALYZER_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPNESTANALYZER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include <optional>

namespace llvm {
class DependenceInfo;
class Loop;
class LoopInfo;
class SCEV;
class ScalarEvolution;
class Type;
class Value;

struct InductionDesc {
  PHINode  *IndVar    = nullptr;
  const SCEV *TripCount = nullptr;
  const SCEV *Step      = nullptr;
};

enum class AccessKind { Read, Write, ReadWrite };

struct MemAccess {
  Value               *BasePtr  = nullptr;
  SmallVector<const SCEV *> IndexExprs; // one per loop dimension
  AccessKind           Kind;
  Type                *ElemType = nullptr;
};

struct LoopNestInfo {
  SmallVector<Loop *>        Loops;         // outermost -> innermost
  SmallVector<InductionDesc> IVs;
  SmallVector<MemAccess>     Accesses;
  bool                       IsPerfectNest = false;
  bool                       IsAffine      = false;
  unsigned                   Depth         = 0;
};

/// Collects outermost loop nests from a function's LoopInfo.
SmallVector<SmallVector<Loop *>> collectLoopNests(LoopInfo &LI);

/// Analyzes a single loop nest and produces LoopNestInfo.
/// Returns std::nullopt if the nest is not analyzable.
std::optional<LoopNestInfo> analyzeLoopNest(ArrayRef<Loop *> Nest,
                                             ScalarEvolution &SE,
                                             DependenceInfo &DI);

} // namespace llvm
#endif
