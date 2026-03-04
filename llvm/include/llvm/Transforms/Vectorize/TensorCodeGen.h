//===- TensorCodeGen.h - Code generation for LoopTensorize ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORCODEGEN_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORCODEGEN_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class DominatorTree;
class Function;
class LoopInfo;
class ScalarEvolution;
class TPlan;
struct LoopNestInfo;
struct PatternHint;
struct SearchState;
struct TensorOpDesc;

/// Apply the best transformation plan to the IR.
/// Returns true if IR was modified.
bool applyPlan(const SearchState &Plan, const PatternHint &Hint,
               ArrayRef<TensorOpDesc> SupportedOps, Function &F,
               LoopInfo &LI, ScalarEvolution &SE, DominatorTree &DT);

/// Apply a TPlan to the IR. Returns true if IR was modified.
/// Currently supports GEMM pattern only; other patterns return false.
bool applyTPlan(const TPlan &Plan, Function &F, LoopInfo &LI,
                ScalarEvolution &SE, DominatorTree &DT);

} // namespace llvm
#endif
