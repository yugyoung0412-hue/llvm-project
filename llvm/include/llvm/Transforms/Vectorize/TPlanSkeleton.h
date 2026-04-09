//===- TPlanSkeleton.h - Tensorized loop skeleton creation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares createTensorizedLoopSkeleton(), which clones the outermost GEMM
/// loop as a scalar fallback and inserts a runtime profitability guard BEFORE
/// the outermost loop — mirroring VPlan's createVectorizedLoopSkeleton().
///
/// The guard is emitted ONCE before the M (outermost) loop, not inside the
/// K (innermost) loop. This prevents the guard from running M×N times.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLANSKELETON_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLANSKELETON_H

#include "llvm/IR/BasicBlock.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class DominatorTree;
class Loop;
class LoopInfo;
class Value;

/// Result of createTensorizedLoopSkeleton().
///
/// After creation, the CFG looks like:
///
///   [OrigPred] → [GuardBB: TC >=u PF ?]
///                    |  true                   false
///                    ↓                           ↓
///              [TensorPreheader]         [ScalarPreheader]
///               → [tensor loop] →         → [scalar clone] →
///                          [MergeBB (original loop exit)]
///
/// GuardBB         — emits `icmp uge RuntimeTC, PF` + `condbr`.
/// TensorPreheader — the original loop's preheader; emitContraction() fills it.
/// ScalarPreheader — the cloned loop's preheader; left unmodified (scalar).
/// MergeBB         — the original loop's single exit block.
/// Valid           — false if any precondition was not met.
///
/// The VMap (original → scalar-clone mapping) is returned via the out-param
/// passed to createTensorizedLoopSkeleton(); it is NOT stored here because
/// ValueToValueMapTy is neither copyable nor movable.
struct TensorizedLoopSkeleton {
  BasicBlock *GuardBB         = nullptr; ///< Runtime TC >=u PF check.
  BasicBlock *TensorPreheader = nullptr; ///< Original loop's preheader.
  BasicBlock *ScalarPreheader = nullptr; ///< Clone's preheader.
  BasicBlock *MergeBB         = nullptr; ///< Common exit block.
  bool Valid = false;
};

/// Creates a tensorized loop skeleton around \p OutermostLoop.
///
/// Clones \p OutermostLoop as a scalar fallback and inserts a GuardBB that
/// checks `RuntimeTC >=u PF` before the loop. True → tensorized path (original
/// loop, subsequently modified by emitContraction). False → scalar clone.
///
/// The original-to-clone block/instruction mapping is written into \p VMap.
///
/// Preconditions:
///   - OutermostLoop must have a unique preheader with a single predecessor.
///   - OutermostLoop must have a single exit block.
///   - RuntimeTC must dominate the preheader.
///
/// \param LI   LoopInfo, updated in place.
/// \param DT   DominatorTree, updated in place.
/// \param VMap Output: maps original blocks/instructions to their clones.
/// Returns Valid=false on any precondition failure.
TensorizedLoopSkeleton createTensorizedLoopSkeleton(Loop *OutermostLoop,
                                                     Value *RuntimeTC,
                                                     unsigned PF,
                                                     LoopInfo &LI,
                                                     DominatorTree &DT,
                                                     ValueToValueMapTy &VMap);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLANSKELETON_H
