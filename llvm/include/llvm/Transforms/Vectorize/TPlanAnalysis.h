//===- TPlanAnalysis.h - Pointer decomposition for TPlan codegen ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Declares decomposePtrForDims(): splits a GEP/bitcast/addrspacecast/PHI
/// pointer chain into a loop-invariant base pointer and per-dimension affine
/// byte strides.  Non-affine GEP steps (e.g. srem-based batch broadcasting)
/// stop the walk; the GEP result at that point becomes the base pointer.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

namespace llvm {

class Loop;
class SCEV;
class ScalarEvolution;
class Value;

/// Result of decomposePtrForDims().
struct PtrDecomposition {
  /// Innermost loop-invariant base pointer after absorbing all non-affine GEPs.
  /// nullptr if the chain could not be walked at all.
  Value *Base = nullptr;

  /// dim → byte stride SCEV for each affine dimension.
  /// Only dims whose GEP index produced a valid SCEVAddRecExpr are present.
  DenseMap<unsigned, const SCEV *> Strides;

  /// Dims for which the GEP index was non-affine (SCEVCouldNotCompute).
  /// The outer loops for these dims iterate scalar; their effect is already
  /// baked into Base.
  SmallBitVector NonAffineDims;

  /// Dims for which stride extraction succeeded.
  SmallBitVector AffineDims;
};

/// Decompose \p Ptr into a base pointer and per-dimension affine byte strides
/// by walking the GEP / bitcast / addrspacecast / loop-invariant-PHI chain.
///
/// \p DimSet        - set of TPlan dimension indices to analyse.
/// \p DimToLoop     - maps each dim index to its corresponding Loop*.
/// \p OutermostGEMMLoop - the outermost loop of the GEMM region; used to
///                    decide whether a PHI incoming value is loop-invariant.
/// \p SE            - ScalarEvolution for SCEV queries.
///
/// Walk rules (applied at each step):
///   bitcast / addrspacecast → transparent skip (pointer identity preserved).
///   PHI node              → follow the incoming value that is loop-invariant
///                           w.r.t. OutermostGEMMLoop; stop if none found.
///   single-index GEP      → call SE.getSCEV(index):
///     SCEVAddRecExpr       → record stride, continue upward.
///     SCEVCouldNotCompute  → mark dim NonAffine, set Base = current ptr, stop.
///     loop-invariant SCEV  → dim not in DimSet at this level; continue upward.
///   multi-index GEP / other → stop.
PtrDecomposition decomposePtrForDims(
    Value *Ptr,
    const SmallBitVector &DimSet,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    Loop *OutermostGEMMLoop,
    ScalarEvolution &SE);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H
