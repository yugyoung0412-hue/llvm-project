//===- TensorTransformSpace.h - Search state and transform primitives -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORTRANSFORMSPACE_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORTRANSFORMSPACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include <limits>
#include <optional>

namespace llvm {

// Forward-declare TPlan to avoid pulling in the full TPlan.h here.
class TPlan;

enum class TransformKind {
  TensorRecognize,
  LoopTile,
  LoopPermute,
  LoopUnroll,
  LoopFuse,
  Vectorize,
  SLPVectorize,
};

struct Transform {
  TransformKind Kind;
  unsigned Dim  = 0;  // for LoopTile, LoopPermute
  unsigned Size = 0;  // for LoopTile tile size, LoopUnroll factor
  SmallVector<unsigned> Order; // for LoopPermute
  /// Index into TensorISAInfo.getSupportedTensorOps(); -1 = unset.
  int TensorOpIdx = -1;
};

struct SearchState {
  LoopNestInfo           Current;
  TPlan                  *Plan      = nullptr; // nullable; owned by caller
  SmallVector<Transform> Applied;
  float                  Cost       = std::numeric_limits<float>::infinity();
  bool                   IsTerminal = false;
};

/// Returns all legal next transforms for the current loop nest state.
SmallVector<Transform>
getLegalTransforms(const LoopNestInfo &Info,
                   ArrayRef<TensorOpDesc> SupportedOps = {});

/// Applies a transform to a SearchState, returning the updated state.
/// Returns std::nullopt if the transform is illegal.
std::optional<SearchState> applyTransform(const SearchState &State,
                                           const Transform &T);

// Forward declaration to avoid including TensorCostModel.h in this header.
struct TensorCostModelParams;

/// Run beam search over the transformation space, returning the best terminal
/// SearchState found. BeamWidth controls the number of candidates kept at each
/// depth. Lower Cost in the returned state is better (roofline cycles).
SearchState runBeamSearch(const SearchState &Initial,
                          ArrayRef<TensorOpDesc> SupportedOps,
                          const TensorCostModelParams &Params,
                          unsigned BeamWidth = 8);

} // namespace llvm
#endif
