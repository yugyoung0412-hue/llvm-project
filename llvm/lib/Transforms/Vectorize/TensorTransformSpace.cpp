//===- TensorTransformSpace.cpp - Search state and transform primitives ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include <algorithm>

using namespace llvm;

SmallVector<Transform>
llvm::getLegalTransforms(const LoopNestInfo &Info,
                         ArrayRef<TensorOpDesc> SupportedOps) {
  SmallVector<Transform> Result;

  // TensorRecognize: offer one entry per supported tensor op
  for (auto [Idx, Op] : llvm::enumerate(SupportedOps)) {
    Transform T;
    T.Kind = TransformKind::TensorRecognize;
    T.TensorOpIdx = static_cast<int>(Idx);
    Result.push_back(T);
  }

  // LoopTile: offer tiling on each dimension with candidate sizes
  if (Info.Depth >= 1 && Info.IsAffine) {
    for (unsigned D = 0; D < Info.Depth; ++D) {
      for (unsigned S : {4u, 8u, 16u, 32u}) {
        Transform T;
        T.Kind = TransformKind::LoopTile;
        T.Dim  = D;
        T.Size = S;
        Result.push_back(T);
      }
    }
  }

  // Vectorize: always available as a terminal option
  {
    Transform T;
    T.Kind = TransformKind::Vectorize;
    Result.push_back(T);
  }

  // SLPVectorize: terminal option
  {
    Transform T;
    T.Kind = TransformKind::SLPVectorize;
    Result.push_back(T);
  }

  return Result;
}

std::optional<SearchState>
llvm::applyTransform(const SearchState &State, const Transform &T) {
  SearchState Next = State;
  Next.Applied.push_back(T);

  switch (T.Kind) {
  case TransformKind::TensorRecognize:
  case TransformKind::Vectorize:
  case TransformKind::SLPVectorize:
    Next.IsTerminal = true;
    break;
  case TransformKind::LoopTile:
    // Symbolic tiling: depth increases by 1 (outer strip + inner tile)
    Next.Current.Depth += 1;
    break;
  case TransformKind::LoopPermute:
  case TransformKind::LoopUnroll:
  case TransformKind::LoopFuse:
    // No structural depth change for these transforms
    break;
  }

  return Next;
}

SearchState llvm::runBeamSearch(const SearchState &Initial,
                                 ArrayRef<TensorOpDesc> SupportedOps,
                                 const TensorCostModelParams &Params,
                                 unsigned BeamWidth) {
  // Maximum search depth to prevent unbounded tiling.
  const unsigned MaxSearchDepth = 4;

  SmallVector<SearchState, 0> Beam = {Initial};
  SearchState Best = Initial;
  Best.Cost = std::numeric_limits<float>::infinity();

  for (unsigned Depth = 0; Depth < MaxSearchDepth; ++Depth) {
    SmallVector<SearchState, 0> Candidates;

    for (auto &State : Beam) {
      if (State.IsTerminal) {
        float Score = scoreCost(State, Params);
        State.Cost = Score;
        if (Score < Best.Cost)
          Best = State;
        continue;
      }

      SmallVector<Transform> Transforms =
          getLegalTransforms(State.Current, SupportedOps);
      for (const auto &T : Transforms) {
        if (auto Next = applyTransform(State, T)) {
          Next->Cost = scoreCost(*Next, Params);
          Candidates.push_back(*Next);
        }
      }
    }

    if (Candidates.empty())
      break;

    // Sort by cost ascending (lower is better) and keep top BeamWidth.
    std::sort(Candidates.begin(), Candidates.end(),
              [](const SearchState &A, const SearchState &B) {
                return A.Cost < B.Cost;
              });
    if (Candidates.size() > BeamWidth)
      Candidates.resize(BeamWidth);
    Beam = std::move(Candidates);
  }

  // Score any remaining non-terminal states as terminal fallbacks.
  for (auto &State : Beam) {
    if (!State.IsTerminal) {
      float Score = scoreCost(State, Params);
      State.Cost = Score;
      if (Score < Best.Cost)
        Best = State;
    }
  }

  return Best;
}
