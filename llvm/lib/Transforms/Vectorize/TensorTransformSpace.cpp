//===- TensorTransformSpace.cpp - Search state and transform primitives ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/TPlan.h"
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

TPlan llvm::searchTPlan(TPlan Initial, ArrayRef<TensorOpDesc> SupportedOps,
                         const TensorCostModelParams &Params,
                         unsigned BeamWidth) {
  const LoopNestInfo &NI = Initial.getNestInfo();
  unsigned Depth = static_cast<unsigned>(NI.IVs.size());

  // Build PF candidates per dimension from trip counts.
  SmallVector<SmallVector<uint32_t>> Candidates(Depth);
  for (unsigned D = 0; D < Depth; ++D) {
    if (const auto *C = dyn_cast_or_null<SCEVConstant>(NI.IVs[D].TripCount)) {
      uint64_t TC = C->getValue()->getZExtValue() + 1; // backedge-taken + 1
      uint32_t PF = 1;
      while (PF <= TC && PF < 512) {
        Candidates[D].push_back(PF);
        PF *= 2;
      }
    } else {
      Candidates[D] = {1, 4, 8, 16};
    }
    if (Candidates[D].empty())
      Candidates[D].push_back(1);
  }

  // Per-dimension beam search.
  SmallVector<TPlan, 0> Beam;
  Beam.push_back(std::move(Initial));

  for (unsigned D = 0; D < Depth; ++D) {
    SmallVector<TPlan, 0> NextBeam;
    for (auto &Plan : Beam) {
      SmallVector<uint32_t> PFs(Plan.getAllPFs().begin(),
                                Plan.getAllPFs().end());
      for (uint32_t PF : Candidates[D]) {
        SmallVector<uint32_t> NewPFs = PFs;
        NewPFs[D] = PF;
        TPlan NewPlan = Plan.withPFs(NewPFs);
        NewPlan.setCost(costTPlan(NewPlan, SupportedOps, Params));
        NextBeam.push_back(std::move(NewPlan));
      }
    }
    std::sort(NextBeam.begin(), NextBeam.end(),
              [](const TPlan &A, const TPlan &B) {
                return A.getCost() < B.getCost();
              });
    if (NextBeam.size() > BeamWidth)
      NextBeam.resize(BeamWidth);
    Beam = std::move(NextBeam);
  }

  // Select best plan.
  TPlan *Best = &Beam[0];
  for (auto &P : Beam)
    if (P.getCost() < Best->getCost())
      Best = &P;

  // Post-search Conv2D: set UseIm2Col if col_matrix fits in L2.
  for (auto &R : Best->recipes()) {
    if (auto *CR = dyn_cast<TPComputeRecipe>(&R)) {
      if (CR->Pattern == PatternKind::Conv2D) {
        float ColMatrixBytes = 1.0f;
        for (uint32_t PF : Best->getAllPFs())
          ColMatrixBytes *= static_cast<float>(PF);
        ColMatrixBytes *= 4.0f; // assume float
        CR->UseIm2Col = (ColMatrixBytes <= static_cast<float>(Params.L2Size));
      }
      break;
    }
  }

  return std::move(*Best);
}
