//===- TensorCostModel.cpp - Roofline cost model for LoopTensorize --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Vectorize/TPlan.h"
#include <algorithm>
#include <cmath>

using namespace llvm;

TensorCostModelParams llvm::buildCostParams(const TargetTransformInfo &TTI,
                                             Type *ElemTy) {
  TensorCostModelParams P;
  unsigned ElemBits = (ElemTy && ElemTy->getPrimitiveSizeInBits() > 0)
                          ? ElemTy->getPrimitiveSizeInBits()
                          : 32;
  unsigned VectorWidth =
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();
  unsigned VF = (VectorWidth > 0) ? (VectorWidth / ElemBits) : 1;
  P.PeakVectorFLOPS = static_cast<float>(VF) * 2e9f;
  P.PeakScalarFLOPS = 2e9f;
  P.PeakTensorFLOPS = TTI.hasTensorOps() ? P.PeakVectorFLOPS * 10.0f : 0.0f;
  P.MemBandwidth = 50e9f;
  return P;
}

float llvm::scoreCost(const SearchState &State,
                      const TensorCostModelParams &Params) {
  // Estimate FLOPs from trip counts
  uint64_t TripProduct = 1;
  for (const auto &IV : State.Current.IVs) {
    if (IV.TripCount) {
      if (const auto *C = dyn_cast<SCEVConstant>(IV.TripCount))
        TripProduct *= C->getValue()->getZExtValue();
      else
        TripProduct *= 64;
    } else {
      TripProduct *= 64;
    }
  }
  if (State.Current.IVs.empty())
    for (unsigned D = 0; D < State.Current.Depth; ++D)
      TripProduct *= 64;

  float FLOPs = static_cast<float>(TripProduct) * 2.0f;

  // Determine peak FLOPS and data-reuse factor from applied transforms.
  // Tensor operations exploit blocking/tiling for better cache reuse,
  // which reduces effective DRAM traffic and increases arithmetic intensity.
  float PeakFLOPS = Params.PeakScalarFLOPS > 0.0f ? Params.PeakScalarFLOPS : 1.0f;
  float ReuseMultiplier = 1.0f;
  for (const auto &T : State.Applied) {
    if (T.Kind == TransformKind::TensorRecognize &&
        Params.PeakTensorFLOPS > 0.0f) {
      PeakFLOPS = Params.PeakTensorFLOPS;
      // Tensor ops use blocked data movement; model ~sqrt(N) reuse
      ReuseMultiplier = std::max(ReuseMultiplier, 8.0f);
    } else if (T.Kind == TransformKind::Vectorize ||
               T.Kind == TransformKind::SLPVectorize) {
      PeakFLOPS = std::max(PeakFLOPS, Params.PeakVectorFLOPS);
      ReuseMultiplier = std::max(ReuseMultiplier, 2.0f);
    } else if (T.Kind == TransformKind::LoopTile) {
      ReuseMultiplier = std::max(ReuseMultiplier, 4.0f);
    }
  }

  // Roofline bound: effective DRAM bytes decrease with data reuse
  float DRAMBytes = static_cast<float>(TripProduct) * 3.0f * 4.0f / ReuseMultiplier;
  float AI = (DRAMBytes > 0.0f) ? (FLOPs / DRAMBytes) : 1.0f;
  float BW = Params.MemBandwidth > 0.0f ? Params.MemBandwidth : 1.0f;
  float BoundedFLOPS = std::min(PeakFLOPS, BW * AI);
  return FLOPs / std::max(BoundedFLOPS, 1.0f);
}

float llvm::costTPlan(const TPlan &Plan, ArrayRef<TensorOpDesc> SupportedOps,
                      const TensorCostModelParams &Params) {
  // Compute tile FLOPs as product of parallel factors × 2.
  float PFProduct = 1.0f;
  for (uint32_t PF : Plan.getAllPFs())
    PFProduct *= static_cast<float>(PF);
  float FLOPs = PFProduct * 2.0f;

  // Peak FLOPS: tensor if available, then vector, then scalar.
  float PeakFLOPS = Params.PeakScalarFLOPS > 0.0f ? Params.PeakScalarFLOPS : 1.0f;
  bool HasTensorOps = !SupportedOps.empty() && Params.PeakTensorFLOPS > 0.0f;
  if (HasTensorOps)
    PeakFLOPS = Params.PeakTensorFLOPS;
  else if (Params.PeakVectorFLOPS > 0.0f)
    PeakFLOPS = Params.PeakVectorFLOPS;

  float ReuseMultiplier = HasTensorOps ? 8.0f : 1.0f;

  // DRAM bytes per tile: 3 operands × 4 bytes / reuse.
  // Not scaled by PFProduct: bigger tiles amortize the same data over more FLOPs,
  // giving higher arithmetic intensity and better hardware utilization.
  float DRAMBytes = 3.0f * 4.0f / ReuseMultiplier;

  float BW = Params.MemBandwidth > 0.0f ? Params.MemBandwidth : 1.0f;
  float AI = FLOPs / DRAMBytes; // grows with PFProduct
  float BoundedFLOPS = std::min(PeakFLOPS, BW * AI);

  // Cost = hardware inefficiency / tile_size.
  // Lower is better: larger tiles amortize overhead and improve utilization.
  // Dividing by PFProduct ensures the beam search prefers bigger tiles even
  // when all plans are compute-bound (BoundedFLOPS == PeakFLOPS).
  float Cost = (PeakFLOPS / std::max(BoundedFLOPS, 1.0f)) / PFProduct;

  // Conv2D+UseIm2Col: add col_matrix transfer overhead.
  for (const auto &R : Plan.recipes()) {
    if (const auto *CR = dyn_cast<TPComputeRecipe>(&R)) {
      if (CR->Pattern == PatternKind::Conv2D && CR->UseIm2Col) {
        float ColMatrixBytes = PFProduct * 4.0f;
        Cost += ColMatrixBytes / BW;
      }
      break;
    }
  }

  return Cost;
}
