//===- TensorCostModel.h - Roofline cost model for LoopTensorize ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORCOSTMODEL_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORCOSTMODEL_H

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include <cstdint>

namespace llvm {
class TargetTransformInfo;
class Type;

struct TensorCostModelParams {
  float    PeakTensorFLOPS = 0.0f;
  float    PeakVectorFLOPS = 0.0f;
  float    PeakScalarFLOPS = 0.0f;
  float    MemBandwidth    = 0.0f;
  uint64_t L1Size          = 32768;
  uint64_t L2Size          = 262144;
};

/// Build TensorCostModelParams from TTI hardware specs.
TensorCostModelParams buildCostParams(const TargetTransformInfo &TTI,
                                      Type *ElemTy);

/// Score a SearchState using the roofline model. Lower is better (cycles).
float scoreCost(const SearchState &State, const TensorCostModelParams &Params);

} // namespace llvm
#endif
