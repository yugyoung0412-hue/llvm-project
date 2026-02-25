//===- CostModelTest.cpp - Tests for TensorCostModel ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(CostModelTest, TensorRecognizeScoreBetterThanVectorize) {
  LoopNestInfo Info;
  Info.Depth = 3;
  Info.IsAffine = true;

  SearchState TensorState;
  TensorState.Current = Info;
  Transform TR;
  TR.Kind = TransformKind::TensorRecognize;
  TensorState.Applied.push_back(TR);
  TensorState.IsTerminal = true;

  SearchState VectorState;
  VectorState.Current = Info;
  Transform TV;
  TV.Kind = TransformKind::Vectorize;
  VectorState.Applied.push_back(TV);
  VectorState.IsTerminal = true;

  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 1e12f;
  Params.PeakVectorFLOPS = 1e11f;
  Params.PeakScalarFLOPS = 2e9f;
  Params.MemBandwidth     = 50e9f;

  float TensorCost = scoreCost(TensorState, Params);
  float VectorCost = scoreCost(VectorState, Params);
  EXPECT_LT(TensorCost, VectorCost);
}

TEST(CostModelTest, ScalarCostHigherThanVector) {
  LoopNestInfo Info;
  Info.Depth = 1;

  SearchState ScalarState;
  ScalarState.Current = Info;
  ScalarState.IsTerminal = true;

  SearchState VectorState;
  VectorState.Current = Info;
  Transform TV;
  TV.Kind = TransformKind::Vectorize;
  VectorState.Applied.push_back(TV);
  VectorState.IsTerminal = true;

  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 0.0f;
  Params.PeakVectorFLOPS = 1e11f;
  Params.PeakScalarFLOPS = 2e9f;
  Params.MemBandwidth     = 50e9f;

  float ScalarCost = scoreCost(ScalarState, Params);
  float VectorCost = scoreCost(VectorState, Params);
  EXPECT_LT(VectorCost, ScalarCost);
}
