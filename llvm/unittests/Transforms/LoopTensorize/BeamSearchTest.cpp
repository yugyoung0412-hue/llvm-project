//===- BeamSearchTest.cpp - Tests for runBeamSearch -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "gtest/gtest.h"

using namespace llvm;

// Build a synthetic GEMM-like LoopNestInfo (depth=3, affine, perfect, 3
// accesses). No real IR is required — the classifier and cost model only use
// the structural fields.
static LoopNestInfo buildGEMMNestInfo() {
  LoopNestInfo Info;
  Info.Depth = 3;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;
  // Add three synthetic memory accesses to satisfy the GEMM 2R+1W check.
  MemAccess ReadA;
  ReadA.Kind = AccessKind::Read;
  MemAccess ReadB;
  ReadB.Kind = AccessKind::Read;
  MemAccess WriteC;
  WriteC.Kind = AccessKind::Write;
  // Use distinct base pointers so the 3-distinct-bases check passes.
  // We cannot allocate real Values here, but we can cast dummy integers.
  static int DummyA = 0, DummyB = 1, DummyC = 2;
  ReadA.BasePtr = reinterpret_cast<Value *>(&DummyA);
  ReadB.BasePtr = reinterpret_cast<Value *>(&DummyB);
  WriteC.BasePtr = reinterpret_cast<Value *>(&DummyC);
  Info.Accesses.push_back(ReadA);
  Info.Accesses.push_back(ReadB);
  Info.Accesses.push_back(WriteC);
  return Info;
}

TEST(BeamSearchTest, TensorRecognizeWinsForGEMM) {
  LoopNestInfo GEMMInfo = buildGEMMNestInfo();

  // A single synthetic MatMul TensorOpDesc.
  TensorOpDesc MatMulOp;
  MatMulOp.OpKind = TensorOpDesc::Kind::MatMul;
  MatMulOp.M = 16;
  MatMulOp.N = 16;
  MatMulOp.K = 16;
  SmallVector<TensorOpDesc> SupportedOps = {MatMulOp};

  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 1e12f;   // high tensor throughput
  Params.PeakVectorFLOPS = 1e11f;
  Params.PeakScalarFLOPS = 2e9f;
  Params.MemBandwidth = 50e9f;

  SearchState Initial;
  Initial.Current = GEMMInfo;
  Initial.Cost = std::numeric_limits<float>::infinity();
  Initial.IsTerminal = false;

  SearchState Best = runBeamSearch(Initial, SupportedOps, Params, /*BeamWidth=*/8);

  // With a high-throughput tensor op available, beam search should prefer
  // TensorRecognize over plain Vectorize.
  bool HasTensorRecognize = llvm::any_of(Best.Applied, [](const Transform &T) {
    return T.Kind == TransformKind::TensorRecognize;
  });
  EXPECT_TRUE(HasTensorRecognize);
}

TEST(BeamSearchTest, VectorizeWinsWithNoTensorOps) {
  LoopNestInfo GEMMInfo = buildGEMMNestInfo();

  // No tensor ops — empty list.
  SmallVector<TensorOpDesc> SupportedOps;

  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 0.0f;    // no tensor hardware
  Params.PeakVectorFLOPS = 1e11f;
  Params.PeakScalarFLOPS = 2e9f;
  Params.MemBandwidth = 50e9f;

  SearchState Initial;
  Initial.Current = GEMMInfo;
  Initial.Cost = std::numeric_limits<float>::infinity();
  Initial.IsTerminal = false;

  SearchState Best = runBeamSearch(Initial, SupportedOps, Params, /*BeamWidth=*/8);

  // Without tensor ops, Vectorize should be chosen.
  bool HasVectorize = llvm::any_of(Best.Applied, [](const Transform &T) {
    return T.Kind == TransformKind::Vectorize;
  });
  EXPECT_TRUE(HasVectorize);
}
