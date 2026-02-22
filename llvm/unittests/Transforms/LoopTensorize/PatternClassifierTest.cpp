//===- PatternClassifierTest.cpp - Tests for TensorPatternClassifier ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "gtest/gtest.h"

using namespace llvm;

// Helper: build a LoopNestInfo that looks like a GEMM nest
static LoopNestInfo makeGEMMLike() {
  LoopNestInfo Info;
  Info.Depth = 3;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;

  // 3 distinct base pointers, 2 reads + 1 write
  // We use nullptr as placeholders; classifyPattern only inspects Kind counts
  Info.Accesses.resize(3);
  Info.Accesses[0].Kind = AccessKind::Read;   // A
  Info.Accesses[1].Kind = AccessKind::Read;   // B
  Info.Accesses[2].Kind = AccessKind::Write;  // C

  // 3 distinct base ptrs (nullptr != nullptr would be false, so use distinct
  // globals -- just make them non-equal by using distinct Values via offsets)
  // Since we can't easily create Values here, set BasePtr to distinct
  // dummy pointers using reinterpret_cast from integers:
  Info.Accesses[0].BasePtr = reinterpret_cast<Value *>(0x1000);
  Info.Accesses[1].BasePtr = reinterpret_cast<Value *>(0x2000);
  Info.Accesses[2].BasePtr = reinterpret_cast<Value *>(0x3000);
  return Info;
}

TEST(PatternClassifierTest, GEMMLikeIsClassifiedAsGEMM) {
  auto Info = makeGEMMLike();
  PatternHint Hint = classifyPattern(Info);
  EXPECT_EQ(Hint.Kind, PatternKind::GEMM);
}

TEST(PatternClassifierTest, Depth1IsNotGEMM) {
  LoopNestInfo Info;
  Info.Depth = 1;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;
  PatternHint Hint = classifyPattern(Info);
  EXPECT_NE(Hint.Kind, PatternKind::GEMM);
}

TEST(PatternClassifierTest, NonAffineIsGeneric) {
  auto Info = makeGEMMLike();
  Info.IsAffine = false;
  PatternHint Hint = classifyPattern(Info);
  EXPECT_EQ(Hint.Kind, PatternKind::Generic);
}
