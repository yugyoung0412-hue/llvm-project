//===- TransformSpaceTest.cpp - Tests for TensorTransformSpace ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(TransformSpaceTest, LegalTransformsForDepth1) {
  LoopNestInfo Info;
  Info.Depth = 1;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;
  auto Transforms = getLegalTransforms(Info);
  bool HasVectorize = llvm::any_of(Transforms, [](const Transform &T) {
    return T.Kind == TransformKind::Vectorize;
  });
  EXPECT_TRUE(HasVectorize);
}

TEST(TransformSpaceTest, ApplyVectorizeIsTerminal) {
  SearchState State;
  State.Current.Depth = 1;
  State.Current.IsAffine = true;
  Transform T;
  T.Kind = TransformKind::Vectorize;
  auto Next = applyTransform(State, T);
  ASSERT_TRUE(Next.has_value());
  EXPECT_TRUE(Next->IsTerminal);
}

TEST(TransformSpaceTest, ApplyLoopTileIncreasesDepth) {
  SearchState State;
  State.Current.Depth = 1;
  State.Current.IsAffine = true;
  Transform T;
  T.Kind = TransformKind::LoopTile;
  T.Dim = 0;
  T.Size = 16;
  auto Next = applyTransform(State, T);
  ASSERT_TRUE(Next.has_value());
  EXPECT_EQ(Next->Current.Depth, 2u);
  EXPECT_FALSE(Next->IsTerminal);
}
