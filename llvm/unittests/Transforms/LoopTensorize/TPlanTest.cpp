//===- TPlanTest.cpp - Unit tests for TPlan -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// TPlan::getPF / setPF tests
// ---------------------------------------------------------------------------

TEST(TPlanTest, DefaultPFIsOne) {
  TPlan Plan;
  EXPECT_EQ(Plan.getPF(0), 1u);
  EXPECT_EQ(Plan.getPF(7), 1u);
}

TEST(TPlanTest, SetAndGetPF) {
  TPlan Plan;
  Plan.setPF(0, 16);
  Plan.setPF(1, 8);
  EXPECT_EQ(Plan.getPF(0), 16u);
  EXPECT_EQ(Plan.getPF(1), 8u);
  EXPECT_EQ(Plan.getPF(2), 1u); // unset dims default to 1
}

// ---------------------------------------------------------------------------
// TPlan structural block tests
// ---------------------------------------------------------------------------

TEST(TPlanTest, PlanHasFourStructuralBlocks) {
  TPlan Plan;
  EXPECT_NE(Plan.getEntry(), nullptr);
  EXPECT_NE(Plan.getVectorBody(), nullptr);
  EXPECT_NE(Plan.getVectorBody()->getHeader(), nullptr);
  EXPECT_NE(Plan.getVectorBody()->getLatch(), nullptr);
  EXPECT_NE(Plan.getMiddleBlock(), nullptr);
  // ScalarTail is optional; defaults to null.
  EXPECT_EQ(Plan.getScalarTail(), nullptr);
}

TEST(TPlanTest, EnableScalarTail) {
  TPlan Plan;
  EXPECT_EQ(Plan.getScalarTail(), nullptr);
  Plan.enableScalarTail();
  EXPECT_NE(Plan.getScalarTail(), nullptr);
}

// ---------------------------------------------------------------------------
// TPBasicBlock::appendRecipe test
// ---------------------------------------------------------------------------

TEST(TPBasicBlockTest, AppendRecipe) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);

  TPlan Plan;
  TPBasicBlock *Header = Plan.getVectorBody()->getHeader();
  EXPECT_TRUE(Header->empty());

  auto *R = new TPHeaderPHIRecipe(0, 1, nullptr, nullptr, I64);
  Header->appendRecipe(R);

  EXPECT_FALSE(Header->empty());
  EXPECT_EQ(Header->size(), 1u);
  EXPECT_EQ(R->getParent(), Header);
}

// ---------------------------------------------------------------------------
// TPlan::verify test
// ---------------------------------------------------------------------------

TEST(TPlanTest, VerifyDefaultPlan) {
  TPlan Plan;
  EXPECT_TRUE(Plan.verify());
}

// ---------------------------------------------------------------------------
// TPlan pattern/op metadata
// ---------------------------------------------------------------------------

TEST(TPlanTest, PatternMetadata) {
  TPlan Plan;
  EXPECT_EQ(Plan.getPattern(), PatternKind::Generic);
  Plan.setPattern(PatternKind::GEMM);
  EXPECT_EQ(Plan.getPattern(), PatternKind::GEMM);
}

// ---------------------------------------------------------------------------
// TPlan::resolvePF with beam tile transforms
// ---------------------------------------------------------------------------

TEST(TPlanTest, ResolvePFFromBeamTiles) {
  TPlan Plan;
  Transform T;
  T.Kind = TransformKind::LoopTile;
  T.Dim = 0;
  T.Size = 32;
  Plan.resolvePF(TensorOpDesc{}, {}, {T});
  EXPECT_EQ(Plan.getPF(0), 32u);
}
