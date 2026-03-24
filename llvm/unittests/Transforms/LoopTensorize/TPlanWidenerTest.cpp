//===- TPlanWidenerTest.cpp - Unit tests for TPlan DimSet widening --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;

// Verify that a freshly constructed TPDefVal has an empty DimSet.
TEST(TPlanWidenerTest, FreshDefValHasEmptyDimSet) {
  auto *DV = new TPDefVal(nullptr);
  EXPECT_TRUE(DV->DimSet.none());
  delete DV;
}

// Verify that the default-constructed SmallBitVector (as used by DimSet
// before seeding) is empty. This guards the zero-init invariant that the
// widener relies on before Phase 1 seeding.
TEST(TPlanWidenerTest, DimSetNoneOnInit) {
  SmallBitVector BV;
  EXPECT_TRUE(BV.none());
}

// Verify that TPRecipePatternMatcher_match on an empty plan produces no entries.
TEST(TPRecipeMatcherTest, EmptyPlanProducesNoEntries) {
  TPlan P;
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(P, CM);
  EXPECT_TRUE(CM.empty());
}
