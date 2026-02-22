//===- TensorISAInfoTest.cpp - Unit tests for TensorISAInfo ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

// Default TTI (no target) should report no tensor ops.
TEST(TensorISAInfoTest, DefaultTTIHasNoTensorOps) {
  LLVMContext Ctx;
  Module M("test", Ctx);
  TargetTransformInfo TTI(M.getDataLayout());
  EXPECT_FALSE(TTI.hasTensorOps());
  EXPECT_TRUE(TTI.getSupportedTensorOps().empty());
}
