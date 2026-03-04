//===- TPRecipeTest.cpp - Unit tests for TPRecipe -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPRecipe.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// TPValue tests
// ---------------------------------------------------------------------------

TEST(TPValueTest, DefaultPFIsOne) {
  LLVMContext Ctx;
  TPValue V(Type::getInt32Ty(Ctx));
  EXPECT_EQ(V.getParallelFactor(), 1u);
}

TEST(TPValueTest, SetPF) {
  LLVMContext Ctx;
  TPValue V(Type::getInt32Ty(Ctx));
  V.setParallelFactor(8);
  EXPECT_EQ(V.getParallelFactor(), 8u);
}

TEST(TPValueTest, GetType) {
  LLVMContext Ctx;
  Type *Ty = Type::getFloatTy(Ctx);
  TPValue V(Ty);
  EXPECT_EQ(V.getType(), Ty);
}

// ---------------------------------------------------------------------------
// TPUser (via TPHeaderPHIRecipe as concrete recipe) tests
// ---------------------------------------------------------------------------

TEST(TPUserTest, AddAndGetOperand) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPValue Start(I64);
  TPValue Step(I64);
  // TPHeaderPHIRecipe adds Start and Step as operands
  TPHeaderPHIRecipe R(/*DimIdx=*/0, /*PF=*/1, &Start, &Step, I64);
  EXPECT_EQ(R.getNumOperands(), 2u);
  EXPECT_EQ(R.getOperand(0), &Start);
  EXPECT_EQ(R.getOperand(1), &Step);
}

TEST(TPUserTest, AddOperandRegistersUser) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPValue V(I64);
  EXPECT_EQ(V.getNumUsers(), 0u);
  TPHeaderPHIRecipe R(0, 1, &V, nullptr, I64);
  EXPECT_EQ(V.getNumUsers(), 1u);
}

// ---------------------------------------------------------------------------
// TPValue::replaceAllUsesWith tests
// ---------------------------------------------------------------------------

TEST(TPValueTest, ReplaceAllUsesWith) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPValue V1(I64);
  TPValue V2(I64);
  TPHeaderPHIRecipe R(0, 1, &V1, nullptr, I64);
  EXPECT_EQ(R.getOperand(0), &V1);
  EXPECT_EQ(V1.getNumUsers(), 1u);
  V1.replaceAllUsesWith(&V2);
  EXPECT_EQ(R.getOperand(0), &V2);
  EXPECT_EQ(V1.getNumUsers(), 0u);
  EXPECT_EQ(V2.getNumUsers(), 1u);
}

// ---------------------------------------------------------------------------
// TPRecipeBase / RecipeKind tests
// ---------------------------------------------------------------------------

TEST(TPRecipeBaseTest, RecipeBaseKind) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPHeaderPHIRecipe R(0, 1, nullptr, nullptr, I64);
  EXPECT_EQ(R.getKind(), RecipeKind::HeaderPHI);
}

// ---------------------------------------------------------------------------
// TPSingleDefRecipe tests
// ---------------------------------------------------------------------------

TEST(TPSingleDefRecipeTest, SingleDefRecipeDef) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPHeaderPHIRecipe R(0, 1, nullptr, nullptr, I64);
  EXPECT_EQ(R.getNumDefinedValues(), 1u);
  // TPSingleDefRecipe registers itself as the single defined value.
  EXPECT_EQ(R.getDefinedValue(0), static_cast<TPValue *>(&R));
}

TEST(TPSingleDefRecipeTest, ResultType) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPHeaderPHIRecipe R(0, 4, nullptr, nullptr, I64);
  EXPECT_EQ(R.getType(), I64);
  EXPECT_EQ(R.getParallelFactor(), 4u);
}

// ---------------------------------------------------------------------------
// Concrete recipe constructors
// ---------------------------------------------------------------------------

TEST(TPMatMulRecipeTest, NullableAccum) {
  LLVMContext Ctx;
  Type *F32 = Type::getFloatTy(Ctx);
  TPValue A(F32);
  TPValue B(F32);
  // Accum = nullptr (no accumulation)
  TPMatMulRecipe R(4, 4, 4, Intrinsic::matrix_multiply, &A, &B, nullptr, F32);
  EXPECT_EQ(R.getKind(), RecipeKind::MatMul);
  EXPECT_EQ(R.getNumOperands(), 3u);
  EXPECT_FALSE(R.hasAccum());
}

TEST(TPMatMulRecipeTest, WithAccum) {
  LLVMContext Ctx;
  Type *F32 = Type::getFloatTy(Ctx);
  TPValue A(F32);
  TPValue B(F32);
  TPValue Accum(F32);
  TPMatMulRecipe R(4, 4, 4, Intrinsic::matrix_multiply, &A, &B, &Accum, F32);
  EXPECT_TRUE(R.hasAccum());
  EXPECT_EQ(R.getOperand(2), &Accum);
}

TEST(TPWidenLoadRecipeTest, OperandCount) {
  LLVMContext Ctx;
  Type *F32 = Type::getFloatTy(Ctx);
  TPValue Ptr(PointerType::getUnqual(Ctx));
  TPWidenLoadRecipe R(&Ptr, Align(16), F32);
  EXPECT_EQ(R.getKind(), RecipeKind::WidenLoad);
  EXPECT_EQ(R.getNumOperands(), 1u);
  EXPECT_EQ(R.getPointerOperand(), &Ptr);
}

TEST(TPWidenStoreRecipeTest, NoDefinedValues) {
  LLVMContext Ctx;
  Type *F32 = Type::getFloatTy(Ctx);
  TPValue Ptr(PointerType::getUnqual(Ctx));
  TPValue Val(F32);
  TPWidenStoreRecipe R(&Ptr, &Val, Align(16));
  EXPECT_EQ(R.getKind(), RecipeKind::WidenStore);
  EXPECT_EQ(R.getNumDefinedValues(), 0u);
}

TEST(TPBranchOnCountRecipeTest, NoDefinedValues) {
  LLVMContext Ctx;
  Type *I64 = Type::getInt64Ty(Ctx);
  TPValue IV(I64);
  TPValue TC(I64);
  TPBranchOnCountRecipe R(&IV, &TC);
  EXPECT_EQ(R.getKind(), RecipeKind::BranchOnCount);
  EXPECT_EQ(R.getNumDefinedValues(), 0u);
}
