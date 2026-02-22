//===- LoopNestAnalyzerTest.cpp - Unit tests for LoopNestAnalyzer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

// IR for: for (int i = 0; i < 16; i++) A[i] = 0.0f;
static const char SimpleLoopIR[] = R"(
define void @simple_loop(ptr %A) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %gep = getelementptr float, ptr %A, i32 %i
  store float 0.0, ptr %gep
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, 16
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
)";

/// Helper: build LoopInfo + ScalarEvolution and run the given test body.
static void runWithLoopInfoPlus(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, LoopInfo &LI, ScalarEvolution &SE)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find function " << FuncName;

  TargetLibraryInfoImpl TLII(M.getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  Test(*F, LI, SE);
}

TEST(LoopNestAnalyzerTest, SimpleLoopModuleParseable) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(SimpleLoopIR, Err, Ctx);
  ASSERT_TRUE(M) << Err.getMessage().str();
  auto *F = M->getFunction("simple_loop");
  ASSERT_TRUE(F);
  EXPECT_EQ(F->getName(), "simple_loop");
}

TEST(LoopNestAnalyzerTest, CollectsTopLevelLoops) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(SimpleLoopIR, Err, Ctx);
  ASSERT_TRUE(M) << Err.getMessage().str();

  runWithLoopInfoPlus(*M, "simple_loop",
                      [](Function &, LoopInfo &LI, ScalarEvolution &) {
                        auto Nests = collectLoopNests(LI);
                        EXPECT_EQ(Nests.size(), 1u);
                        EXPECT_EQ(Nests[0].size(), 1u); // depth 1
                      });
}

TEST(LoopNestAnalyzerTest, AnalyzeSimpleLoopNest) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(SimpleLoopIR, Err, Ctx);
  ASSERT_TRUE(M) << Err.getMessage().str();

  runWithLoopInfoPlus(
      *M, "simple_loop",
      [](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
        auto Nests = collectLoopNests(LI);
        ASSERT_EQ(Nests.size(), 1u);

        // DependenceInfo is not used for this single-loop test, but the
        // analyzeLoopNest API requires it.
        DependenceInfo DI(&F, nullptr, nullptr, nullptr);
        auto Info = analyzeLoopNest(Nests[0], SE, DI);
        ASSERT_TRUE(Info.has_value());
        EXPECT_EQ(Info->Depth, 1u);
        EXPECT_TRUE(Info->IsPerfectNest);
        EXPECT_EQ(Info->IVs.size(), 1u);
      });
}
