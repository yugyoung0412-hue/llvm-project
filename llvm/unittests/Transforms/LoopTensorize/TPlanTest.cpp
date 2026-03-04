//===- TPlanTest.cpp - Tests for TPlan and TPRecipe -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
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

// Build a GEMM-like LoopNestInfo with 3 IVs and 2 reads + 1 write.
static LoopNestInfo makeGEMMNestInfo() {
  LoopNestInfo Info;
  Info.Depth = 3;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;

  // 3 IVs with nullptr TripCount (will use PF candidates {1,4,8,16} in search)
  Info.IVs.resize(3);

  // 3 accesses: 2 reads (A, B) + 1 write (C)
  Info.Accesses.resize(3);
  Info.Accesses[0].Kind = AccessKind::Read;
  Info.Accesses[0].BasePtr = reinterpret_cast<Value *>(0x1000);
  Info.Accesses[1].Kind = AccessKind::Read;
  Info.Accesses[1].BasePtr = reinterpret_cast<Value *>(0x2000);
  Info.Accesses[2].Kind = AccessKind::Write;
  Info.Accesses[2].BasePtr = reinterpret_cast<Value *>(0x3000);
  return Info;
}

// ---- TPlanBuildInitial -------------------------------------------------------

TEST(TPlanTest, TPlanBuildInitial) {
  LoopNestInfo Info = makeGEMMNestInfo();
  TPlan Plan = TPlan::buildInitial(Info);

  // Count recipes by kind.
  unsigned Inductions = 0, MemReads = 0, MemWrites = 0, Computes = 0;
  TPComputeRecipe::ComputeKind ComputeKind = TPComputeRecipe::Elementwise;

  for (const auto &R : Plan.recipes()) {
    switch (R.getKind()) {
    case TPRecipeBase::Induction:
      ++Inductions;
      break;
    case TPRecipeBase::Mem: {
      const auto &MR = cast<TPMemRecipe>(R);
      if (MR.IsWrite)
        ++MemWrites;
      else
        ++MemReads;
      break;
    }
    case TPRecipeBase::Compute:
      ++Computes;
      ComputeKind = cast<TPComputeRecipe>(R).Kind;
      break;
    }
  }

  EXPECT_EQ(Inductions, 3u);
  EXPECT_EQ(MemReads, 2u);
  EXPECT_EQ(MemWrites, 1u);
  EXPECT_EQ(Computes, 1u);
  EXPECT_EQ(ComputeKind, TPComputeRecipe::MatMul);

  // All PFs should be 1.
  for (unsigned I = 0; I < 3; ++I)
    EXPECT_EQ(Plan.getPF(I), 1u);
}

// ---- TPlanWithPFs -----------------------------------------------------------

TEST(TPlanTest, TPlanWithPFs) {
  LoopNestInfo Info = makeGEMMNestInfo();
  TPlan Original = TPlan::buildInitial(Info);

  TPlan NewPlan = Original.withPFs({256, 128, 64});

  // New plan has the requested PFs.
  EXPECT_EQ(NewPlan.getPF(0), 256u);
  EXPECT_EQ(NewPlan.getPF(1), 128u);
  EXPECT_EQ(NewPlan.getPF(2), 64u);

  // Original plan is unchanged.
  EXPECT_EQ(Original.getPF(0), 1u);
  EXPECT_EQ(Original.getPF(1), 1u);
  EXPECT_EQ(Original.getPF(2), 1u);
}

// ---- ClassifyPatternFromTPlan_GEMM ------------------------------------------

TEST(TPlanTest, ClassifyPatternFromTPlan_GEMM) {
  LoopNestInfo Info = makeGEMMNestInfo();
  TPlan Plan = TPlan::buildInitial(Info);

  PatternHint Hint = classifyPattern(Plan);
  EXPECT_EQ(Hint.Kind, PatternKind::GEMM);
}

// ---- ClassifyPatternFromTPlan_Conv2D ----------------------------------------

// Conv2D IR from PatternClassifierTest — 4-deep nest with sliding-window.
static const char Conv2DIR[] = R"(
define void @conv2d(ptr %input, ptr %kernel, ptr %output) {
entry:
  br label %oh.header

oh.header:
  %oh = phi i32 [ 0, %entry ], [ %oh.next, %oh.latch ]
  br label %ow.header

ow.header:
  %ow = phi i32 [ 0, %oh.header ], [ %ow.next, %ow.latch ]
  br label %kh.header

kh.header:
  %kh = phi i32 [ 0, %ow.header ], [ %kh.next, %kh.latch ]
  br label %kw.header

kw.header:
  %kw = phi i32 [ 0, %kh.header ], [ %kw.next, %kw.header ]
  %ih   = add i32 %oh, %kh
  %row  = mul i32 %ih, 8
  %iw   = add i32 %ow, %kw
  %idx_in = add i32 %row, %iw
  %in.ptr = getelementptr float, ptr %input, i32 %idx_in
  %in.val = load float, ptr %in.ptr
  %ki    = mul i32 %kh, 3
  %kidx  = add i32 %ki, %kw
  %k.ptr = getelementptr float, ptr %kernel, i32 %kidx
  %k.val = load float, ptr %k.ptr
  %oi    = mul i32 %oh, 6
  %oidx  = add i32 %oi, %ow
  %o.ptr = getelementptr float, ptr %output, i32 %oidx
  %o.old = load float, ptr %o.ptr
  %mul   = fmul float %in.val, %k.val
  %acc   = fadd float %o.old, %mul
  store float %acc, ptr %o.ptr
  %kw.next = add i32 %kw, 1
  %kw.cond = icmp slt i32 %kw.next, 3
  br i1 %kw.cond, label %kw.header, label %kh.latch

kh.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.header, label %ow.latch

ow.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 6
  br i1 %ow.cond, label %ow.header, label %oh.latch

oh.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 6
  br i1 %oh.cond, label %oh.header, label %exit

exit:
  ret void
}
)";

TEST(TPlanTest, ClassifyPatternFromTPlan_Conv2D) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(Conv2DIR, Err, Ctx);
  ASSERT_NE(M, nullptr) << Err.getMessage().str();
  Function *F = M->getFunction("conv2d");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoImpl TLII(M->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  DependenceInfo DI(F, nullptr, nullptr, nullptr);

  for (Loop *Root : LI.getTopLevelLoops()) {
    SmallVector<Loop *, 8> Nest;
    Nest.push_back(Root);
    Loop *L = Root;
    while (!L->getSubLoops().empty()) {
      L = L->getSubLoops()[0];
      Nest.push_back(L);
    }
    auto Info = analyzeLoopNest(Nest, SE, DI);
    if (!Info)
      continue;

    TPlan Plan = TPlan::buildInitial(*Info);
    PatternHint Hint = classifyPattern(Plan);
    EXPECT_EQ(Hint.Kind, PatternKind::Conv2D)
        << "Expected Conv2D but got " << static_cast<int>(Hint.Kind);
    return;
  }
  FAIL() << "No analyzable loop nest found";
}

// ---- CostTPlan_LowerWithHigherPF --------------------------------------------

TEST(TPlanTest, CostTPlan_LowerWithHigherPF) {
  LoopNestInfo Info = makeGEMMNestInfo();

  TPlan Plan1 = TPlan::buildInitial(Info);
  // Plan1 has default PFs {1,1,1}

  TPlan Plan16 = Plan1.withPFs({16, 16, 16});

  // Use params with non-zero vector FLOPS and memory bandwidth.
  TensorCostModelParams Params;
  Params.PeakVectorFLOPS = 16e9f;
  Params.PeakScalarFLOPS = 2e9f;
  Params.PeakTensorFLOPS = 0.0f;
  Params.MemBandwidth = 50e9f;

  float Cost1 = costTPlan(Plan1, {}, Params);
  float Cost16 = costTPlan(Plan16, {}, Params);

  // Larger PF → higher arithmetic intensity → better hardware utilization
  // → lower cost (closer to compute-bound optimal).
  EXPECT_LT(Cost16, Cost1)
      << "Expected PF={16,16,16} cost (" << Cost16
      << ") < PF={1,1,1} cost (" << Cost1 << ")";
}
