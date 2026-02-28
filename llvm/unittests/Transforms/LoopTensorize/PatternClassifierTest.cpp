//===- PatternClassifierTest.cpp - Tests for TensorPatternClassifier ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

// LLVM IR for a 4-deep Conv2D nest:
//   output[oh*6+ow] += input[(oh+kh)*8+(ow+kw)] * kernel[kh*3+kw]
// Each outer loop is {header -> body -> latch} in LoopSimplifyForm.
// The innermost (kw) loop is a single-block self-loop.
// SCEV for the input pointer is a nested SCEVAddRecExpr chain;
// collectTermsByStep must walk it recursively to find two AddRecs
// sharing the same step value, which is the Conv2D sliding-window signature.
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

/// Parse IR, run LoopNestAnalyzer on the first loop forest, call Test.
static void runClassify(
    const char *IR,
    function_ref<void(const LoopNestInfo &, PatternHint)> Test) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(IR, Err, Ctx);
  ASSERT_NE(M, nullptr) << Err.getMessage().str();
  Function *F = M->getFunction("conv2d");
  if (!F) F = &*M->begin();
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
    if (!Info) continue;
    PatternHint Hint = classifyPattern(*Info);
    Test(*Info, Hint);
    return;
  }
  FAIL() << "No analyzable loop nest found";
}

TEST(PatternClassifierTest, Conv2DIsClassifiedAsConv2D) {
  runClassify(Conv2DIR, [](const LoopNestInfo &Info, PatternHint Hint) {
    EXPECT_EQ(Hint.Kind, PatternKind::Conv2D)
        << "Expected Conv2D but got "
        << static_cast<int>(Hint.Kind);
  });
}

TEST(PatternClassifierTest, Conv2DIsNotMisclassifiedAsGEMM) {
  runClassify(Conv2DIR, [](const LoopNestInfo &Info, PatternHint Hint) {
    EXPECT_NE(Hint.Kind, PatternKind::GEMM);
  });
}
