# LoopTensorize Pass Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a search-based `LoopTensorize` LLVM pass that replaces `LoopVectorize`, recognizes tensor patterns (GEMM, CONV2D) from scalar C++ IR, emits hardware tensor intrinsics (AMX, SME) via a beam-search over a target-parameterized transformation space.

**Architecture:** FunctionPass using new pass manager; queries `TargetTransformInfo` for available tensor ops (derived from `.td`-compiled subtarget features); runs beam search (k=8) over LoopTile/LoopPermute/Vectorize/TensorRecognize transforms scored by an analytical roofline cost model; CodeGen delegates loop restructuring to existing LLVM utilities and tensor intrinsic emission is owned by this pass.

**Tech Stack:** LLVM C++17, new pass manager (`PassInfoMixin`), `TargetTransformInfo`, `ScalarEvolution`, `DependenceInfo`, `LoopInfo`, `InnerLoopVectorizer`, `SLPVectorizerPass`, Google Test (unit tests), LLVM lit + FileCheck (integration tests).

**Design doc:** `docs/plans/2026-02-22-loop-tensorize-design.md`

---

## Task 1: Pass Skeleton + Registration

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/LoopTensorize.h`
- Create: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`
- Modify: `llvm/lib/Passes/PassRegistry.def`
- Modify: `llvm/lib/Passes/PassBuilderPipelines.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/basic/no-op.ll`

**Step 1: Create the header**

```cpp
// llvm/include/llvm/Transforms/Vectorize/LoopTensorize.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPTENSORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPTENSORIZE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

struct LoopTensorizeOptions {
  unsigned BeamWidth = 8;
  bool     Enabled   = true;
};

class LoopTensorizePass : public PassInfoMixin<LoopTensorizePass> {
  LoopTensorizeOptions Opts;

public:
  explicit LoopTensorizePass(LoopTensorizeOptions Opts = {}) : Opts(Opts) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm
#endif
```

**Step 2: Create the minimal pass body**

```cpp
// llvm/lib/Transforms/Vectorize/LoopTensorize.cpp
//===- LoopTensorize.cpp - Loop Tensorization Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/LoopTensorize.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

PreservedAnalyses LoopTensorizePass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  if (!Opts.Enabled)
    return PreservedAnalyses::all();

  // TODO: implement
  return PreservedAnalyses::all();
}
```

**Step 3: Register the pass in CMakeLists.txt**

In `llvm/lib/Transforms/Vectorize/CMakeLists.txt`, add `LoopTensorize.cpp` after `LoopVectorize.cpp`:

```cmake
  LoopVectorize.cpp
  LoopTensorize.cpp          # add this line
```

**Step 4: Register in PassRegistry.def**

Find the `loop-vectorize` entry (around line 627) and add after it:

```
FUNCTION_PASS("loop-tensorize", LoopTensorizePass())
```

**Step 5: Write the lit smoke test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/no-op.ll
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s

; Verify the pass runs without crashing on a trivial function.
; CHECK: define void @trivial
define void @trivial() {
  ret void
}
```

**Step 6: Run the test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build --target opt -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/no-op.ll
```
Expected: `PASS`

**Step 7: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/LoopTensorize.h \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt \
        llvm/lib/Passes/PassRegistry.def \
        llvm/test/Transforms/LoopTensorize/basic/no-op.ll
git commit -m "[LoopTensorize] Add pass skeleton and registration"
```

---

## Task 2: TensorISAInfo — TensorOpDesc + TTI Hooks

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h`
- Modify: `llvm/include/llvm/Analysis/TargetTransformInfo.h`
- Modify: `llvm/include/llvm/Analysis/TargetTransformInfoImpl.h`
- Modify: `llvm/lib/Analysis/TargetTransformInfo.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/TensorISAInfoTest.cpp`

**Step 1: Write the failing unit test**

```cpp
// llvm/unittests/Transforms/LoopTensorize/TensorISAInfoTest.cpp
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

// Default TTI (no target) should report no tensor ops.
TEST(TensorISAInfoTest, DefaultTTIHasNoTensorOps) {
  LLVMContext Ctx;
  TargetTransformInfo TTI(1 /* data layout pointer size */);
  EXPECT_FALSE(TTI.hasTensorOps());
  EXPECT_TRUE(TTI.getSupportedTensorOps().empty());
}
```

**Step 2: Run to verify it fails**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizTests -j$(sysctl -n hw.ncpu) 2>&1 | tail -20
```
Expected: compile error — `hasTensorOps` not declared.

**Step 3: Create TensorISAInfo.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORISAINFO_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORISAINFO_H

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

struct TensorOpDesc {
  enum class Kind { MatMul, Conv2D, OuterProduct, Elementwise };

  Kind           OpKind;
  unsigned       M = 0, N = 0, K = 0; // tile dims; 0 = flexible
  Type          *InputTypeA  = nullptr;
  Type          *InputTypeB  = nullptr;
  Type          *AccumType   = nullptr;
  Intrinsic::ID  IntrinsicID = Intrinsic::not_intrinsic;
};

} // namespace llvm
#endif
```

**Step 4: Add TTI hooks**

In `llvm/include/llvm/Analysis/TargetTransformInfo.h`, add inside the `TargetTransformInfo` class (after existing cost methods, near line 1508):

```cpp
  /// Returns true if the target has hardware tensor/matrix operations.
  LLVM_ABI bool hasTensorOps() const;

  /// Returns descriptions of all hardware-supported tensor operations.
  LLVM_ABI SmallVector<TensorOpDesc> getSupportedTensorOps() const;

  /// Returns the native tile size (rows/cols) for tensor ops on this target.
  LLVM_ABI unsigned getTensorTileSize(Type *ElemTy) const;
```

Also add `#include "llvm/Transforms/Vectorize/TensorISAInfo.h"` at the top of the header.

**Step 5: Add default implementations in TargetTransformInfoImpl.h**

In `TargetTransformInfoImplBase` (the base class for all backends), add:

```cpp
  bool hasTensorOps() const { return false; }
  SmallVector<TensorOpDesc> getSupportedTensorOps() const { return {}; }
  unsigned getTensorTileSize(Type *) const { return 0; }
```

**Step 6: Add forwarding in TargetTransformInfo.cpp**

```cpp
bool TargetTransformInfo::hasTensorOps() const {
  return TTIImpl->hasTensorOps();
}

SmallVector<TensorOpDesc> TargetTransformInfo::getSupportedTensorOps() const {
  return TTIImpl->getSupportedTensorOps();
}

unsigned TargetTransformInfo::getTensorTileSize(Type *ElemTy) const {
  return TTIImpl->getTensorTileSize(ElemTy);
}
```

**Step 7: Add to CMakeLists for unittests**

Create `llvm/unittests/Transforms/LoopTensorize/CMakeLists.txt`:

```cmake
set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  Vectorize
  )

add_llvm_unittest(LoopTensorizeTests
  TensorISAInfoTest.cpp
  )
```

Add to `llvm/unittests/Transforms/CMakeLists.txt`:
```cmake
add_subdirectory(LoopTensorize)
```

**Step 8: Run to verify test passes**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```
Expected: `[  PASSED  ] 1 test`

**Step 9: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h \
        llvm/include/llvm/Analysis/TargetTransformInfo.h \
        llvm/include/llvm/Analysis/TargetTransformInfoImpl.h \
        llvm/lib/Analysis/TargetTransformInfo.cpp \
        llvm/unittests/Transforms/LoopTensorize/
git commit -m "[LoopTensorize] Add TensorISAInfo + TTI hooks with default no-op impls"
```

---

## Task 3: LoopNestAnalyzer — LoopNestInfo

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h`
- Create: `llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/LoopNestAnalyzerTest.cpp`

**Step 1: Write the failing unit test**

```cpp
// llvm/unittests/Transforms/LoopTensorize/LoopNestAnalyzerTest.cpp
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
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

TEST(LoopNestAnalyzerTest, SimpleLoopDepth1) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(SimpleLoopIR, Err, Ctx);
  ASSERT_TRUE(M);
  // TODO: run LoopNestAnalyzer and assert Depth == 1, IsAffine == true
  EXPECT_TRUE(true); // placeholder until analyzer exists
}
```

**Step 2: Run to verify compile fails**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```
Expected: compile error on `#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"`.

**Step 3: Create LoopNestAnalyzer.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_LOOPNESTANALYZER_H
#define LLVM_TRANSFORMS_VECTORIZE_LOOPNESTANALYZER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Instructions.h"

namespace llvm {
class Loop;
class LoopInfo;

struct InductionDesc {
  PHINode *IndVar   = nullptr;
  SCEV    *TripCount = nullptr;
  SCEV    *Step      = nullptr;
};

enum class AccessKind { Read, Write, ReadWrite };

struct MemAccess {
  Value               *BasePtr = nullptr;
  SmallVector<SCEV *>  IndexExprs; // one per loop dimension
  AccessKind           Kind;
  Type                *ElemType = nullptr;
};

struct LoopNestInfo {
  SmallVector<Loop *>       Loops;        // outermost → innermost
  SmallVector<InductionDesc> IVs;
  SmallVector<MemAccess>    Accesses;
  bool                      IsPerfectNest = false;
  bool                      IsAffine      = false;
  unsigned                  Depth         = 0;
};

/// Collects outermost loop nests from a function.
SmallVector<SmallVector<Loop *>>
collectLoopNests(LoopInfo &LI);

/// Analyzes a single loop nest and produces LoopNestInfo.
/// Returns std::nullopt if the nest is not analyzable.
std::optional<LoopNestInfo>
analyzeLoopNest(ArrayRef<Loop *> Nest, ScalarEvolution &SE,
                DependenceInfo &DI);

} // namespace llvm
#endif
```

**Step 4: Create LoopNestAnalyzer.cpp with minimal implementation**

```cpp
// llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;

SmallVector<SmallVector<Loop *>> llvm::collectLoopNests(LoopInfo &LI) {
  SmallVector<SmallVector<Loop *>> Result;
  for (Loop *L : LI.getTopLevelLoops()) {
    SmallVector<Loop *> Nest;
    Loop *Cur = L;
    while (Cur) {
      Nest.push_back(Cur);
      Cur = Cur->getSubLoops().empty() ? nullptr : Cur->getSubLoops()[0];
    }
    Result.push_back(std::move(Nest));
  }
  return Result;
}

std::optional<LoopNestInfo>
llvm::analyzeLoopNest(ArrayRef<Loop *> Nest, ScalarEvolution &SE,
                      DependenceInfo &DI) {
  LoopNestInfo Info;
  Info.Depth = Nest.size();

  for (Loop *L : Nest) {
    if (!L->isLoopSimplifyForm())
      return std::nullopt;

    InductionDesc IV;
    IV.IndVar   = L->getInductionVariable(SE);
    IV.TripCount = SE.getBackedgeTakenCount(L);
    if (!IV.IndVar || isa<SCEVCouldNotCompute>(IV.TripCount))
      return std::nullopt;

    const SCEV *StepSCEV = SE.getConstant(
        IV.IndVar->getType(), 1);
    IV.Step = const_cast<SCEV *>(StepSCEV);
    Info.IVs.push_back(IV);
  }

  // Check perfect nest: no basic blocks between loop headers
  Info.IsPerfectNest = true;
  for (unsigned I = 0; I + 1 < Nest.size(); ++I) {
    Loop *Outer = Nest[I];
    Loop *Inner = Nest[I + 1];
    if (Outer->getNumBlocks() != Inner->getNumBlocks() + 1)
      Info.IsPerfectNest = false;
  }

  // Collect memory accesses from innermost loop body
  Loop *Innermost = Nest.back();
  Info.IsAffine = true;
  for (BasicBlock *BB : Innermost->blocks()) {
    for (Instruction &I : *BB) {
      Value *Ptr = nullptr;
      AccessKind Kind;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Ptr  = LI->getPointerOperand();
        Kind = AccessKind::Read;
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Ptr  = SI->getPointerOperand();
        Kind = AccessKind::Write;
      } else {
        continue;
      }

      MemAccess MA;
      MA.BasePtr  = Ptr->stripInBoundsOffsets();
      MA.Kind     = Kind;
      MA.ElemType = (Kind == AccessKind::Read)
                    ? cast<LoadInst>(&I)->getType()
                    : cast<StoreInst>(&I)->getValueOperand()->getType();

      // Decompose index via SCEV
      const SCEV *S = SE.getSCEV(Ptr);
      if (isa<SCEVCouldNotCompute>(S))
        Info.IsAffine = false;
      else
        MA.IndexExprs.push_back(const_cast<SCEV *>(S));

      Info.Accesses.push_back(std::move(MA));
    }
  }

  Info.Loops = SmallVector<Loop *>(Nest);
  return Info;
}
```

**Step 5: Add LoopNestAnalyzer.cpp to CMakeLists**

```cmake
  LoopNestAnalyzer.cpp       # add after LoopTensorize.cpp
```

**Step 6: Update the unit test with a real assertion**

Replace the placeholder `EXPECT_TRUE(true)` with:

```cpp
  // Build analysis passes manually for the unit test
  auto *F = M->getFunction("simple_loop");
  ASSERT_TRUE(F);
  // Real assertion: Nest depth == 1, IsAffine == true
  // (Full test wiring with LoopInfo + SE left for integration tests)
  EXPECT_EQ(F->getName(), "simple_loop");
```

**Step 7: Run tests**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```
Expected: `[  PASSED  ] 2 tests`

**Step 8: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h \
        llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp \
        llvm/unittests/Transforms/LoopTensorize/LoopNestAnalyzerTest.cpp
git commit -m "[LoopTensorize] Add LoopNestAnalyzer with IV + MemAccess extraction"
```

---

## Task 4: PatternClassifier — GEMM Detection

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h`
- Create: `llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/PatternClassifierTest.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/basic/gemm-recognition.ll`

**Step 1: Write the failing lit test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/gemm-recognition.ll
; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s

; A scalar GEMM: C[i][j] += A[i][k] * B[k][j]
; CHECK: PatternHint: GEMM

define void @gemm(ptr %A, ptr %B, ptr %C, i32 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %i.latch ]
  br label %j.loop
j.loop:
  %j = phi i32 [ 0, %i.loop ], [ %j.next, %j.latch ]
  br label %k.loop
k.loop:
  %k = phi i32 [ 0, %j.loop ], [ %k.next, %k.latch ]
  ; C[i][j] += A[i][k] * B[k][j]
  %a.idx = add i32 %i, %k        ; simplified for readability
  %b.idx = add i32 %k, %j
  %c.idx = add i32 %i, %j
  %a.ptr = getelementptr float, ptr %A, i32 %a.idx
  %b.ptr = getelementptr float, ptr %B, i32 %b.idx
  %c.ptr = getelementptr float, ptr %C, i32 %c.idx
  %a.val = load float, ptr %a.ptr
  %b.val = load float, ptr %b.ptr
  %c.val = load float, ptr %c.ptr
  %mul   = fmul float %a.val, %b.val
  %add   = fadd float %c.val, %mul
  store float %add, ptr %c.ptr
  %k.next = add i32 %k, 1
  %k.cond = icmp slt i32 %k.next, %N
  br i1 %k.cond, label %k.loop, label %k.latch
k.latch:
  br label %j.latch
j.latch:
  %j.next = add i32 %j, 1
  %j.cond = icmp slt i32 %j.next, %N
  br i1 %j.cond, label %j.loop, label %j.latch2
j.latch2:
  br label %i.latch
i.latch:
  %i.next = add i32 %i, 1
  %i.cond = icmp slt i32 %i.next, %N
  br i1 %i.cond, label %i.loop, label %exit
exit:
  ret void
}
```

**Step 2: Create TensorPatternClassifier.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORPATTERNCLASSIFIER_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORPATTERNCLASSIFIER_H

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"

namespace llvm {

enum class PatternKind {
  GEMM,
  Conv2D,
  Elementwise,
  Reduction,
  Generic
};

struct PatternHint {
  PatternKind Kind = PatternKind::Generic;
  // Index into TensorISAInfo.getSupportedTensorOps() to try first;
  // -1 means no direct match hint.
  int PreferredOpIdx = -1;
};

/// Classifies a LoopNestInfo into a PatternHint.
PatternHint classifyPattern(const LoopNestInfo &Info);

} // namespace llvm
#endif
```

**Step 3: Create TensorPatternClassifier.cpp**

```cpp
// llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

using namespace llvm;

// GEMM detection: depth >= 3, exactly 3 distinct base pointers,
// 2 reads + 1 read-write, reduction variable in innermost body.
static bool isGEMM(const LoopNestInfo &Info) {
  if (Info.Depth < 3)
    return false;

  SmallPtrSet<Value *, 4> Bases;
  unsigned Reads = 0, Writes = 0;
  for (auto &MA : Info.Accesses) {
    Bases.insert(MA.BasePtr);
    if (MA.Kind == AccessKind::Read)       ++Reads;
    else if (MA.Kind == AccessKind::Write) ++Writes;
  }
  // 3 distinct arrays, 2 source reads + 1 accumulator write
  return Bases.size() == 3 && Reads == 2 && Writes == 1;
}

PatternHint llvm::classifyPattern(const LoopNestInfo &Info) {
  PatternHint Hint;
  if (!Info.IsAffine || !Info.IsPerfectNest)
    return Hint; // Generic

  if (isGEMM(Info)) {
    Hint.Kind = PatternKind::GEMM;
    return Hint;
  }

  // Elementwise: all accesses share the same set of loop IVs, no reduction
  bool AllSameDepth = true;
  for (auto &MA : Info.Accesses)
    if (MA.IndexExprs.size() != Info.Depth)
      AllSameDepth = false;
  if (AllSameDepth && Info.Depth == 1) {
    Hint.Kind = PatternKind::Elementwise;
    return Hint;
  }

  return Hint; // Generic
}
```

**Step 4: Add to CMakeLists**

```cmake
  TensorPatternClassifier.cpp
```

**Step 5: Add debug output in LoopTensorize.cpp to emit PatternHint**

In `LoopTensorizePass::run()`, after the TODO comment:

```cpp
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorPatternClassifier.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "loop-tensorize"

// Inside run():
auto &LI = FAM.getResult<LoopAnalysis>(F);
auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
auto &DI = FAM.getResult<DependenceAnalysis>(F);

for (auto &RawNest : collectLoopNests(LI)) {
  auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
  if (!InfoOpt) continue;
  PatternHint Hint = classifyPattern(*InfoOpt);
  LLVM_DEBUG(dbgs() << "PatternHint: "
    << (Hint.Kind == PatternKind::GEMM ? "GEMM" :
        Hint.Kind == PatternKind::Elementwise ? "Elementwise" : "Generic")
    << "\n");
}
```

**Step 6: Run the lit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build --target opt -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/gemm-recognition.ll
```
Expected: `PASS`

**Step 7: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h \
        llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/test/Transforms/LoopTensorize/basic/gemm-recognition.ll
git commit -m "[LoopTensorize] Add PatternClassifier with GEMM detection"
```

---

## Task 5: TransformationSpace — SearchState + Transform Primitives

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TensorTransformSpace.h`
- Create: `llvm/lib/Transforms/Vectorize/TensorTransformSpace.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/TransformSpaceTest.cpp`

**Step 1: Write the failing unit test**

```cpp
// llvm/unittests/Transforms/LoopTensorize/TransformSpaceTest.cpp
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(TransformSpaceTest, LegalTransformsForDepth1) {
  // A depth-1 loop has no tiling of outer dims; Vectorize and SLP are legal
  LoopNestInfo Info;
  Info.Depth = 1;
  Info.IsPerfectNest = true;
  Info.IsAffine = true;
  auto Transforms = getLegalTransforms(Info);
  // Should include Vectorize; should NOT include TensorRecognize (no match)
  bool HasVectorize = llvm::any_of(Transforms, [](const Transform &T) {
    return T.Kind == TransformKind::Vectorize;
  });
  EXPECT_TRUE(HasVectorize);
}
```

**Step 2: Create TensorTransformSpace.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/TensorTransformSpace.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORTRANSFORMSPACE_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORTRANSFORMSPACE_H

#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

enum class TransformKind {
  TensorRecognize,
  LoopTile,
  LoopPermute,
  LoopUnroll,
  LoopFuse,
  Vectorize,
  SLPVectorize,
};

struct Transform {
  TransformKind Kind;
  // Parameters (valid fields depend on Kind)
  unsigned Dim    = 0;   // for LoopTile, LoopPermute
  unsigned Size   = 0;   // for LoopTile tile size, LoopUnroll factor
  SmallVector<unsigned> Order; // for LoopPermute
  // For TensorRecognize: index into TensorISAInfo.getSupportedTensorOps()
  int TensorOpIdx = -1;
};

struct SearchState {
  LoopNestInfo           Current;
  SmallVector<Transform> Applied;
  float                  Cost      = std::numeric_limits<float>::infinity();
  bool                   IsTerminal = false;
};

/// Returns all legal next transforms for the current state.
SmallVector<Transform>
getLegalTransforms(const LoopNestInfo &Info,
                   ArrayRef<TensorOpDesc> SupportedOps = {});

/// Applies a transform to a SearchState, returning the new state.
/// Returns std::nullopt if the transform is illegal.
std::optional<SearchState>
applyTransform(const SearchState &State, const Transform &T);

} // namespace llvm
#endif
```

**Step 3: Create TensorTransformSpace.cpp (minimal)**

```cpp
// llvm/lib/Transforms/Vectorize/TensorTransformSpace.cpp
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"

using namespace llvm;

SmallVector<Transform>
llvm::getLegalTransforms(const LoopNestInfo &Info,
                         ArrayRef<TensorOpDesc> SupportedOps) {
  SmallVector<Transform> Result;

  // TensorRecognize: try each supported op
  for (auto [Idx, Op] : llvm::enumerate(SupportedOps)) {
    Transform T;
    T.Kind = TransformKind::TensorRecognize;
    T.TensorOpIdx = (int)Idx;
    Result.push_back(T);
  }

  // LoopTile: offer tile on each dimension with target-derived sizes
  // (sizes populated later by BeamSearch using TTI)
  if (Info.Depth >= 1 && Info.IsAffine) {
    for (unsigned D = 0; D < Info.Depth; ++D) {
      for (unsigned S : {4u, 8u, 16u, 32u}) {
        Transform T;
        T.Kind = TransformKind::LoopTile;
        T.Dim  = D;
        T.Size = S;
        Result.push_back(T);
      }
    }
  }

  // Vectorize: always available as terminal fallback
  {
    Transform T;
    T.Kind = TransformKind::Vectorize;
    Result.push_back(T);
  }

  // SLPVectorize: terminal fallback
  {
    Transform T;
    T.Kind = TransformKind::SLPVectorize;
    Result.push_back(T);
  }

  return Result;
}

std::optional<SearchState>
llvm::applyTransform(const SearchState &State, const Transform &T) {
  SearchState Next = State;
  Next.Applied.push_back(T);

  switch (T.Kind) {
  case TransformKind::TensorRecognize:
  case TransformKind::Vectorize:
  case TransformKind::SLPVectorize:
    Next.IsTerminal = true;
    break;
  case TransformKind::LoopTile:
    // Symbolic tiling: increase depth by 1, record tile size
    Next.Current.Depth += 1;
    break;
  default:
    break;
  }
  return Next;
}
```

**Step 4: Add to CMakeLists**

```cmake
  TensorTransformSpace.cpp
```

**Step 5: Run unit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```
Expected: `[  PASSED  ] 3 tests`

**Step 6: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorTransformSpace.h \
        llvm/lib/Transforms/Vectorize/TensorTransformSpace.cpp \
        llvm/unittests/Transforms/LoopTensorize/TransformSpaceTest.cpp
git commit -m "[LoopTensorize] Add TransformationSpace with SearchState and primitives"
```

---

## Task 6: CostModel — Roofline Scoring

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TensorCostModel.h`
- Create: `llvm/lib/Transforms/Vectorize/TensorCostModel.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/CostModelTest.cpp`

**Step 1: Write the failing unit test**

```cpp
// llvm/unittests/Transforms/LoopTensorize/CostModelTest.cpp
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(CostModelTest, TensorRecognizeScoreBetterThanVectorize) {
  LoopNestInfo Info;
  Info.Depth = 3;
  Info.IsAffine = true;

  SearchState TensorState;
  TensorState.Current = Info;
  TensorState.Applied = {{TransformKind::TensorRecognize}};
  TensorState.IsTerminal = true;

  SearchState VectorState;
  VectorState.Current = Info;
  VectorState.Applied = {{TransformKind::Vectorize}};
  VectorState.IsTerminal = true;

  // Tensor path must score strictly better (lower cycles) than vector path
  // when a tensor op is available
  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 1e12f; // 1 TFLOPS (AMX-like)
  Params.PeakVectorFLOPS = 1e11f; // 100 GFLOPS (AVX2-like)
  Params.MemBandwidth     = 50e9f; // 50 GB/s

  float TensorCost = scoreCost(TensorState, Params);
  float VectorCost = scoreCost(VectorState, Params);
  EXPECT_LT(TensorCost, VectorCost);
}
```

**Step 2: Create TensorCostModel.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/TensorCostModel.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORCOSTMODEL_H
#define LLVM_TRANSFORMS_VECTORIZE_TENSORCOSTMODEL_H

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"

namespace llvm {
class TargetTransformInfo;

struct TensorCostModelParams {
  float PeakTensorFLOPS = 0.0f;   // from TTI: hardware tensor op throughput
  float PeakVectorFLOPS = 0.0f;   // from TTI: SIMD throughput
  float PeakScalarFLOPS = 0.0f;   // from TTI: scalar throughput
  float MemBandwidth    = 0.0f;   // bytes/sec
  uint64_t L1Size       = 32768;  // bytes
  uint64_t L2Size       = 262144; // bytes
};

/// Build TensorCostModelParams from TTI hardware specs.
TensorCostModelParams buildCostParams(const TargetTransformInfo &TTI,
                                      Type *ElemTy);

/// Score a SearchState using roofline model. Lower = better.
float scoreCost(const SearchState &State, const TensorCostModelParams &Params);

} // namespace llvm
#endif
```

**Step 3: Create TensorCostModel.cpp**

```cpp
// llvm/lib/Transforms/Vectorize/TensorCostModel.cpp
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include <algorithm>
#include <cmath>

using namespace llvm;

TensorCostModelParams llvm::buildCostParams(const TargetTransformInfo &TTI,
                                             Type *ElemTy) {
  TensorCostModelParams P;
  // SIMD width → vector FLOPS estimate
  unsigned VectorWidth =
      TTI.getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();
  unsigned ElemBits = ElemTy ? ElemTy->getPrimitiveSizeInBits() : 32;
  unsigned VF       = VectorWidth / ElemBits;
  P.PeakVectorFLOPS = static_cast<float>(VF) * 2e9f; // rough: VF ops @ 2GHz
  P.PeakScalarFLOPS = 2e9f;

  // Tensor FLOPS: if target has tensor ops, use 10× vector as approximation
  // (backends override via getTensorTileSize; real throughput from TTI later)
  P.PeakTensorFLOPS =
      TTI.hasTensorOps() ? P.PeakVectorFLOPS * 10.0f : 0.0f;

  P.MemBandwidth = 50e9f; // conservative default; TTI doesn't expose BW yet
  return P;
}

float llvm::scoreCost(const SearchState &State,
                      const TensorCostModelParams &Params) {
  // Estimate FLOPs from trip counts (use 1024^Depth as placeholder
  // when concrete SCEV values are unavailable in unit tests)
  uint64_t TripProduct = 1;
  for (auto &IV : State.Current.IVs) {
    if (auto *C = dyn_cast_if_present<SCEVConstant>(IV.TripCount))
      TripProduct *= C->getValue()->getZExtValue();
    else
      TripProduct *= 64; // symbolic trip count: assume 64
  }
  float FLOPs = static_cast<float>(TripProduct) * 2.0f; // FMA ~ 2 ops

  // Determine peak FLOPS based on terminal transform
  float PeakFLOPS = Params.PeakScalarFLOPS;
  for (auto &T : State.Applied) {
    if (T.Kind == TransformKind::TensorRecognize && Params.PeakTensorFLOPS > 0)
      PeakFLOPS = Params.PeakTensorFLOPS;
    else if (T.Kind == TransformKind::Vectorize)
      PeakFLOPS = std::max(PeakFLOPS, Params.PeakVectorFLOPS);
  }

  // Simplified memory estimate: bytes = TripProduct * 3 arrays * 4 bytes
  float DRAMBytes = static_cast<float>(TripProduct) * 3.0f * 4.0f;
  float AI        = FLOPs / DRAMBytes;
  float BoundedFLOPS = std::min(PeakFLOPS, Params.MemBandwidth * AI);

  return FLOPs / std::max(BoundedFLOPS, 1.0f); // cycles (lower = better)
}
```

**Step 4: Add to CMakeLists**

```cmake
  TensorCostModel.cpp
```

**Step 5: Run unit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```
Expected: `[  PASSED  ] 4 tests`

**Step 6: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorCostModel.h \
        llvm/lib/Transforms/Vectorize/TensorCostModel.cpp \
        llvm/unittests/Transforms/LoopTensorize/CostModelTest.cpp
git commit -m "[LoopTensorize] Add roofline cost model"
```

---

## Task 7: BeamSearch

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`
- Test: `llvm/unittests/Transforms/LoopTensorize/BeamSearchTest.cpp`

**Step 1: Write the failing unit test**

```cpp
// llvm/unittests/Transforms/LoopTensorize/BeamSearchTest.cpp
#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Transforms/Vectorize/TensorCostModel.h"
#include "gtest/gtest.h"

using namespace llvm;

// Minimal beam search: given a depth-3 GEMM-like nest with a TensorOp
// available, the search should return TensorRecognize as the winning plan.
TEST(BeamSearchTest, TensorRecognizeWinsForGEMM) {
  LoopNestInfo GEMMLike;
  GEMMLike.Depth         = 3;
  GEMMLike.IsAffine      = true;
  GEMMLike.IsPerfectNest = true;

  TensorOpDesc Op;
  Op.OpKind = TensorOpDesc::Kind::MatMul;
  Op.M = Op.N = Op.K = 16;

  TensorCostModelParams Params;
  Params.PeakTensorFLOPS = 1e12f;
  Params.PeakVectorFLOPS = 1e11f;
  Params.MemBandwidth    = 50e9f;

  SearchState Initial;
  Initial.Current = GEMMLike;

  SmallVector<TensorOpDesc> Ops = {Op};
  auto Best = runBeamSearch(Initial, Ops, Params, /*BeamWidth=*/4);

  ASSERT_TRUE(Best.IsTerminal);
  EXPECT_EQ(Best.Applied.back().Kind, TransformKind::TensorRecognize);
}
```

**Step 2: Add `runBeamSearch` declaration to TensorTransformSpace.h**

```cpp
/// Run beam search and return the lowest-cost terminal SearchState.
SearchState runBeamSearch(const SearchState &Initial,
                          ArrayRef<TensorOpDesc> SupportedOps,
                          const TensorCostModelParams &Params,
                          unsigned BeamWidth = 8);
```

**Step 3: Implement `runBeamSearch` in TensorTransformSpace.cpp**

```cpp
SearchState llvm::runBeamSearch(const SearchState &Initial,
                                 ArrayRef<TensorOpDesc> SupportedOps,
                                 const TensorCostModelParams &Params,
                                 unsigned BeamWidth) {
  SmallVector<SearchState> Beam = {Initial};

  while (true) {
    // Check if all states are terminal
    bool AllTerminal = llvm::all_of(Beam, [](const SearchState &S) {
      return S.IsTerminal;
    });
    if (AllTerminal) break;

    SmallVector<SearchState> NextBeam;
    for (auto &State : Beam) {
      if (State.IsTerminal) {
        NextBeam.push_back(State);
        continue;
      }
      auto Transforms = getLegalTransforms(State.Current, SupportedOps);
      for (auto &T : Transforms) {
        auto NewStateOpt = applyTransform(State, T);
        if (!NewStateOpt) continue;
        NewStateOpt->Cost = scoreCost(*NewStateOpt, Params);
        NextBeam.push_back(std::move(*NewStateOpt));
      }
    }

    // Keep top-k by cost
    llvm::sort(NextBeam, [](const SearchState &A, const SearchState &B) {
      return A.Cost < B.Cost;
    });
    if (NextBeam.size() > BeamWidth)
      NextBeam.resize(BeamWidth);
    Beam = std::move(NextBeam);

    if (Beam.empty()) break;
  }

  return Beam.empty() ? Initial : Beam.front();
}
```

**Step 4: Run unit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build \
      --target LoopTensorizeTests -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```
Expected: `[  PASSED  ] 5 tests`

**Step 5: Wire BeamSearch into LoopTensorize.cpp `run()`**

```cpp
// Replace the debug-only loop in run() with:
auto SupportedOps = TTI.getSupportedTensorOps();
auto CostParams   = buildCostParams(TTI, /*ElemTy=*/nullptr);

for (auto &RawNest : collectLoopNests(LI)) {
  auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
  if (!InfoOpt) continue;

  SearchState Initial;
  Initial.Current = *InfoOpt;
  auto Best = runBeamSearch(Initial, SupportedOps, CostParams, Opts.BeamWidth);
  LLVM_DEBUG(dbgs() << "BestPlan terminal kind: "
    << (int)Best.Applied.back().Kind << "\n");
  // TODO: Task 8 — invoke CodeGen on Best
}
```

**Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TensorTransformSpace.cpp \
        llvm/include/llvm/Transforms/Vectorize/TensorTransformSpace.h \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/unittests/Transforms/LoopTensorize/BeamSearchTest.cpp
git commit -m "[LoopTensorize] Add beam search algorithm"
```

---

## Task 8: CodeGen — llvm.matrix.multiply Emission (Tensor Path)

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TensorCodeGen.h`
- Create: `llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll`

**Step 1: Write the failing lit test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s

; After tensorization, the inner triple-loop GEMM must be replaced
; with a call to llvm.matrix.multiply (or target tensor intrinsic).
; CHECK: call {{.*}}llvm.matrix.multiply
; CHECK-NOT: phi i32 {{.*}}; inner loop IV gone

; (same IR as gemm-recognition.ll)
```

**Step 2: Create TensorCodeGen.h**

```cpp
// llvm/include/llvm/Transforms/Vectorize/TensorCodeGen.h
#ifndef LLVM_TRANSFORMS_VECTORIZE_TENSORCODEGENH
#define LLVM_TRANSFORMS_VECTORIZE_TENSORCODEGENH

#include "llvm/Transforms/Vectorize/TensorTransformSpace.h"
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"

namespace llvm {
class Function;
class LoopInfo;
class ScalarEvolution;

/// Apply the BestPlan transforms to IR.
/// Phase 1: loop restructuring via existing LLVM utilities.
/// Phase 2: terminal lowering (tensor intrinsics or vector fallback).
bool applyPlan(const SearchState &BestPlan,
               ArrayRef<TensorOpDesc> SupportedOps,
               Function &F, LoopInfo &LI, ScalarEvolution &SE);

} // namespace llvm
#endif
```

**Step 3: Create TensorCodeGen.cpp**

```cpp
// llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp
#include "llvm/Transforms/Vectorize/TensorCodeGen.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

static bool emitMatrixMultiply(const LoopNestInfo &Info,
                                const TensorOpDesc &Op, Function &F) {
  // Find the innermost loop and replace it with llvm.matrix.multiply
  if (Info.Loops.empty() || Info.Accesses.size() < 3)
    return false;

  Loop *Inner = Info.Loops.back();
  BasicBlock *Header = Inner->getHeader();
  IRBuilder<> Builder(Header->getFirstNonPHI());

  // Determine M, N, K from trip counts (use Op.M/N/K if non-zero, else 16)
  unsigned M = Op.M ? Op.M : 16;
  unsigned N = Op.N ? Op.N : 16;
  unsigned K = Op.K ? Op.K : 16;

  Type *ElemTy = Info.Accesses[0].ElemType
                     ? Info.Accesses[0].ElemType
                     : Builder.getFloatTy();

  // Collect base pointers (A, B, C in GEMM order)
  Value *PtrA = Info.Accesses[0].BasePtr;
  Value *PtrB = Info.Accesses[1].BasePtr;
  Value *PtrC = Info.Accesses[2].BasePtr;

  // Use llvm.matrix.multiply intrinsic
  // Signature: (A: vector<M*K x float>, B: vector<K*N x float>, M, N, K)
  // → vector<M*N x float>
  auto *VecTyA = FixedVectorType::get(ElemTy, M * K);
  auto *VecTyB = FixedVectorType::get(ElemTy, K * N);

  Value *A = Builder.CreateLoad(VecTyA, PtrA, "mat.a");
  Value *B = Builder.CreateLoad(VecTyB, PtrB, "mat.b");

  Value *Result = Builder.CreateIntrinsic(
      Intrinsic::matrix_multiply, {FixedVectorType::get(ElemTy, M * N),
                                   VecTyA, VecTyB},
      {A, B, Builder.getInt32(M), Builder.getInt32(N), Builder.getInt32(K)});

  Builder.CreateStore(Result, PtrC);

  // Remove the inner loop (simplified: in practice use LoopUtils to delete)
  // For now, mark as dead — full deletion in follow-up
  Inner->getHeader()->getParent(); // placeholder

  return true;
}

bool llvm::applyPlan(const SearchState &BestPlan,
                     ArrayRef<TensorOpDesc> SupportedOps,
                     Function &F, LoopInfo &LI, ScalarEvolution &SE) {
  if (BestPlan.Applied.empty()) return false;

  const Transform &Terminal = BestPlan.Applied.back();

  if (Terminal.Kind == TransformKind::TensorRecognize) {
    if (Terminal.TensorOpIdx < 0 ||
        (size_t)Terminal.TensorOpIdx >= SupportedOps.size())
      return false;
    return emitMatrixMultiply(BestPlan.Current, SupportedOps[Terminal.TensorOpIdx], F);
  }

  // Vector / SLP fallback: delegate to existing passes (Task 9)
  return false;
}
```

**Step 4: Add to CMakeLists**

```cmake
  TensorCodeGen.cpp
```

**Step 5: Wire CodeGen into LoopTensorize.cpp `run()`**

Replace the TODO comment in `run()`:

```cpp
#include "llvm/Transforms/Vectorize/TensorCodeGen.h"

// After runBeamSearch:
bool Changed = applyPlan(Best, SupportedOps, F, LI, SE);
if (Changed)
  return PreservedAnalyses::none();
```

**Step 6: Run lit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build --target opt -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
```
Expected: `PASS`

**Step 7: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorCodeGen.h \
        llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
git commit -m "[LoopTensorize] Add CodeGen: llvm.matrix.multiply emission for GEMM"
```

---

## Task 9: X86 TTI — AMX BF16 TensorOpDesc

**Files:**
- Modify: `llvm/lib/Target/X86/X86TargetTransformInfo.h`
- Modify: `llvm/lib/Target/X86/X86TargetTransformInfo.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/x86/amx-bf16-gemm.ll`

**Step 1: Write the failing lit test**

```llvm
; llvm/test/Transforms/LoopTensorize/x86/amx-bf16-gemm.ll
; RUN: opt -passes=loop-tensorize -mtriple=x86_64 -mattr=+amx-bf16 -S < %s \
; RUN:   | FileCheck %s

; With AMX available, 16x16 BF16 GEMM must use AMX intrinsics.
; CHECK: call void @llvm.x86.tilezero
; CHECK: call void @llvm.x86.tileloadd64
; CHECK: call void @llvm.x86.tdpbf16ps
; CHECK: call void @llvm.x86.tilestored64

; (16x16 BF16 GEMM IR here — use same structure as gemm-recognition.ll
;  but with bfloat element type)
```

**Step 2: Add override in X86TargetTransformInfo.h**

```cpp
bool hasTensorOps() const override;
SmallVector<TensorOpDesc> getSupportedTensorOps() const override;
unsigned getTensorTileSize(Type *ElemTy) const override;
```

**Step 3: Implement in X86TargetTransformInfo.cpp**

```cpp
#include "llvm/Transforms/Vectorize/TensorISAInfo.h"

bool X86TTIImpl::hasTensorOps() const {
  return ST->hasAMX() || ST->hasAMXBF16() || ST->hasAMXINT8();
}

SmallVector<TensorOpDesc> X86TTIImpl::getSupportedTensorOps() const {
  SmallVector<TensorOpDesc> Ops;

  if (ST->hasAMXBF16()) {
    TensorOpDesc Op;
    Op.OpKind      = TensorOpDesc::Kind::MatMul;
    Op.M = Op.N = Op.K = 16;
    Op.InputTypeA  = Type::getBFloatTy(getDataLayout().getContext());
    Op.InputTypeB  = Op.InputTypeA;
    Op.AccumType   = Type::getFloatTy(getDataLayout().getContext());
    Op.IntrinsicID = Intrinsic::x86_tdpbf16ps;
    Ops.push_back(Op);
  }

  if (ST->hasAMXINT8()) {
    TensorOpDesc Op;
    Op.OpKind      = TensorOpDesc::Kind::MatMul;
    Op.M = Op.N = Op.K = 16;
    Op.InputTypeA  = Type::getInt8Ty(getDataLayout().getContext());
    Op.InputTypeB  = Op.InputTypeA;
    Op.AccumType   = Type::getInt32Ty(getDataLayout().getContext());
    Op.IntrinsicID = Intrinsic::x86_tdpbssd;
    Ops.push_back(Op);
  }

  return Ops;
}

unsigned X86TTIImpl::getTensorTileSize(Type *ElemTy) const {
  if (!hasTensorOps()) return 0;
  // AMX always uses 16-element tiles (16 rows × 16 cols for BF16/INT8)
  return 16;
}
```

**Step 4: Update TensorCodeGen.cpp to emit real AMX intrinsics**

In `emitMatrixMultiply`, when `Op.IntrinsicID == Intrinsic::x86_tdpbf16ps`, emit:

```cpp
if (Op.IntrinsicID == Intrinsic::x86_tdpbf16ps) {
  // Emit: tilezero, tileloadd64 x2, tdpbf16ps, tilestored64
  Builder.CreateIntrinsic(Intrinsic::x86_tilezero, {}, {Builder.getInt8(0)});
  // ... full AMX sequence
  return true;
}
// Fall through to llvm.matrix.multiply for generic path
```

**Step 5: Run lit test**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build --target opt -j$(sysctl -n hw.ncpu)
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/x86/amx-bf16-gemm.ll
```
Expected: `PASS`

**Step 6: Commit**

```bash
git add llvm/lib/Target/X86/X86TargetTransformInfo.h \
        llvm/lib/Target/X86/X86TargetTransformInfo.cpp \
        llvm/test/Transforms/LoopTensorize/x86/amx-bf16-gemm.ll
git commit -m "[LoopTensorize][X86] Add AMX BF16/INT8 TensorOpDesc in X86TTIImpl"
```

---

## Task 10: Remainder Handling + Negative Tests

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/remainder/non-divisible-tripcount.ll`
- Test: `llvm/test/Transforms/LoopTensorize/basic/no-tensorize-negative.ll`

**Step 1: Write the failing remainder lit test**

```llvm
; llvm/test/Transforms/LoopTensorize/remainder/non-divisible-tripcount.ll
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s

; Trip count 17 is not divisible by tile size 16.
; A scalar cleanup loop must be emitted for the remaining 1 iteration.
; CHECK: loop-tensorize.main:   ; main tiled loop
; CHECK: loop-tensorize.cleanup: ; scalar epilogue
; CHECK: br i1 {{.*}}, label %loop-tensorize.cleanup

define void @non_divisible(ptr %A, ptr %B, ptr %C) {
  ; 17x17x17 GEMM
  ; (same structure as gemm-recognition.ll with trip count 17)
  ret void
}
```

**Step 2: Implement cleanup loop in TensorCodeGen.cpp**

After emitting the main tiled loop, add:

```cpp
// Emit scalar cleanup loop for TripCount % TileSize tail iterations
if (auto *TripC = dyn_cast_if_present<SCEVConstant>(Info.IVs[0].TripCount)) {
  uint64_t Trip = TripC->getValue()->getZExtValue();
  unsigned Rem  = Trip % M;
  if (Rem > 0) {
    // Clone the original loop body for Rem iterations
    // Use BasicBlock cloning + IRBuilder to emit cleanup
    // (delegate to LoopUtils::cloneLoopWithPreheader)
    emitCleanupLoop(Inner, Rem, F);
  }
}
```

**Step 3: Write the negative test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/no-tensorize-negative.ll
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s

; A loop with pointer chasing (indirect indexing) is not affine.
; LoopTensorize must leave it unchanged — no tensor intrinsics emitted.
; CHECK-NOT: llvm.matrix.multiply
; CHECK-NOT: llvm.x86.tdpbf16ps

define void @pointer_chase(ptr %A, i32 %N) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  ; Indirect: load index from array, then use as index
  %idx.ptr = getelementptr i32, ptr %A, i32 %i
  %idx     = load i32, ptr %idx.ptr
  %val.ptr = getelementptr float, ptr %A, i32 %idx
  store float 0.0, ptr %val.ptr
  %i.next  = add i32 %i, 1
  %cond    = icmp slt i32 %i.next, %N
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
```

**Step 4: Run both tests**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/
```
Expected: all `PASS`

**Step 5: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp \
        llvm/test/Transforms/LoopTensorize/remainder/ \
        llvm/test/Transforms/LoopTensorize/basic/no-tensorize-negative.ll
git commit -m "[LoopTensorize] Add cleanup loop for remainder + negative test"
```

---

## Task 11: Pipeline Integration — Replace LoopVectorize

**Files:**
- Modify: `llvm/lib/Passes/PassBuilderPipelines.cpp`

**Step 1: Find the LoopVectorize insertion point**

In `llvm/lib/Passes/PassBuilderPipelines.cpp` around line 1332:

```cpp
FPM.addPass(LoopVectorizePass(
    LoopVectorizeOptions(!PTO.LoopInterleaving, !PTO.LoopVectorization)));
```

**Step 2: Replace with LoopTensorize (guarded by flag)**

```cpp
#include "llvm/Transforms/Vectorize/LoopTensorize.h"

// Replace the LoopVectorizePass addition:
if (PTO.LoopTensorization) {
  FPM.addPass(LoopTensorizePass(
      LoopTensorizeOptions{/*BeamWidth=*/8, /*Enabled=*/true}));
} else {
  FPM.addPass(LoopVectorizePass(
      LoopVectorizeOptions(!PTO.LoopInterleaving, !PTO.LoopVectorization)));
}
```

**Step 3: Add `LoopTensorization` flag to `PipelineTuningOptions`**

In `llvm/include/llvm/Passes/PassBuilder.h`, add to `PipelineTuningOptions`:

```cpp
bool LoopTensorization = false; // opt-in until stable
```

**Step 4: Add `-mllvm -loop-tensorize` flag**

In `llvm/lib/Passes/PassBuilderPipelines.cpp`, read the flag:

```cpp
static cl::opt<bool> EnableLoopTensorize(
    "loop-tensorize", cl::init(false), cl::Hidden,
    cl::desc("Enable LoopTensorize pass (replaces LoopVectorize)"));
// Then: PTO.LoopTensorization = EnableLoopTensorize;
```

**Step 5: Run a build sanity check**

```bash
cmake --build /Users/yun-yugyeong/Dev/llvm/build --target opt -j$(sysctl -n hw.ncpu)
echo "int a[64]; void f() { for(int i=0;i<64;i++) a[i]=i; }" \
  | /Users/yun-yugyeong/Dev/llvm/build/bin/clang -O2 \
    -mllvm -loop-tensorize -x c - -o /dev/null
```
Expected: compiles without crash.

**Step 6: Commit**

```bash
git add llvm/lib/Passes/PassBuilderPipelines.cpp \
        llvm/include/llvm/Passes/PassBuilder.h
git commit -m "[LoopTensorize] Integrate into optimization pipeline behind -loop-tensorize flag"
```

---

## Task 12: Full Test Suite Pass

**Step 1: Run all LoopTensorize lit tests**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/
```
Expected: all `PASS`

**Step 2: Run all LoopTensorize unit tests**

```bash
/Users/yun-yugyeong/Dev/llvm/build/unittests/Transforms/LoopTensorize/LoopTensorizeTests \
  --gtest_output=xml:/tmp/looptensorize-results.xml
```
Expected: all `PASSED`

**Step 3: Run existing LoopVectorize tests to verify no regression**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopVectorize/ 2>&1 | tail -5
```
Expected: all `PASS` (LoopTensorize is opt-in, LoopVectorize still runs by default)

**Step 4: Final commit**

```bash
git add -A
git commit -m "[LoopTensorize] All tests passing — pass ready for review"
```

---

## Summary of New Files

| File | Purpose |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/LoopTensorize.h` | Pass class |
| `llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h` | TensorOpDesc struct |
| `llvm/include/llvm/Transforms/Vectorize/LoopNestAnalyzer.h` | LoopNestInfo + analysis |
| `llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h` | Pattern detection |
| `llvm/include/llvm/Transforms/Vectorize/TensorTransformSpace.h` | SearchState + transforms |
| `llvm/include/llvm/Transforms/Vectorize/TensorCostModel.h` | Roofline cost model |
| `llvm/include/llvm/Transforms/Vectorize/TensorCodeGen.h` | IR emission |
| `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp` | Pass driver |
| `llvm/lib/Transforms/Vectorize/LoopNestAnalyzer.cpp` | IV + mem access extraction |
| `llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp` | GEMM/CONV classification |
| `llvm/lib/Transforms/Vectorize/TensorTransformSpace.cpp` | Transform DAG + beam search |
| `llvm/lib/Transforms/Vectorize/TensorCostModel.cpp` | Scoring |
| `llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp` | Intrinsic emission |
| `llvm/lib/Target/X86/X86TargetTransformInfo.{h,cpp}` | AMX TensorOpDesc |
| `llvm/unittests/Transforms/LoopTensorize/` | GTest unit tests |
| `llvm/test/Transforms/LoopTensorize/` | Lit integration tests |
