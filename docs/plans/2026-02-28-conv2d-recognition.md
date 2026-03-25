# CONV2D Recognition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Conv2D pattern recognition to `LoopTensorize` via a SCEV step-equality check, wire the lowering-decision infrastructure, and guard codegen with a Conv2D branch stub.

**Architecture:** `isConv2D()` detects the sliding-window access signature by walking the pointer SCEV stored in `MemAccess.IndexExprs[0]` and finding two `SCEVAddRecExpr` nodes sharing the same step value (e.g. `oh` and `kh` both step by `W*C`). A `UseIm2Col` flag on `PatternHint` communicates the lowering decision to `applyPlan()`. Im2col codegen is deferred; this PR delivers recognition + infrastructure.

**Tech Stack:** LLVM C++17, `ScalarEvolution` / `SCEVAddRecExpr`, new pass manager, GTest unit tests, LLVM lit + FileCheck.

**Design doc:** `docs/plans/2026-02-28-conv2d-recognition-design.md`

---

## Task 1: Conv2D Pattern Detection in TensorPatternClassifier

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h`
- Modify: `llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp`
- Modify: `llvm/unittests/Transforms/LoopTensorize/PatternClassifierTest.cpp`

### Step 1: Write the failing unit test

Add the following tests to `PatternClassifierTest.cpp`. They require parsing real IR because Conv2D detection walks SCEV objects stored in `MemAccess.IndexExprs`. Add these at the end of the file, after the existing tests.

**New includes at the top of `PatternClassifierTest.cpp`:**

```cpp
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
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
```

**New helper and tests to append to `PatternClassifierTest.cpp`:**

```cpp
// LLVM IR for a 4-deep Conv2D nest:
//   output[oh*6+ow] += input[(oh+kh)*8+(ow+kw)] * kernel[kh*3+kw]
// SCEV for input ptr has two AddRecs with step=8 (oh,kh) and two with step=1
// (ow,kw).  collectTermsByStep groups them → two entries per step → Conv2D.
static const char Conv2DIR[] = R"(
define void @conv2d(ptr %input, ptr %kernel, ptr %output) {
entry:
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %entry ], [ %oh.next, %oh.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %ow.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kh.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.latch ]
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
  br i1 %kw.cond, label %kw.loop, label %kw.latch
kw.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.loop, label %kh.latch
kh.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 6
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 6
  br i1 %oh.cond, label %oh.loop, label %exit
exit:
  ret void
}
)";

/// Parse \p IR, run LoopNestAnalyzer on the first loop forest, and call \p Test
/// with the resulting LoopNestInfo and PatternHint.
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
```

### Step 2: Run the failing tests

```bash
ninja -C build LoopTensorizeTests 2>&1 | tail -5
./build/unittests/Transforms/LoopTensorize/LoopTensorizeTests \
    --gtest_filter="*Conv2D*" 2>&1 | tail -10
```

Expected: compile error or test FAIL with `Expected Conv2D but got 4` (PatternKind::Generic).

### Step 3: Add `collectTermsByStep` and `isConv2D` to `TensorPatternClassifier.cpp`

**Add these new includes** after the existing includes at the top of `TensorPatternClassifier.cpp`:

```cpp
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
```

**Add these two static functions** immediately before the existing `isGEMM` function:

```cpp
/// Walk a SCEV and collect (step, SCEVAddRecExpr*) pairs from linear AddRecs.
/// Groups AddRec nodes by their step operand (by SCEV pointer identity, which
/// is sufficient because SCEV constants are uniqued within an SE instance).
static void collectTermsByStep(
    const SCEV *S,
    DenseMap<const SCEV *, SmallVector<const SCEVAddRecExpr *, 2>> &Out) {
  auto Handle = [&](const SCEV *Op) {
    const auto *AR = dyn_cast<SCEVAddRecExpr>(Op);
    // Only linear (affine) AddRecs: start + step * i
    if (!AR || !AR->isAffine())
      return;
    Out[AR->getOperand(1)].push_back(AR); // operand(1) is the step
  };

  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (const SCEV *Op : Add->operands())
      Handle(Op);
  } else {
    Handle(S);
  }
}

/// Detect Conv2D sliding-window pattern:
///   depth >= 4, perfect+affine, 3 base pointers, 2 reads + 1 write,
///   and at least one read's pointer SCEV has two SCEVAddRecExpr nodes
///   sharing the same step value (e.g. oh and kh both step by W*C).
static bool isConv2D(const LoopNestInfo &Info) {
  if (Info.Depth < 4)
    return false;

  SmallPtrSet<Value *, 4> Bases;
  unsigned Reads = 0, Writes = 0;
  for (const auto &MA : Info.Accesses) {
    Bases.insert(MA.BasePtr);
    if (MA.Kind == AccessKind::Read)
      ++Reads;
    else if (MA.Kind == AccessKind::Write)
      ++Writes;
  }
  if (Bases.size() != 3 || Reads != 2 || Writes != 1)
    return false;

  for (const auto &MA : Info.Accesses) {
    if (MA.Kind != AccessKind::Read || MA.IndexExprs.empty())
      continue;
    DenseMap<const SCEV *, SmallVector<const SCEVAddRecExpr *, 2>> StepMap;
    collectTermsByStep(MA.IndexExprs[0], StepMap);
    for (const auto &[Step, ARs] : StepMap)
      if (ARs.size() >= 2)
        return true; // Two IVs share the same stride → sliding window
  }
  return false;
}
```

### Step 4: Update `classifyPattern` ordering — Conv2D before GEMM

**In `TensorPatternClassifier.cpp`**, modify `classifyPattern` so it checks `isConv2D` before `isGEMM`. A depth-4+ nest that matches both structural tests should be classified as Conv2D, not GEMM.

Replace:

```cpp
PatternHint llvm::classifyPattern(const LoopNestInfo &Info) {
  PatternHint Hint;

  if (!Info.IsAffine || !Info.IsPerfectNest)
    return Hint; // Generic

  if (isGEMM(Info)) {
    Hint.Kind = PatternKind::GEMM;
    return Hint;
  }
```

With:

```cpp
PatternHint llvm::classifyPattern(const LoopNestInfo &Info) {
  PatternHint Hint;

  if (!Info.IsAffine || !Info.IsPerfectNest)
    return Hint; // Generic

  // Check Conv2D first: stricter than GEMM (depth >= 4 + coefficient equality).
  // A 4-deep nest with 3 base pointers would otherwise also match isGEMM.
  if (isConv2D(Info)) {
    Hint.Kind = PatternKind::Conv2D;
    return Hint;
  }

  if (isGEMM(Info)) {
    Hint.Kind = PatternKind::GEMM;
    return Hint;
  }
```

### Step 5: Run the tests and verify they pass

```bash
ninja -C build LoopTensorizeTests 2>&1 | tail -5
./build/unittests/Transforms/LoopTensorize/LoopTensorizeTests \
    --gtest_filter="*Conv2D*"
```

Expected: `[  PASSED  ] 2 tests.`

Also verify no existing test regressed:

```bash
./build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
```

Expected: all previously passing tests still pass.

### Step 6: Commit

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h \
        llvm/lib/Transforms/Vectorize/TensorPatternClassifier.cpp \
        llvm/unittests/Transforms/LoopTensorize/PatternClassifierTest.cpp
git commit -m "[LoopTensorize] Add Conv2D pattern detection via SCEV step equality"
```

---

## Task 2: Wire Conv2D into the Pass Driver + Debug Lit Test

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-gemm.ll`

### Step 1: Write the failing lit test

Create `llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Conv2D: output[oh*6+ow] += input[(oh+kh)*8+(ow+kw)] * kernel[kh*3+kw]
; The input pointer SCEV contains two AddRecs with step 8 (oh,kh) and two
; with step 1 (ow,kw).  isConv2D() fires → PatternHint: Conv2D.
;
; CHECK: PatternHint: Conv2D

define void @conv2d(ptr %input, ptr %kernel, ptr %output) {
entry:
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %entry ], [ %oh.next, %oh.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %ow.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kh.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.latch ]
  %ih     = add i32 %oh, %kh
  %row    = mul i32 %ih, 8
  %iw     = add i32 %ow, %kw
  %idx_in = add i32 %row, %iw
  %in.ptr = getelementptr float, ptr %input, i32 %idx_in
  %in.val = load float, ptr %in.ptr
  %ki     = mul i32 %kh, 3
  %kidx   = add i32 %ki, %kw
  %k.ptr  = getelementptr float, ptr %kernel, i32 %kidx
  %k.val  = load float, ptr %k.ptr
  %oi     = mul i32 %oh, 6
  %oidx   = add i32 %oi, %ow
  %o.ptr  = getelementptr float, ptr %output, i32 %oidx
  %o.old  = load float, ptr %o.ptr
  %mul    = fmul float %in.val, %k.val
  %acc    = fadd float %o.old, %mul
  store float %acc, ptr %o.ptr
  %kw.next = add i32 %kw, 1
  %kw.cond = icmp slt i32 %kw.next, 3
  br i1 %kw.cond, label %kw.loop, label %kw.latch
kw.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.loop, label %kh.latch
kh.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 6
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 6
  br i1 %oh.cond, label %oh.loop, label %exit
exit:
  ret void
}
```

Create `llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-gemm.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; A standard 3-deep GEMM must NOT be classified as Conv2D.
; CHECK: PatternHint: GEMM
; CHECK-NOT: PatternHint: Conv2D

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
  %a.idx = add i32 %i, %k
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
  %j.next = add i32 %j, 1
  %j.cond = icmp slt i32 %j.next, %N
  br i1 %j.cond, label %j.loop, label %j.latch
j.latch:
  %i.next = add i32 %i, 1
  %i.cond = icmp slt i32 %i.next, %N
  br i1 %i.cond, label %i.loop, label %exit
exit:
  ret void
}
```

### Step 2: Run the failing lit tests

```bash
ninja -C build opt
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll
```

Expected: `FAIL` because the debug output says `Generic` or `GEMM`, not `Conv2D`.

### Step 3: Update debug output in `LoopTensorize.cpp`

**In `LoopTensorize.cpp`**, find the `LLVM_DEBUG` block (lines 49–53) that stringifies `PatternHint`. Replace:

```cpp
      LLVM_DEBUG(dbgs() << "PatternHint: "
        << (Hint.Kind == PatternKind::GEMM        ? "GEMM"
          : Hint.Kind == PatternKind::Elementwise ? "Elementwise"
          :                                         "Generic")
        << "\n");
```

With:

```cpp
      LLVM_DEBUG(dbgs() << "PatternHint: "
        << (Hint.Kind == PatternKind::GEMM        ? "GEMM"
          : Hint.Kind == PatternKind::Conv2D      ? "Conv2D"
          : Hint.Kind == PatternKind::Elementwise ? "Elementwise"
          :                                         "Generic")
        << "\n");
```

### Step 4: Run lit tests and verify they pass

```bash
ninja -C build opt
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll \
             llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-gemm.ll
```

Expected: both `PASS`.

Also verify existing tests still pass:

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll \
        llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-gemm.ll
git commit -m "[LoopTensorize] Wire Conv2D debug output; add conv2d recognition lit tests"
```

---

## Task 3: Col-Matrix Size Estimation + UseIm2Col Infrastructure

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h`
- Modify: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/conv2d-im2col-decision.ll`

### Step 1: Write the failing lit test

Create `llvm/test/Transforms/LoopTensorize/basic/conv2d-im2col-decision.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; 5-deep nest: N=1, OH=8, OW=8, KH=3, KW=3.
; col_matrix_bytes = 1*8*8*3*3*4 = 2304 bytes << default 256KB L2.
; Expect: Conv2D detected AND use_im2col=1.
;
; CHECK: PatternHint: Conv2D
; CHECK: Conv2D: col_matrix_bytes={{[0-9]+}} L2={{[0-9]+}} use_im2col=1

define void @conv2d_small(ptr %input, ptr %kernel, ptr %output) {
entry:
  br label %n.loop
n.loop:
  %n = phi i32 [ 0, %entry ], [ %n.next, %n.latch ]
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %n.loop ], [ %oh.next, %oh.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %ow.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kh.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.latch ]
  ; input[(oh+kh)*8 + (ow+kw)]
  %ih     = add i32 %oh, %kh
  %row    = mul i32 %ih, 8
  %iw     = add i32 %ow, %kw
  %idx_in = add i32 %row, %iw
  %in.ptr = getelementptr float, ptr %input, i32 %idx_in
  %in.val = load float, ptr %in.ptr
  ; kernel[kh*3+kw]
  %ki     = mul i32 %kh, 3
  %kidx   = add i32 %ki, %kw
  %k.ptr  = getelementptr float, ptr %kernel, i32 %kidx
  %k.val  = load float, ptr %k.ptr
  ; output[oh*8+ow]
  %oi     = mul i32 %oh, 8
  %oidx   = add i32 %oi, %ow
  %o.ptr  = getelementptr float, ptr %output, i32 %oidx
  %o.old  = load float, ptr %o.ptr
  %mul    = fmul float %in.val, %k.val
  %acc    = fadd float %o.old, %mul
  store float %acc, ptr %o.ptr
  %kw.next = add i32 %kw, 1
  %kw.cond = icmp slt i32 %kw.next, 3
  br i1 %kw.cond, label %kw.loop, label %kw.latch
kw.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.loop, label %kh.latch
kh.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 8
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 8
  br i1 %oh.cond, label %oh.loop, label %oh.latch
oh.latch:
  %n.next = add i32 %n, 1
  %n.cond = icmp slt i32 %n.next, 1
  br i1 %n.cond, label %n.loop, label %exit
exit:
  ret void
}
```

### Step 2: Run the failing test

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/conv2d-im2col-decision.ll
```

Expected: `FAIL` — the debug line `Conv2D: col_matrix_bytes=...` does not exist yet.

### Step 3: Add `UseIm2Col` field to `PatternHint`

**In `TensorPatternClassifier.h`**, modify the `PatternHint` struct:

```cpp
struct PatternHint {
  PatternKind Kind = PatternKind::Generic;
  /// Index into TTI.getSupportedTensorOps() to try first; -1 = no hint.
  int PreferredOpIdx = -1;
  /// True when col_matrix fits in L2 and im2col -> GEMM lowering is preferred.
  /// Only meaningful when Kind == Conv2D.
  bool UseIm2Col = false;
};
```

### Step 4: Add col-matrix estimation to `LoopTensorize.cpp`

**In `LoopTensorize.cpp`**, after the debug output block (after the `LLVM_DEBUG` call that prints `PatternHint`) and before the `ElemTy` selection, add:

```cpp
      // Conv2D lowering decision: estimate col_matrix size and choose
      // im2col -> GEMM when it fits in L2 cache, direct tile otherwise.
      if (Hint.Kind == PatternKind::Conv2D) {
        uint64_t ColMatrixBytes = 1;
        bool AllConstant = true;
        for (const auto &IV : InfoOpt->IVs) {
          if (const auto *SC = dyn_cast<SCEVConstant>(IV.TripCount))
            ColMatrixBytes *= SC->getValue()->getZExtValue() + 1;
          else { AllConstant = false; break; }
        }
        // Multiply by element size (default float = 4 bytes)
        Type *TmpElemTy = InfoOpt->Accesses.empty()
                              ? Type::getFloatTy(F.getContext())
                              : InfoOpt->Accesses[0].ElemType;
        if (!TmpElemTy) TmpElemTy = Type::getFloatTy(F.getContext());
        uint64_t ElemBytes = TmpElemTy->getPrimitiveSizeInBits() / 8;
        if (ElemBytes == 0) ElemBytes = 4;
        ColMatrixBytes *= ElemBytes;

        uint64_t L2Size = 262144; // 256 KiB fallback
        if (auto MaybeL2 = TTI.getCacheSize(TargetTransformInfo::CacheLevel::L2D))
          L2Size = *MaybeL2;

        Hint.UseIm2Col = AllConstant && (ColMatrixBytes <= L2Size);

        LLVM_DEBUG(dbgs() << "Conv2D: col_matrix_bytes=" << ColMatrixBytes
                          << " L2=" << L2Size
                          << " use_im2col=" << Hint.UseIm2Col << "\n");
      }
```

### Step 5: Run all Conv2D lit tests

```bash
ninja -C build opt
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/conv2d-im2col-decision.ll \
             llvm/test/Transforms/LoopTensorize/basic/conv2d-recognition.ll \
             llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-gemm.ll
```

Expected: all three `PASS`.

Confirm no regressions:

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

### Step 6: Commit

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorPatternClassifier.h \
        llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
        llvm/test/Transforms/LoopTensorize/basic/conv2d-im2col-decision.ll
git commit -m "[LoopTensorize] Add Conv2D col_matrix size estimation and UseIm2Col decision"
```

---

## Task 4: Guard `applyPlan` for Conv2D + Negative Lit Tests

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-non-affine.ll`

### Step 1: Write the failing lit test

Create `llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-non-affine.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; A 4-deep loop with indirect (non-affine) indexing must not be transformed.
; The pass should be a no-op: no matrix intrinsics emitted.
;
; CHECK-NOT: @llvm.matrix.multiply
; CHECK-NOT: col_matrix

define void @non_affine(ptr %ptrs, ptr %kernel, ptr %output, i32 %N) {
entry:
  br label %oh.loop
oh.loop:
  %oh = phi i32 [ 0, %entry ], [ %oh.next, %oh.latch ]
  br label %ow.loop
ow.loop:
  %ow = phi i32 [ 0, %oh.loop ], [ %ow.next, %ow.latch ]
  br label %kh.loop
kh.loop:
  %kh = phi i32 [ 0, %ow.loop ], [ %kh.next, %kh.latch ]
  br label %kw.loop
kw.loop:
  %kw = phi i32 [ 0, %kh.loop ], [ %kw.next, %kw.latch ]
  ; Indirect pointer load — non-affine, analyzeLoopNest rejects this nest.
  %pp  = getelementptr ptr, ptr %ptrs, i32 %oh
  %p   = load ptr, ptr %pp
  %val = load float, ptr %p
  %ki  = mul i32 %kh, 3
  %ki2 = add i32 %ki, %kw
  %kp  = getelementptr float, ptr %kernel, i32 %ki2
  %kv  = load float, ptr %kp
  %oi  = mul i32 %oh, 6
  %oi2 = add i32 %oi, %ow
  %op  = getelementptr float, ptr %output, i32 %oi2
  %ov  = load float, ptr %op
  %m   = fmul float %val, %kv
  %a   = fadd float %ov, %m
  store float %a, ptr %op
  %kw.next = add i32 %kw, 1
  %kw.cond = icmp slt i32 %kw.next, 3
  br i1 %kw.cond, label %kw.loop, label %kw.latch
kw.latch:
  %kh.next = add i32 %kh, 1
  %kh.cond = icmp slt i32 %kh.next, 3
  br i1 %kh.cond, label %kh.loop, label %kh.latch
kh.latch:
  %ow.next = add i32 %ow, 1
  %ow.cond = icmp slt i32 %ow.next, 6
  br i1 %ow.cond, label %ow.loop, label %ow.latch
ow.latch:
  %oh.next = add i32 %oh, 1
  %oh.cond = icmp slt i32 %oh.next, 6
  br i1 %oh.cond, label %oh.loop, label %exit
exit:
  ret void
}
```

### Step 2: Run the test to see its current status

```bash
ninja -C build opt
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-non-affine.ll
```

This test is likely already passing (the pass is a no-op for non-affine loops). Confirm.

### Step 3: Guard Conv2D in `applyPlan`

**In `TensorCodeGen.cpp`**, replace the existing early-exit at line 69:

```cpp
  // Only handle GEMM for now.
  if (Hint.Kind != PatternKind::GEMM)
    return false;
```

With:

```cpp
  // Conv2D: recognized but codegen deferred to follow-up PR.
  // Return false so the loop is left unchanged.
  if (Hint.Kind == PatternKind::Conv2D)
    return false;

  if (Hint.Kind != PatternKind::GEMM)
    return false;
```

This makes the Conv2D guard explicit and documents the intent. The existing behavior (no transformation) is preserved; future PRs will fill in the im2col or direct-tile emission.

### Step 4: Run the full test suite

```bash
ninja -C build opt LoopTensorizeTests
./build/unittests/Transforms/LoopTensorize/LoopTensorizeTests
llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add llvm/lib/Transforms/Vectorize/TensorCodeGen.cpp \
        llvm/test/Transforms/LoopTensorize/basic/conv2d-no-match-non-affine.ll
git commit -m "[LoopTensorize] Add explicit Conv2D codegen stub in applyPlan; add negative lit test"
```

---

## Open Issues (Deferred)

| Issue | File | Notes |
|---|---|---|
| Im2col loop emission | `TensorCodeGen.cpp` | Requires emitting new 5-deep loop nest via IRBuilder; follow-up PR |
| Dynamic-shape fallback | `LoopNestAnalyzer.cpp` | Currently rejects `SCEVCouldNotCompute` trip counts; need relaxation |
| Stride/padding in im2col | `TensorCodeGen.cpp` | Strided/padded convolutions need bounds checks in the copy loops |
| ARM SME Conv2D | `AArch64TargetTransformInfo.cpp` | No native conv2d intrinsic in SME; im2col → SME GEMM is the path |
