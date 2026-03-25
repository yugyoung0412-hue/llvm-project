# TPlan Build + Print Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TPlan::buildInitial()` and `TPlan::print()`, wire them into `LoopTensorize.cpp` after `analyzeLoopNest()`, and run the pass on `ggml_compute_forward_mul_mat.ll` to produce a human-readable TPlan dump.

**Architecture:** Two new files (`TPlan.h`, `TPlan.cpp`) define a recipe-based IR — one `TPInductionRecipe` per loop IV, one `TPMemRecipe` per memory access, one `TPComputeRecipe` for the detected compute kind. `LoopTensorize.cpp` calls `TPlan::buildInitial(*InfoOpt)` immediately after `analyzeLoopNest()` succeeds and emits the plan via `LLVM_DEBUG(Plan.print(dbgs()))` under the existing `loop-tensorize` debug type. The final step runs `opt -debug-only=loop-tensorize` on the example `.ll` and saves stderr to `ggml_compute_forward_mul_mat_tplan.txt`.

**Tech Stack:** LLVM C++17, `iplist<>` / `ilist_node<>` (ADT), `ScalarEvolution` SCEV printing, `raw_ostream`.

**Reference design doc:** `docs/plans/2026-03-04-tplan-design.md` (Status: Approved).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Full class hierarchy: `TPRecipeBase`, `TPInductionRecipe`, `TPMemRecipe`, `TPComputeRecipe`, `TPlan` |
| **Create** | `llvm/lib/Transforms/Vectorize/TPlan.cpp` | `buildInitial()` factory + all `print()` implementations |
| **Modify** | `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | Add `TPlan.cpp` after `TensorCostModel.cpp` |
| **Modify** | `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp` | Include `TPlan.h`; call `buildInitial` + `LLVM_DEBUG(Plan.print(dbgs()))` after `analyzeLoopNest` |
| **Create** | `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll` | Lit test: verifies TPlan text output format via FileCheck |
| **Create** | `ggml_compute_forward_mul_mat_tplan.txt` | Captured TPlan dump from running the pass on the example input |

---

## Task 1: Create `TPlan.h`

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

- [ ] **Step 1: Write the header**

```cpp
//===- TPlan.h - Tensor Plan IR for LoopTensorize -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLAN_H

#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/LoopNestAnalyzer.h"
#include <cstdint>
#include <limits>

namespace llvm {

class TPlan;

//===----------------------------------------------------------------------===//
// TPRecipeBase — base for all TPlan recipe nodes
//===----------------------------------------------------------------------===//
class TPRecipeBase : public ilist_node<TPRecipeBase> {
public:
  enum class RecipeKind { Induction, Mem, Compute };

  RecipeKind getKind() const { return Kind; }
  /// Back-pointer to the owning TPlan. Set by TPlan::buildInitial().
  TPlan *getPlan() const { return Plan; }

  virtual void print(raw_ostream &OS) const = 0;
  virtual ~TPRecipeBase() = default;

protected:
  explicit TPRecipeBase(RecipeKind K) : Kind(K) {}

private:
  RecipeKind Kind;
  TPlan *Plan = nullptr; // set after construction by buildInitial()
  friend class TPlan;
};

//===----------------------------------------------------------------------===//
// TPInductionRecipe — one per loop induction variable
//===----------------------------------------------------------------------===//
class TPInductionRecipe : public TPRecipeBase {
public:
  TPInductionRecipe(unsigned Dim, const InductionDesc &D)
      : TPRecipeBase(RecipeKind::Induction), DimIndex(Dim), Desc(D) {}

  unsigned getDimIndex() const { return DimIndex; }
  uint32_t getPF() const { return PF; }
  void setPF(uint32_t V) { PF = V; }
  const InductionDesc &getDesc() const { return Desc; }

  void print(raw_ostream &OS) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::Induction;
  }

private:
  unsigned DimIndex;
  InductionDesc Desc;
  uint32_t PF = 1;
};

//===----------------------------------------------------------------------===//
// TPMemRecipe — one per load/store in the loop body
//===----------------------------------------------------------------------===//
class TPMemRecipe : public TPRecipeBase {
public:
  TPMemRecipe(const MemAccess &MA, bool Write)
      : TPRecipeBase(RecipeKind::Mem), Access(MA), IsWrite(Write) {}

  bool isWrite() const { return IsWrite; }
  const MemAccess &getAccess() const { return Access; }

  void print(raw_ostream &OS) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::Mem;
  }

private:
  MemAccess Access;
  bool IsWrite;
};

//===----------------------------------------------------------------------===//
// TPComputeRecipe — the single compute node for the whole nest
//===----------------------------------------------------------------------===//
class TPComputeRecipe : public TPRecipeBase {
public:
  enum class ComputeKind { Elementwise, Reduction, MatMul, Conv };

  explicit TPComputeRecipe(ComputeKind K)
      : TPRecipeBase(RecipeKind::Compute), CKind(K) {}

  ComputeKind getComputeKind() const { return CKind; }

  void print(raw_ostream &OS) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getKind() == RecipeKind::Compute;
  }

private:
  ComputeKind CKind;
};

//===----------------------------------------------------------------------===//
// TPlan — the top-level container; owns its recipes via iplist
//===----------------------------------------------------------------------===//
class TPlan {
public:
  /// Build an initial TPlan from an analyzed loop nest.
  /// PFs are all set to 1; searchTPlan() will assign optimal values later.
  static TPlan buildInitial(const LoopNestInfo &Info);

  uint32_t getPF(unsigned Dim) const {
    return Dim < PFs.size() ? PFs[Dim] : 1u;
  }
  float getCost() const { return Cost; }
  void setCost(float C) { Cost = C; }
  const LoopNestInfo &getNestInfo() const { return NestInfo; }

  void print(raw_ostream &OS) const;

private:
  LoopNestInfo NestInfo;
  SmallVector<uint32_t> PFs; // PFs[i] == parallel factor for IVs[i]; all 1 at build time
  iplist<TPRecipeBase> Recipes;
  float Cost = std::numeric_limits<float>::infinity();
};

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLAN_H
```

- [ ] **Step 2: Verify it compiles standalone** (no .cpp yet — just make sure the header parses)

```bash
cd /Users/yun-yugyeong/Dev/llvm
echo '#include "llvm/Transforms/Vectorize/TPlan.h"' | \
  build/bin/clang++ -std=c++17 -x c++ - \
  -I llvm/include -I build/include -fsyntax-only 2>&1 | head -20
```

Expected: no errors (warnings about unused includes are OK).

---

## Task 2: Write the Lit Test (Failing First)

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll`

- [ ] **Step 1: Write a minimal test loop and CHECK patterns**

```llvm
; RUN: opt -passes=loop-tensorize -debug-only=loop-tensorize %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Verify that LoopTensorize builds and prints an initial TPlan for a simple
; 3-deep loop nest (GEMM shape: 2 reads + 1 write).

; CHECK: TPlan: depth=3
; CHECK: TPInductionRecipe[0]:
; CHECK: TPInductionRecipe[1]:
; CHECK: TPInductionRecipe[2]:
; CHECK: TPMemRecipe[read]
; CHECK: TPMemRecipe[read]
; CHECK: TPMemRecipe[write]
; CHECK: TPComputeRecipe: MatMul

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Simple 3-deep GEMM loop: C[i][j] += A[i][k] * B[k][j]
define void @gemm(ptr %A, ptr %B, ptr %C, i64 %M, i64 %N, i64 %K) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %middle

middle:
  %j = phi i64 [ 0, %outer ], [ %j.next, %middle.latch ]
  br label %inner

inner:
  %k = phi i64 [ 0, %middle ], [ %k.next, %inner.latch ]
  %ai = mul i64 %i, %K
  %ak = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av = load float, ptr %aptr, align 4
  %bk = mul i64 %k, %N
  %bj = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %ci = mul i64 %i, %N
  %cj = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  %cv = load float, ptr %cptr, align 4
  %sum = fadd float %cv, %prod
  store float %sum, ptr %cptr, align 4
  br label %inner.latch

inner.latch:
  %k.next = add nuw nsw i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %middle.latch, label %inner

middle.latch:
  %j.next = add nuw nsw i64 %j, 1
  %j.done = icmp eq i64 %j.next, %N
  br i1 %j.done, label %outer.latch, label %middle

outer.latch:
  %i.next = add nuw nsw i64 %i, 1
  %i.done = icmp eq i64 %i.next, %M
  br i1 %i.done, label %exit, label %outer

exit:
  ret void
}
```

- [ ] **Step 2: Confirm the test fails before implementation**

```bash
cd /Users/yun-yugyeong/Dev/llvm
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll 2>&1 | tail -20
```

Expected: FAIL — `TPlan:` not found in output (TPlan.cpp not yet compiled in).

---

## Task 3: Implement `TPlan.cpp` and Register in CMakeLists

**Files:**
- Create: `llvm/lib/Transforms/Vectorize/TPlan.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`

- [ ] **Step 1: Write `TPlan.cpp`**

```cpp
//===- TPlan.cpp - Tensor Plan IR for LoopTensorize -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlan.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// TPlan::buildInitial
//===----------------------------------------------------------------------===//

TPlan TPlan::buildInitial(const LoopNestInfo &Info) {
  TPlan P;
  P.NestInfo = Info;
  P.PFs.assign(Info.Depth, 1u);

  // 1. One induction recipe per IV (outermost → innermost).
  for (unsigned I = 0, N = Info.IVs.size(); I < N; ++I)
    P.Recipes.push_back(*new TPInductionRecipe(I, Info.IVs[I]));

  // 2. One mem recipe per memory access (reads first, then writes, preserving
  //    the order returned by analyzeLoopNest).
  for (const auto &MA : Info.Accesses) {
    bool Write =
        (MA.Kind == AccessKind::Write || MA.Kind == AccessKind::ReadWrite);
    P.Recipes.push_back(*new TPMemRecipe(MA, Write));
  }

  // 3. One compute recipe — kind inferred from access signature and depth.
  unsigned Reads = 0, Writes = 0;
  for (const auto &MA : Info.Accesses) {
    if (MA.Kind == AccessKind::Read)
      ++Reads;
    else
      ++Writes; // Write or ReadWrite
  }

  TPComputeRecipe::ComputeKind CK = TPComputeRecipe::ComputeKind::Elementwise;
  if (Reads >= 2 && Writes >= 1 && Info.Depth >= 4)
    CK = TPComputeRecipe::ComputeKind::Conv;
  else if (Reads >= 2 && Writes >= 1 && Info.Depth >= 3)
    CK = TPComputeRecipe::ComputeKind::MatMul;

  P.Recipes.push_back(*new TPComputeRecipe(CK));

  // 4. Wire Plan back-pointers now that the list is stable (no more moves).
  for (auto &R : P.Recipes)
    R.Plan = &P;

  return P;
}

//===----------------------------------------------------------------------===//
// print() implementations
//===----------------------------------------------------------------------===//

void TPInductionRecipe::print(raw_ostream &OS) const {
  OS << "  TPInductionRecipe[" << DimIndex << "]: PF=" << PF;
  if (Desc.TripCount) {
    OS << " trip=";
    Desc.TripCount->print(OS);
  }
  if (Desc.Step) {
    OS << " step=";
    Desc.Step->print(OS);
  }
  OS << "\n";
}

void TPMemRecipe::print(raw_ostream &OS) const {
  OS << "  TPMemRecipe[" << (IsWrite ? "write" : "read") << "]: base=";
  if (Access.BasePtr)
    Access.BasePtr->printAsOperand(OS, /*PrintType=*/false);
  else
    OS << "<null>";
  OS << " elem=";
  if (Access.ElemType)
    Access.ElemType->print(OS);
  else
    OS << "<null>";
  OS << "\n";
}

void TPComputeRecipe::print(raw_ostream &OS) const {
  OS << "  TPComputeRecipe: ";
  switch (CKind) {
  case ComputeKind::Elementwise: OS << "Elementwise"; break;
  case ComputeKind::Reduction:   OS << "Reduction";   break;
  case ComputeKind::MatMul:      OS << "MatMul";      break;
  case ComputeKind::Conv:        OS << "Conv";        break;
  }
  OS << "\n";
}

void TPlan::print(raw_ostream &OS) const {
  OS << "TPlan: depth=" << NestInfo.Depth
     << " cost=" << Cost
     << " perfect=" << NestInfo.IsPerfectNest
     << " affine=" << NestInfo.IsAffine << "\n";
  for (const auto &R : Recipes)
    R.print(OS);
}
```

**Important note on the Plan back-pointer after `return P`:**
`TPlan::buildInitial()` sets `R.Plan = &P` before returning. With NRVO (Named Return Value Optimization), the compiler constructs `P` directly in the caller's storage, so the pointer is stable. NRVO applies here because there is a single named return value. However, if a compiler fails to apply NRVO (e.g., with `-fno-elide-constructors`), the pointer would be stale. The `Plan` back-pointer is only needed by future `applyTPlan()` (out of scope here); `print()` does not use it, so this is safe for the current task.

- [ ] **Step 2: Add `TPlan.cpp` to `CMakeLists.txt`**

In `llvm/lib/Transforms/Vectorize/CMakeLists.txt`, add `TPlan.cpp` after `TensorCostModel.cpp`:

```cmake
  TensorCostModel.cpp
  TPlan.cpp
```

---

## Task 4: Wire TPlan into `LoopTensorize.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`

- [ ] **Step 1: Add the include**

At the top of `LoopTensorize.cpp`, after the other tensor includes:

```cpp
#include "llvm/Transforms/Vectorize/TPlan.h"
```

- [ ] **Step 2: Call `buildInitial` and print after `analyzeLoopNest`**

In `LoopTensorizePass::run()`, the existing code is:

```cpp
    auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
    if (!InfoOpt)
      continue;

    PatternHint Hint = classifyPattern(*InfoOpt);
```

Insert the TPlan build + print between the null check and `classifyPattern`:

```cpp
    auto InfoOpt = analyzeLoopNest(RawNest, SE, DI);
    if (!InfoOpt)
      continue;

    // Build and print the initial TPlan (all PFs = 1).
    TPlan Plan = TPlan::buildInitial(*InfoOpt);
    LLVM_DEBUG(Plan.print(dbgs()));

    PatternHint Hint = classifyPattern(*InfoOpt);
```

---

## Task 5: Build and Verify the Lit Test

- [ ] **Step 1: Build the Vectorize library and opt**

```bash
ninja -C /Users/yun-yugyeong/Dev/llvm/build LLVMVectorize opt 2>&1 | tail -20
```

Expected: build succeeds with no errors.

- [ ] **Step 2: Run the lit test**

```bash
cd /Users/yun-yugyeong/Dev/llvm
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
```

Expected: PASS — all CHECK lines match.

If the test fails because `analyzeLoopNest` rejects the simple loop (non-affine / non-perfect), inspect with:

```bash
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll -o /dev/null 2>&1
```

Fix the test loop IR if needed (e.g., simplify the CFG or use `opt -O1` to normalize it first).

---

## Task 6: Run on `ggml_compute_forward_mul_mat.ll` and Save TPlan

- [ ] **Step 1: Run the pass on the example input**

```bash
cd /Users/yun-yugyeong/Dev/llvm
build/bin/opt \
  -passes=loop-tensorize \
  -debug-only=loop-tensorize \
  ggml_compute_forward_mul_mat.ll \
  -o /dev/null \
  2> ggml_compute_forward_mul_mat_tplan.txt
echo "Exit: $?"
```

Expected: exit 0; `ggml_compute_forward_mul_mat_tplan.txt` contains one or more TPlan blocks.

- [ ] **Step 2: Check the output**

```bash
cat ggml_compute_forward_mul_mat_tplan.txt
```

Expected output shape (values will differ):
```
TPlan: depth=3 cost=inf perfect=1 affine=1
  TPInductionRecipe[0]: PF=1 trip=<scev-expr> step=<1>
  TPInductionRecipe[1]: PF=1 trip=<scev-expr> step=<1>
  TPInductionRecipe[2]: PF=1 trip=<scev-expr> step=<1>
  TPMemRecipe[read]: base=%A elem=float
  TPMemRecipe[read]: base=%B elem=float
  TPMemRecipe[write]: base=%C elem=float
  TPComputeRecipe: MatMul
```

**If the output is empty** (no TPlan lines): the ggml loop nest has dynamic strides or non-affine accesses that `analyzeLoopNest` rejects. In that case, add a temporary `LLVM_DEBUG(dbgs() << "analyzeLoopNest returned nullopt\n")` before the `continue` to confirm, then note the limitation in a comment.

- [ ] **Step 3: Commit**

```bash
cd /Users/yun-yugyeong/Dev/llvm
git add \
  llvm/include/llvm/Transforms/Vectorize/TPlan.h \
  llvm/lib/Transforms/Vectorize/TPlan.cpp \
  llvm/lib/Transforms/Vectorize/CMakeLists.txt \
  llvm/lib/Transforms/Vectorize/LoopTensorize.cpp \
  llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll \
  ggml_compute_forward_mul_mat_tplan.txt
git commit -m "$(cat <<'EOF'
[LoopTensorize] Add TPlan::buildInitial and print; wire into pass pipeline

Implements the initial TPlan construction (TPlan.h, TPlan.cpp) per the
approved design doc (docs/plans/2026-03-04-tplan-design.md). Wires
buildInitial() into LoopTensorizePass::run() after analyzeLoopNest() and
emits the plan via LLVM_DEBUG. Adds a lit test and the TPlan dump for
ggml_compute_forward_mul_mat.ll.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```
