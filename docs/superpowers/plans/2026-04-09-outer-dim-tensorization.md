# Outer-Dim Tensorization via buildInitial() Ordering Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `buildInitial()` ordering bug in `TPlan.cpp` so that outer-loop body blocks are inserted into ValueMap *before* the inner region is recursively built, enabling correct DimSet propagation for preheader-split GEP chains (e.g. ggml-style GEMM where A's row pointer is computed between the M-loop header and the N-loop).

**Architecture:** A single reordering inside `BuildRegion` in `TPlan.cpp`: the `EmitBlock` loop that processes outer-loop body blocks (lines 864–870) currently runs *after* the recursive `BuildRegion(Idx+1)` call (line 851). Moving those lines *before* the recursive call ensures that any intermediate values (e.g. `%a.row = gep %A, %i * %K`) are in ValueMap when the inner loop body GEP recipes are built, so DimSet BFS propagates the M-dim bit through the A GEP chain correctly.

**Tech Stack:** C++ (LLVM), LLVM IR lit tests, `opt`/`llvm-lit` for testing.

---

## Background: Root Cause

In `TPlan.cpp`, `BuildRegion(Idx)` processes one loop level. For non-innermost loops it:

1. Emits the header (line 835) → puts outer IVs (e.g. `%i`, `%j`) in `ValueMap`
2. **Line 851**: `auto *Child = BuildRegion(Idx + 1)` ← recurses into inner loop **immediately**
3. **Lines 864–870**: Emits outer body blocks (not header, not latch, not in inner loop) into InnerPH

Step 3 is too late. When `BuildRegion(Idx+1)` recurses and eventually processes the k-loop body, it calls `P.getTPValue(%a.row)`. Since `i.body` (the block computing `%a.row`) hasn't been emitted yet, `%a.row` becomes `ir<%a.row>` — a live-in with `DimSet={}`. The BFS never propagates the M-dim bit through this live-in. Result: A-load `DimSet={0}` (K only) → `tensor.contract.1d.2d.1d.f32` instead of `tensor.contract.2d.2d.2d.f32`.

**Fix**: Move lines 864–870 (the `InnerPH->setInsertionBB` + the `EmitBlock` loop) to **before** line 851 (the recursive call).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Fix `BuildRegion` ordering: emit outer body blocks before recursing |
| Create | `llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll` | New lit test: preheader-split GEP → must produce `tensor.contract.2d.2d.2d.f32` |

---

## Task 1: Write the Failing Lit Test

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll`

- [ ] **Step 1: Write the test file**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; 3-loop GEMM (M=N=K=16, static) where A's row pointer is computed in a
; separate body block (i.body) between the M-loop header and the N-loop.
;
; This exercises the buildInitial() ordering bug: without the fix,
; i.body is emitted AFTER BuildRegion(j.loop) returns, so %a.row is
; seen as a live-in ir<%a.row> (DimSet={}) when the k-loop body builds
; its GEP recipe. BFS then gives A-load DimSet={0} (K only), producing
; tensor.contract.1d.2d.1d.f32.
;
; After the fix, i.body is emitted BEFORE the recursive BuildRegion call,
; %a.row resolves to tp<%a.row> (DimSet will contain M-dim via BFS),
; A-load DimSet={0,2}, B-load DimSet={0,1}, OutputDimSet={1,2}, RankC=2
; → tensor.contract.2d.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: call void @llvm.tensor.contract.1d.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_preheader_gep(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %i.body

i.body:
  ; A's row pointer computed in a separate body block — between the M-loop
  ; header and the N-loop. This is the pattern that triggers the bug.
  %a.row.off = mul i64 %i, 16
  %a.row = getelementptr float, ptr %A, i64 %a.row.off
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.body ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; A[i][k]: uses %a.row from i.body (cross-block, outer-loop body reference)
  %a.ptr = getelementptr float, ptr %a.row, i64 %k
  ; B[k][j]: flat 2-D layout, K×N
  %bk    = mul i64 %k, 16
  %bj    = add i64 %bk, %j
  %b.ptr = getelementptr float, ptr %B, i64 %bj
  %av    = load float, ptr %a.ptr
  %bv    = load float, ptr %b.ptr
  %mul   = fmul float %av, %bv
  %sum   = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop

k.latch:
  %ci    = mul i64 %i, 16
  %cj    = add i64 %ci, %j
  %c.ptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %c.ptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop

j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop

exit:
  ret void
}
```

- [ ] **Step 2: Run the test to verify it currently FAILS**

```bash
cd /Users/yun-yugyeong/Dev/llvm
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
```

Expected: **FAIL** — `tensor.contract.2d.2d.2d.f32` not found (currently emits `1d` or falls back to scalar).

- [ ] **Step 3: Commit the failing test**

```bash
cd /Users/yun-yugyeong/Dev/llvm
git add llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
git commit -m "$(cat <<'EOF'
test: add failing lit for preheader-split GEP outer-dim DimSet propagation

The test has A's row pointer computed in a separate i.body block between
the M-loop header and the N-loop. Without the buildInitial() fix this
GEP is invisible to the k-loop BFS and A-load gets DimSet={K only},
producing the wrong tensor.contract rank.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fix `buildInitial()` Ordering in `TPlan.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp` lines 846–870

The current code (abbreviated):

```cpp
if (Idx + 1 < AllLoops.size()) {
  unsigned ChildLevel = Level - 1;
  std::string ChildStr = std::to_string(ChildLevel);
  auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);
  auto *Child = BuildRegion(Idx + 1);          // ← BUG: recurse FIRST
  auto *MiddleBB = P.createTPBasicBlock(...);
  // ...
  InnerPH->setInsertionBB(L->getHeader());     // ← emitted AFTER recursion
  Loop *InnerLoop = AllLoops[Idx + 1];
  for (BasicBlock *BB : L->blocks()) {
    if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
    if (InnerLoop->contains(BB)) continue;
    EmitBlock(BB, InnerPH);                    // ← too late: inner already built
  }
```

- [ ] **Step 1: Apply the fix**

Open `llvm/lib/Transforms/Vectorize/TPlan.cpp`. Locate the block starting at line 846:

```cpp
    if (Idx + 1 < AllLoops.size()) {
      // Non-innermost: create inner preheader + recurse.
      unsigned ChildLevel = Level - 1;
      std::string ChildStr = std::to_string(ChildLevel);
      auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);
      auto *Child = BuildRegion(Idx + 1);
      auto *MiddleBB = P.createTPBasicBlock("middle.block" + ChildStr);
```

Replace with (moving the `setInsertionBB` + `EmitBlock` loop to BEFORE `BuildRegion`):

```cpp
    if (Idx + 1 < AllLoops.size()) {
      // Non-innermost: create inner preheader + recurse.
      unsigned ChildLevel = Level - 1;
      std::string ChildStr = std::to_string(ChildLevel);
      auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);

      // Emit outer body blocks BEFORE recursing into the inner region.
      // Blocks that belong to L but are neither the header, the latch, nor
      // inside the inner loop (e.g. a row-pointer preheader like A's row GEP
      // in ggml-style GEMM) must be in ValueMap when BuildRegion processes the
      // inner loop body. Without this ordering the inner GEP recipe sees a
      // live-in ir<> for %a.row instead of a TP recipe, so BFS never propagates
      // the outer-dim bit and DimSet ends up K-only.
      InnerPH->setInsertionBB(L->getHeader());
      Loop *InnerLoop = AllLoops[Idx + 1];
      for (BasicBlock *BB : L->blocks()) {
        if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
        if (InnerLoop->contains(BB)) continue;
        EmitBlock(BB, InnerPH);
      }

      auto *Child = BuildRegion(Idx + 1);
      auto *MiddleBB = P.createTPBasicBlock("middle.block" + ChildStr);
```

Also remove the now-duplicate lines 864–870 that appear later in the same `if` block (they were the original `setInsertionBB` + `EmitBlock` loop position).

- [ ] **Step 2: Build the affected target**

```bash
cd /Users/yun-yugyeong/Dev/llvm
ninja -C build LLVMVectorize 2>&1 | tail -20
```

Expected: build succeeds with no errors.

- [ ] **Step 3: Run the new test to verify it now PASSES**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
```

Expected: **PASS** — `tensor.contract.2d.2d.2d.f32` found, `tensor.contract.1d.` not found.

- [ ] **Step 4: Commit the fix**

```bash
cd /Users/yun-yugyeong/Dev/llvm
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "$(cat <<'EOF'
tplan: emit outer body blocks before recursing into inner region in buildInitial()

BuildRegion(Idx) previously called BuildRegion(Idx+1) at line 851, then
emitted outer body blocks (the EmitBlock loop) at lines 864-870. Any
value computed in those outer body blocks (e.g. A's row-pointer GEP in
ggml-style GEMM: %a.row = gep %A, %i * K, in a block between the M-loop
header and the N-loop) was not in ValueMap when the inner loop body's GEP
recipes were built. Those recipes fell back to ir<> live-ins with DimSet={},
so BFS never propagated the outer-dim bit through the A GEP chain.

Fix: move the setInsertionBB + EmitBlock loop to BEFORE the recursive
BuildRegion call. After this change A-load correctly gets DimSet={K,M},
OutputDimSet={M,N}, RankC=2, and emits tensor.contract.2d.2d.2d.f32.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Regression Tests

- [ ] **Step 1: Run the full LoopTensorize lit suite**

```bash
cd /Users/yun-yugyeong/Dev/llvm
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all previously passing tests still pass; new test passes.

Key tests to watch (most likely to regress):

| Test | What it checks |
|------|---------------|
| `basic/tensor-contract-gemm-2d.ll` | Flat GEP GEMM still emits `2d.2d.2d` |
| `basic/pf-dimset-gemm.ll` | DimSet propagation baseline |
| `basic/ptr-decompose-srem.ll` | srem stop + outer-body GEP (batched GEMM) |
| `basic/tensor-contract-batched.ll` | Batched contraction ranks |
| `basic/skeleton-guard.ll` | Guard + dynamic-K tiling |
| `basic/tiling-dynamic-multidim.ll` | Multi-dim tiling |
| `remainder/non-divisible-tripcount.ll` | Remainder path |

- [ ] **Step 2: If any test fails, diagnose**

For a DimSet-related failure, run with `-debug-only=loop-tensorize` to see Stage 2 output:

```bash
./build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  -S --disable-verify \
  llvm/test/Transforms/LoopTensorize/basic/<failing-test>.ll 2>&1 | head -80
```

Look for:
- `WIDEN-GEP` recipes where `DimSet` changed (or didn't change when expected)
- `Contraction (contractDim=X)` — ContractDim and A/B DimSet should include ContractDim
- `TPlanLowering: dim=X mode=... PF=...` lines — check Policy classification

- [ ] **Step 3: Commit regression fixes (if any)**

If a regression is found and fixed:

```bash
git add <fixed-files>
git commit -m "$(cat <<'EOF'
tplan: fix regression in <test-name> after buildInitial ordering fix

<Describe what broke and why. E.g.:>
The EmitBlock reordering caused <block> to be emitted before
<other-block> which depended on it, because <reason>. Fix: <remedy>.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- [x] Root cause (buildInitial ordering) → Task 2
- [x] Failing test first (TDD) → Task 1
- [x] Correct intrinsic rank after fix (`2d.2d.2d`) → Task 1 CHECK pattern
- [x] No regressions → Task 3

**Placeholder scan:** No TBDs; all code blocks are complete.

**Type consistency:** `EmitBlock`, `BuildRegion`, `InnerPH`, `InnerLoop` — all refer to the same variables in the existing code; fix uses exact same identifiers.
