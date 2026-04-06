# Tensor Binary Ops & Dot Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend TPlan lowering to support 1D×1D dot product (RankC=0) and N-D element-wise/broadcast binary ops, and register all tensor intrinsics in `Intrinsics.td`.

**Architecture:** Three sequential work streams: (1) rename existing contract intrinsics to include `<Rc>d` for unambiguous Intrinsics.td registration, (2) fix RankC=0 rejection to enable dot product emission, (3) unify ElementWise+BroadcastBinary into a single `BinaryOp` path that emits `llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>`. All follow TDD: failing test first, then minimal implementation.

**Tech Stack:** C++ (LLVM IR builder), LLVM TableGen (Intrinsics.td), LLVM lit (FileCheck tests)

---

## File Map

| File | Change |
|------|--------|
| `llvm/include/llvm/IR/IntrinsicsTPlan.td` | **Create** — TableGen definitions for all tensor intrinsics |
| `llvm/include/llvm/IR/Intrinsics.td` | **Modify** — add `include "llvm/IR/IntrinsicsTPlan.td"` |
| `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h` | **Modify** — add `BinaryOp` to `TensorOpKind` enum |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | **Modify** — `classifyBinaryOp()` returns `BinaryOp` instead of `ElementWise`/`BroadcastBinary` |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | **Modify** — rename contract name (+Rc), fix RankC=0, add `getTensorBinaryFn()`, `emitBinaryOp()`, update dispatch |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll` | **Modify** — update CHECK to `contract.1d.2d.1d.f32` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll` | **Modify** — update CHECK to `contract.2d.2d.2d.f32` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll` | **Modify** — update CHECK to `contract.3d.3d.3d.f32` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll` | **Create** — 1D×1D dot product test |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll` | **Create** — 1D+1D element-wise fadd |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll` | **Create** — 1D+2D broadcast fadd |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll` | **Create** — 2D fsub |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll` | **Create** — 3D fmul |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll` | **Create** — 1D integer AND |

---

## Task 1: Rename contract intrinsic to include `<Rc>d` + update existing tests

The name `llvm.tensor.contract.<Ra>d.<Rb>d.<type>` is ambiguous — same (Ra,Rb) can produce different RankC values (e.g., batched GEMM 3d.3d has RankC=3, but a fully-independent 3d.3d contraction has RankC=4). Adding `<Rc>d` makes each name uniquely identify its parameter count.

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:156-157`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll:6-8`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll:5-8`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll:8-11`

- [ ] **Step 1: Update `getTensorContractFn()` to include RankC in name**

In `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`, find `getTensorContractFn()` at line ~150.
Replace lines 156-157:
```cpp
  std::string Name = (Twine("llvm.tensor.contract.") + Twine(RankA) + "d." +
                      Twine(RankB) + "d." + TypeSuffix).str();
```
With:
```cpp
  std::string Name = (Twine("llvm.tensor.contract.") + Twine(RankA) + "d." +
                      Twine(RankB) + "d." + Twine(RankC) + "d." +
                      TypeSuffix).str();
```

- [ ] **Step 2: Update GEMV test CHECK line**

In `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll`, replace:
```llvm
; CHECK: call void @llvm.tensor.contract.1d.2d.f32(
```
With:
```llvm
; CHECK: call void @llvm.tensor.contract.1d.2d.1d.f32(
```

- [ ] **Step 3: Update GEMM test CHECK line**

In `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll`, replace:
```llvm
; CHECK: call void @llvm.tensor.contract.2d.2d.f32(
```
With:
```llvm
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
```

- [ ] **Step 4: Update batched GEMM test CHECK line**

In `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll`, replace:
```llvm
; CHECK: call void @llvm.tensor.contract.3d.3d.f32(
```
With:
```llvm
; CHECK: call void @llvm.tensor.contract.3d.3d.3d.f32(
```

- [ ] **Step 5: Run existing tests — verify all pass**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll
git commit -m "tplan-lower: include output rank (Rc) in tensor.contract intrinsic name"
```

---

## Task 2: Fix RankC=0 — support 1D×1D dot product

For `acc += A[k] * B[k]`, OutputDimSet is empty (RankC=0). The existing guard `RankC < 1` rejects it. The fix is trivial: allow 0. The formula `3 + 4·RankC + 3` with Rc=0 naturally gives a 6-parameter signature — no stride arrays, just two pointers + contract strides + K.

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:376`

- [ ] **Step 1: Write the failing dot product test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; Dot product: acc += A[k] * B[k].
; A: DimSet={k} (1D), B: DimSet={k} (1D).
; OutputDimSet={} (empty), RankC=0.  Emits contract.1d.1d.0d.f32.
; Signature: void(ptr C, ptr A, i64 A_stride, ptr B, i64 B_stride, i64 K)
; — no stride arrays since RankC=0.
;
; CHECK: call void @llvm.tensor.contract.1d.1d.0d.f32(
; CHECK-NOT: i64 0{{.*}}i64 0
; CHECK-SAME: i64 1
; CHECK-SAME: i64 1

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @dot_product(ptr %A, ptr %B, ptr %acc) {
entry:
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %entry  ], [ %k.next, %k.loop ]
  %sum = phi float [ 0.0, %entry  ], [ %res,    %k.loop ]
  %aptr = getelementptr float, ptr %A, i64 %k
  %bptr = getelementptr float, ptr %B, i64 %k
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %prod = fmul float %av, %bv
  %res  = fadd float %sum, %prod
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %exit, label %k.loop

exit:
  store float %res, ptr %acc
  ret void
}
```

- [ ] **Step 2: Run test — verify it fails**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll
```
Expected: FAIL — the pass falls back to scalar (no `tensor.contract.1d.1d.0d` call emitted).

- [ ] **Step 3: Remove the `RankC < 1` guard in `emitContraction()`**

In `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`, find line ~376:
```cpp
  unsigned RankC = OutputDimSet.count();
  if (RankC < 1 || RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction output rank out of [1,4]\n");
    return nullptr;
  }
```
Replace with:
```cpp
  unsigned RankC = OutputDimSet.count();
  if (RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction output rank out of [0,4]\n");
    return nullptr;
  }
```

- [ ] **Step 4: Run test — verify it passes**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll
```
Expected: PASS — `@llvm.tensor.contract.1d.1d.0d.f32` is emitted with 6 args.

- [ ] **Step 5: Run all contract tests to check no regression**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-contract-dot.ll
git commit -m "tplan-lower: allow RankC=0 in emitContraction for 1D dot product"
```

---

## Task 3: Unify ElementWise + BroadcastBinary into BinaryOp kind

Currently the pattern matcher returns two separate kinds for binary ops. We unify them to `BinaryOp` so that a single `emitBinaryOp()` function handles both, using the stride=0 broadcast convention already established by the contract path.

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h:20-27`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp:131-163`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:260-270` (kindToStr)

- [ ] **Step 1: Add `BinaryOp` to `TensorOpKind` in TPlanTypes.h**

In `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h`, replace the enum body:
```cpp
enum class TensorOpKind {
  Scalar,           ///< DimSet empty — scalar op, no tensor parallelism
  ElementWise,      ///< Binary op, both operand DimSets equal
  BroadcastBinary,  ///< Binary op, one DimSet is strict subset of the other
  OuterProduct,     ///< Binary op, operand DimSets are disjoint
  Contraction,      ///< Reduction update of mul-like op sharing a reduction dim
  PlainReduction,   ///< Reduction update with no fuseable mul-like producer
};
```
With:
```cpp
enum class TensorOpKind {
  Scalar,           ///< DimSet empty — scalar op, no tensor parallelism
  ElementWise,      ///< Binary op, both operand DimSets equal (kept for legacy)
  BroadcastBinary,  ///< Binary op, one DimSet is strict subset (kept for legacy)
  BinaryOp,         ///< Binary op — unified element-wise + broadcast path
  OuterProduct,     ///< Binary op, operand DimSets are disjoint
  Contraction,      ///< Reduction update of mul-like op sharing a reduction dim
  PlainReduction,   ///< Reduction update with no fuseable mul-like producer
};
```

- [ ] **Step 2: Update `classifyBinaryOp()` to return `BinaryOp`**

In `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`, replace lines 143-150:
```cpp
  if (A == B)
    return TensorOpKind::ElementWise;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BroadcastBinary; // A ⊆ B
  if (Intersection == B) return TensorOpKind::BroadcastBinary; // B ⊆ A
```
With:
```cpp
  if (A == B)
    return TensorOpKind::BinaryOp;

  // Subset check.
  SmallBitVector Intersection = A;
  Intersection &= B;
  if (Intersection == A) return TensorOpKind::BinaryOp; // A ⊆ B (broadcast)
  if (Intersection == B) return TensorOpKind::BinaryOp; // B ⊆ A (broadcast)
```

- [ ] **Step 3: Add `BinaryOp` to `kindToStr()` in TPlanLowering.cpp**

In `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`, find `kindToStr()` at line ~260.
Add a case before `return "Unknown"`:
```cpp
  case TensorOpKind::ElementWise:     return "ElementWise";
  case TensorOpKind::BroadcastBinary: return "BroadcastBinary";
  case TensorOpKind::BinaryOp:        return "BinaryOp";
```

- [ ] **Step 4: Run all existing tests — verify no regression**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/
```
Expected: all PASS (ElementWise/BroadcastBinary dispatch still exists in execute(), BinaryOp just isn't dispatched yet — those recipes fall through to scalar clone, same as before this task).

- [ ] **Step 5: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h \
        llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan: unify ElementWise+BroadcastBinary into BinaryOp kind"
```

---

## Task 4: Write failing binary op tests

Write all 5 lit tests before implementing `emitBinaryOp()`. They should fail (fall back to scalar) until Task 5 wires up the dispatch.

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll`

- [ ] **Step 1: Write 1D+1D fadd test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D + 1D element-wise fadd: C[i] = A[i] + B[i].
; A: DimSet={i}, B: DimSet={i}.  OutputDimSet={i}, RankC=1.
; Emits binary.fadd.1d.1d.1d.f32.
;
; CHECK: call void @llvm.tensor.binary.fadd.1d.1d.1d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @add_1d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.loop ]
  %aptr = getelementptr float, ptr %A, i64 %i
  %bptr = getelementptr float, ptr %B, i64 %i
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fadd float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %i
  store float %res, ptr %cptr
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
```

- [ ] **Step 2: Write 1D+2D broadcast fadd test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D vector + 2D matrix broadcast fadd: C[i*N+j] = A[j] + B[i*N+j].
; A: DimSet={j} (1D), B: DimSet={i,j} (2D).
; OutputDimSet={i,j}, RankC=2.  A_strides[i]=0 (broadcast).
; Emits binary.fadd.1d.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.binary.fadd.1d.2d.2d.f32(
; CHECK-SAME: i64 0

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @add_broadcast(ptr %A, ptr %B, ptr %C, i64 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %aptr = getelementptr float, ptr %A, i64 %j
  %bi   = mul i64 %i, %N
  %bij  = add i64 %bi, %j
  %bptr = getelementptr float, ptr %B, i64 %bij
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fadd float %av, %bv
  %cij  = add i64 %bi, %j
  %cptr = getelementptr float, ptr %C, i64 %cij
  store float %res, ptr %cptr
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

- [ ] **Step 3: Write 2D−2D fsub test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 2D element-wise fsub: C[i*N+j] = A[i*N+j] - B[i*N+j].
; A: DimSet={i,j}, B: DimSet={i,j}.  OutputDimSet={i,j}, RankC=2.
; Emits binary.fsub.2d.2d.2d.f32.
;
; CHECK: call void @llvm.tensor.binary.fsub.2d.2d.2d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @sub_2d(ptr %A, ptr %B, ptr %C, i64 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %ij   = mul i64 %i, %N
  %idx  = add i64 %ij, %j
  %aptr = getelementptr float, ptr %A, i64 %idx
  %bptr = getelementptr float, ptr %B, i64 %idx
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fsub float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %idx
  store float %res, ptr %cptr
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

- [ ] **Step 4: Write 3D×3D fmul test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3D element-wise fmul: C[b*IN+i*N+j] = A[b*IN+i*N+j] * B[b*IN+i*N+j].
; A,B,C: DimSet={b,i,j}.  OutputDimSet={b,i,j}, RankC=3.
; Emits binary.fmul.3d.3d.3d.f32.
;
; CHECK: call void @llvm.tensor.binary.fmul.3d.3d.3d.f32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @mul_3d(ptr %A, ptr %B, ptr %C, i64 %IN, i64 %N) {
entry:
  br label %b.loop
b.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %i.latch ]
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %b.loop ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %j.loop ]
  %bi   = mul i64 %b, %IN
  %ii   = mul i64 %i, %N
  %idx  = add i64 %bi, %ii
  %idx2 = add i64 %idx, %j
  %aptr = getelementptr float, ptr %A, i64 %idx2
  %bptr = getelementptr float, ptr %B, i64 %idx2
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %res  = fmul float %av, %bv
  %cptr = getelementptr float, ptr %C, i64 %idx2
  store float %res, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %i.latch, label %i.loop
i.latch:
  %b.next = add i64 %b, 1
  %b.done = icmp eq i64 %b.next, 16
  br i1 %b.done, label %exit, label %b.loop
exit:
  ret void
}
```

- [ ] **Step 5: Write 1D integer AND test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll`:
```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 1D integer AND: C[i] = A[i] & B[i].
; A: DimSet={i} (i32), B: DimSet={i} (i32).  OutputDimSet={i}, RankC=1.
; Emits binary.and.1d.1d.1d.i32.
;
; CHECK: call void @llvm.tensor.binary.and.1d.1d.1d.i32(

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @and_1d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %i.loop ]
  %aptr = getelementptr i32, ptr %A, i64 %i
  %bptr = getelementptr i32, ptr %B, i64 %i
  %av   = load i32, ptr %aptr
  %bv   = load i32, ptr %bptr
  %res  = and i32 %av, %bv
  %cptr = getelementptr i32, ptr %C, i64 %i
  store i32 %res, ptr %cptr
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
```

- [ ] **Step 6: Run all new tests — verify they all fail**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll
```
Expected: all FAIL — `llvm.tensor.binary.*` not yet emitted.

- [ ] **Step 7: Commit the failing tests**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll
git commit -m "test(tplan): add failing tests for tensor.binary intrinsic emission"
```

---

## Task 5: Implement `getTensorBinaryFn()` + `emitBinaryOp()` + wire dispatch

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

- [ ] **Step 1: Add `getTensorBinaryFn()` after `getTensorReduceFn()`**

In `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`, insert after line ~254 (after `getTensorReduceFn()`):

```cpp
/// Returns (creating if needed) @llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>.
/// Signature:
///   void(ptr C, i64×RankC C_strides,
///        ptr A, i64×RankC A_strides,   ; 0 if A ∌ that dim (broadcast)
///        ptr B, i64×RankC B_strides,   ; 0 if B ∌ that dim (broadcast)
///        i64×RankC output_dims)
/// RankC = |(A.DimSet ∪ B.DimSet)|  (no contraction dim removed).
static FunctionCallee getTensorBinaryFn(Module &M, StringRef Op,
                                         unsigned RankA, unsigned RankB,
                                         unsigned RankC, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type for binary");
  std::string Name = (Twine("llvm.tensor.binary.") + Op + "." +
                      Twine(RankA) + "d." + Twine(RankB) + "d." +
                      Twine(RankC) + "d." + TypeSuffix).str();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  Params.push_back(PtrTy);                                          // C
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // C strides
  Params.push_back(PtrTy);                                          // A
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // A strides
  Params.push_back(PtrTy);                                          // B
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // B strides
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);   // output dims
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params,
                                        /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}
```

- [ ] **Step 2: Add `emitBinaryOp()` after `emitContraction()` (~line 472)**

Returns `true` on success, `false` to trigger scalar fallback (same convention as
the `tryVectorize` lambdas in `ElementWise`/`BroadcastBinary`).

```cpp
//===----------------------------------------------------------------------===//
// Helper: emit @llvm.tensor.binary for a BinaryOp recipe
//===----------------------------------------------------------------------===//

/// Returns true if the tensor.binary intrinsic was emitted; false triggers
/// scalar-clone fallback in the caller.
static bool emitBinaryOp(const TPWidenRecipe *WR,
                          TPTransformState &State) {
  auto *ADR = dyn_cast<TPSingleDefRecipe>(WR->getOperand(0));
  auto *BDR = dyn_cast<TPSingleDefRecipe>(WR->getOperand(1));
  if (!ADR || !BDR) return false;

  unsigned RankA = ADR->DimSet.count();
  unsigned RankB = BDR->DimSet.count();
  if (RankA < 1 || RankA > 4 || RankB < 1 || RankB > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: BinaryOp rank out of [1,4]\n");
    return false;
  }

  Instruction *Inst = WR->getInstruction();

  std::string OpName = getOpcodeStr(Inst);
  if (OpName.empty()) return false;

  // For CmpInst the result type is i1 — use operand type for the suffix.
  Type *ElemTy = Inst->getType()->getScalarType();
  if (isa<CmpInst>(Inst) && Inst->getNumOperands() >= 1)
    ElemTy = Inst->getOperand(0)->getType()->getScalarType();
  if (getTypeSuffix(ElemTy).empty()) return false;

  auto *ALoad = dyn_cast<TPWidenLoadRecipe>(ADR);
  auto *BLoad = dyn_cast<TPWidenLoadRecipe>(BDR);
  if (!ALoad || !BLoad) return false;

  auto *APtrDR = dyn_cast<TPSingleDefRecipe>(ALoad->getOperand(0));
  auto *BPtrDR = dyn_cast<TPSingleDefRecipe>(BLoad->getOperand(0));
  if (!APtrDR || !BPtrDR) return false;

  Value *APtr = State.getValue(APtrDR);
  Value *BPtr = State.getValue(BPtrDR);
  if (!APtr || !BPtr) return false;

  // Build OutputDimSet = A.DimSet ∪ B.DimSet (no contraction dim removed).
  unsigned NBits = std::max(ADR->DimSet.size(), BDR->DimSet.size());
  SmallBitVector ABits = ADR->DimSet, BBits = BDR->DimSet;
  ABits.resize(NBits); BBits.resize(NBits);
  SmallBitVector OutputDimSet = ABits;
  OutputDimSet |= BBits;
  unsigned RankC = OutputDimSet.count();
  if (RankC < 1 || RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: BinaryOp output rank out of [1,4]\n");
    return false;
  }

  // Locate C pointer from the store recipe that uses this recipe's result.
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = WR->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        if (auto *PD = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0)))
          CPtr = State.getValue(PD);
        break;
      }
    }
  }
  if (!CPtr) return false;

  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };
  // Returns stride for output dim D in operand DR; 0 if DR doesn't span it.
  auto getOperandStride = [&](const TPSingleDefRecipe *DR,
                               unsigned D) -> Value * {
    if (D >= DR->DimSet.size() || !DR->DimSet.test(D))
      return I64(0);
    return expandStride(DR->getMemStride(D, State.Plan, *State.SE), D);
  };

  // Build stride/dim vectors in output-dim order.
  SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    if (CStoreRecipe && State.SE)
      CStrides.push_back(expandStride(
          CStoreRecipe->getMemStride(UD, State.Plan, *State.SE), UD));
    else
      CStrides.push_back(I64(State.Plan.getDenseStrideForDim(UD)));
    AStrides.push_back(getOperandStride(ADR, UD));
    BStrides.push_back(getOperandStride(BDR, UD));
    OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee BinFn =
      getTensorBinaryFn(*Mod, StringRef(OpName), RankA, RankB, RankC, ElemTy);

  SmallVector<Value *> Args;
  Args.push_back(CPtr);
  Args.append(CStrides.begin(), CStrides.end());
  Args.push_back(APtr);
  Args.append(AStrides.begin(), AStrides.end());
  Args.push_back(BPtr);
  Args.append(BStrides.begin(), BStrides.end());
  Args.append(OutDims.begin(), OutDims.end());
  B.CreateCall(BinFn, Args);
  return true;
}
```

- [ ] **Step 3: Add `BinaryOp` dispatch case to `TPWidenRecipe::execute()`**

In `TPlanLowering.cpp`, find `TPWidenRecipe::execute()` at line ~557. Add a new case in the switch statement, after the `Contraction` case and before `ElementWise`. If `emitBinaryOp` returns false, fall through to the scalar clone path:

```cpp
  case TensorOpKind::BinaryOp: {
    if (emitBinaryOp(this, State)) return;
    // Scalar fallback — same as default below.
    auto *Clone = Inst->clone();
    State.remapClone(Clone);
    Value *Result = State.Builder.Insert(Clone);
    applyFlags(*cast<Instruction>(Result));
    State.EmittedMap[Inst] = Result;
    State.setValue(this, Result);
    return;
  }
```

- [ ] **Step 4: Run all binary tests — verify they all pass**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-1d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-add-broadcast.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-sub-2d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-mul-3d.ll \
            llvm/test/Transforms/LoopTensorize/basic/tensor-binary-and-1d.ll
```
Expected: all PASS.

- [ ] **Step 5: Run full test suite — verify no regression**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: add getTensorBinaryFn and emitBinaryOp for N-D binary ops"
```

---

## Task 6: Register intrinsics in `Intrinsics.td`

Create `IntrinsicsTPlan.td` and wire it into the main `Intrinsics.td`. This makes tensor intrinsics first-class LLVM intrinsics with proper memory attribute annotations.

**Files:**
- Create: `llvm/include/llvm/IR/IntrinsicsTPlan.td`
- Modify: `llvm/include/llvm/IR/Intrinsics.td:2778` (before hardware loop section)

- [ ] **Step 1: Create `IntrinsicsTPlan.td`**

Create `llvm/include/llvm/IR/IntrinsicsTPlan.td`:

```tablegen
//===- IntrinsicsTPlan.td - TPlan tensor intrinsics --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Declares the tensor intrinsic families emitted by TPlanLowering:
///
///   llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>
///   llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>
///
/// Ra, Rb ∈ {1,2,3,4}   Rc ∈ {0,1,2,3,4} (contract) or {1,2,3,4} (binary)
/// type  ∈ {f16, f32, f64, i8, i16, i32, i64}
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>
//
// void(ptr C, i64×Rc C_strides,
//      ptr A, i64×Rc A_strides, i64 A_contract_stride,
//      ptr B, i64×Rc B_strides, i64 B_contract_stride,
//      i64 K, i64×Rc output_dims)
//
// stride=0 means the operand does not span that output dim (broadcast).
// Rc=0 is the dot-product case — no stride arrays, 6 total parameters.
//===----------------------------------------------------------------------===//

multiclass TensorContractByType<int Ra, int Rb, int Rc> {
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_f16 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_f32 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_f64 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_i8 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_i16 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_i32 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_contract_#Ra#d_#Rb#d_#Rc#d_i64 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc), [llvm_i64_ty],
                [llvm_i64_ty], !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
}

foreach Ra = [1, 2, 3, 4] in {
  foreach Rb = [1, 2, 3, 4] in {
    foreach Rc = [0, 1, 2, 3, 4] in {
      defm : TensorContractByType<Ra, Rb, Rc>;
    }
  }
}

//===----------------------------------------------------------------------===//
// llvm.tensor.binary.<op>.<Ra>d.<Rb>d.<Rc>d.<type>
//
// void(ptr C, i64×Rc C_strides,
//      ptr A, i64×Rc A_strides,   ; 0 if A ∌ that dim (broadcast)
//      ptr B, i64×Rc B_strides,   ; 0 if B ∌ that dim (broadcast)
//      i64×Rc output_dims)
//
// Rc = |(A.DimSet ∪ B.DimSet)| — no contraction dim removed.
// Ops: fadd fsub fmul fdiv add sub mul sdiv udiv and or xor shl lshr ashr
//===----------------------------------------------------------------------===//

multiclass TensorBinaryForOp<string Op, int Ra, int Rb, int Rc> {
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_f16 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_f32 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_f64 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_i8 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_i16 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_i32 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
  def int_tensor_binary_#Op#_#Ra#d_#Rb#d_#Rc#d_i64 : Intrinsic<[],
    !listconcat([llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                [llvm_ptr_ty], !listsplat(llvm_i64_ty, Rc),
                !listsplat(llvm_i64_ty, Rc)),
    [IntrArgMemOnly, IntrWillReturn]>;
}

foreach Op = ["fadd", "fsub", "fmul", "fdiv", "frem",
              "add",  "sub",  "mul",  "sdiv", "udiv",
              "and",  "or",   "xor",  "shl",  "lshr", "ashr"] in {
  foreach Ra = [1, 2, 3, 4] in {
    foreach Rb = [1, 2, 3, 4] in {
      foreach Rc = [1, 2, 3, 4] in {
        defm : TensorBinaryForOp<Op, Ra, Rb, Rc>;
      }
    }
  }
}
```

- [ ] **Step 2: Include `IntrinsicsTPlan.td` from `Intrinsics.td`**

In `llvm/include/llvm/IR/Intrinsics.td`, find line ~2778 (just before the hardware loop section):
```tablegen
//===---------- Intrinsics to control hardware supported loops ----------===//
```
Insert before it:
```tablegen
//===---------- TPlan tensor intrinsics ---------------------------------===//

include "llvm/IR/IntrinsicsTPlan.td"

```

- [ ] **Step 3: Build LLVM to verify TableGen compiles**

```bash
ninja -C build llvm-tblgen
```
Expected: build succeeds with no TableGen errors.

- [ ] **Step 4: Build opt to verify integration**

```bash
ninja -C build opt
```
Expected: build succeeds.

- [ ] **Step 5: Run all tensor tests — verify nothing broken**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add llvm/include/llvm/IR/IntrinsicsTPlan.td \
        llvm/include/llvm/IR/Intrinsics.td
git commit -m "intrinsics: register tensor.contract and tensor.binary families in IntrinsicsTPlan.td"
```
