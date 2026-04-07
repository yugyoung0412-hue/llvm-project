# TPlanAnalysis: decomposePtrForDims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `decomposePtrForDims()` in a new `TPlanAnalysis` layer that generalizes GEP-chain pointer decomposition into {base pointer + per-dim affine strides + non-affine dims}, and wire it into `emitContraction()` in `TPlanLowering.cpp`.

**Architecture:** A new file pair (`TPlanAnalysis.h` / `TPlanAnalysis.cpp`) exposes `PtrDecomposition` and `decomposePtrForDims()`. The function walks a GEP/bitcast/addrspacecast/PHI chain upward, extracting SCEV-based strides for affine dims and stopping at non-affine instructions (srem, udiv, etc.) whose result becomes the loop-invariant base pointer. `emitContraction()` replaces its ad-hoc `State.getValue(PtrDR)` + `getMemStride()` calls with a single `decomposePtrForDims()` call, using only `AffineDims` for the intrinsic rank and `Base` as the tensor pointer. Lit tests cover bitcast, addrspacecast, PHI, and srem cases.

**Tech Stack:** LLVM C++17, ScalarEvolution, LoopInfo, SCEV AddRecExpr, `llvm/test/Transforms/LoopTensorize/` lit tests, FileCheck.

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `llvm/include/llvm/Transforms/Vectorize/TPlanAnalysis.h` | `PtrDecomposition` struct + `decomposePtrForDims()` declaration |
| Create | `llvm/lib/Transforms/Vectorize/TPlanAnalysis.cpp` | Full implementation: chain walker, SCEV stride extractor |
| Modify | `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | Add `TPlanAnalysis.cpp` to build |
| Modify | `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Replace ad-hoc ptr/stride lookup with `decomposePtrForDims()` in `emitContraction()` |
| Create | `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll` | bitcast skip test |
| Create | `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-addrspacecast.ll` | addrspacecast skip test |
| Create | `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll` | loop-invariant PHI test |
| Create | `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll` | srem non-affine base test |

---

## Task 1: Define `PtrDecomposition` struct and declare `decomposePtrForDims()`

**Files:**
- Create: `llvm/include/llvm/Transforms/Vectorize/TPlanAnalysis.h`

- [ ] **Step 1: Create the header**

```cpp
//===- TPlanAnalysis.h - Pointer decomposition for TPlan codegen ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Declares decomposePtrForDims(): splits a GEP/bitcast/addrspacecast/PHI
/// pointer chain into a loop-invariant base pointer and per-dimension affine
/// byte strides.  Non-affine GEP steps (e.g. srem-based batch broadcasting)
/// stop the walk; the GEP result at that point becomes the base pointer.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H
#define LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

namespace llvm {

class Loop;
class SCEV;
class ScalarEvolution;
class Value;

/// Result of decomposePtrForDims().
struct PtrDecomposition {
  /// Innermost loop-invariant base pointer after absorbing all non-affine GEPs.
  /// nullptr if the chain could not be walked at all.
  Value *Base = nullptr;

  /// dim → byte stride SCEV for each affine dimension.
  /// Only dims whose GEP index produced a valid SCEVAddRecExpr are present.
  DenseMap<unsigned, const SCEV *> Strides;

  /// Dims for which the GEP index was non-affine (SCEVCouldNotCompute).
  /// The outer loops for these dims iterate scalar; their effect is already
  /// baked into Base.
  SmallBitVector NonAffineDims;

  /// Dims for which stride extraction succeeded.
  SmallBitVector AffineDims;
};

/// Decompose \p Ptr into a base pointer and per-dimension affine byte strides
/// by walking the GEP / bitcast / addrspacecast / loop-invariant-PHI chain.
///
/// \p DimSet        - set of TPlan dimension indices to analyse.
/// \p DimToLoop     - maps each dim index to its corresponding Loop*.
/// \p OutermostGEMMLoop - the outermost loop of the GEMM region; used to
///                    decide whether a PHI incoming value is loop-invariant.
/// \p SE            - ScalarEvolution for SCEV queries.
///
/// Walk rules (applied at each step):
///   bitcast / addrspacecast → transparent skip (pointer identity preserved).
///   PHI node              → follow the incoming value that is loop-invariant
///                           w.r.t. OutermostGEMMLoop; stop if none found.
///   single-index GEP      → call SE.getSCEV(index):
///     SCEVAddRecExpr       → record stride, continue upward.
///     SCEVCouldNotCompute  → mark dim NonAffine, set Base = current ptr, stop.
///     loop-invariant SCEV  → dim not in DimSet at this level; continue upward.
///   multi-index GEP / other → stop.
PtrDecomposition decomposePtrForDims(
    Value *Ptr,
    const SmallBitVector &DimSet,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    Loop *OutermostGEMMLoop,
    ScalarEvolution &SE);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TPLANANALYSIS_H
```

- [ ] **Step 2: Verify header compiles standalone (no .cpp yet)**

```bash
cd /path/to/llvm && ninja -C build check-llvm-transforms-vectorize 2>&1 | grep -i "tplananalysis\|error" | head -20
```
Expected: no errors about TPlanAnalysis.h (it's not included anywhere yet).

- [ ] **Step 3: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlanAnalysis.h
git commit -m "tplan-analysis: add PtrDecomposition struct and decomposePtrForDims() declaration"
```

---

## Task 2: Write lit tests (failing) for all four chain cases

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-addrspacecast.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll`

- [ ] **Step 1: Write bitcast test**

This test verifies that a `bitcast` between the load and the GEP is transparently skipped and the k-dim stride is still extracted.

```llvm
; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll
; RUN: opt -passes=loop-tensorize -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; A 2D GEMM where the load pointer has a bitcast between the GEP and the load.
; decomposePtrForDims must skip the bitcast and still extract stride %nb0.
;
; CHECK: Contraction (contractDim=0)

define void @gemm_bitcast(ptr %A, ptr %B, ptr %C,
                           i64 %nb0_A, i64 %nb1_A,
                           i64 %nb0_B, i64 %nb1_B,
                           i64 %nb0_C, i64 %nb1_C) {
entry:
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  %j.off.B = mul i64 %j, %nb1_B
  %j.off.C = mul i64 %j, %nb1_C
  %B.j = getelementptr inbounds i8, ptr %B, i64 %j.off.B
  %C.j = getelementptr inbounds i8, ptr %C, i64 %j.off.C
  br label %k.loop

k.loop:
  %k = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  ; A[i][k]: GEP then bitcast
  %i.off.A = mul i64 %i, %nb1_A
  %k.off.A = mul i64 %k, %nb0_A
  %A.i = getelementptr inbounds i8, ptr %A, i64 %i.off.A
  %A.ik.i8 = getelementptr inbounds i8, ptr %A.i, i64 %k.off.A
  %A.ik = bitcast ptr %A.ik.i8 to ptr   ; ← bitcast to skip

  ; B[k][j]
  %k.off.B = mul i64 %k, %nb0_B
  %B.kj.i8 = getelementptr inbounds i8, ptr %B.j, i64 %k.off.B
  %B.kj = bitcast ptr %B.kj.i8 to ptr

  %a.val = load float, ptr %A.ik, align 4
  %b.val = load float, ptr %B.kj, align 4
  %mul   = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  ; C[i][j]
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C2 = mul i64 %j, %nb0_C
  %C.ij.i8 = getelementptr inbounds i8, ptr %C, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.ij.i8, i64 %j.off.C2
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %exit, label %i.loop

exit:
  ret void
}
```

- [ ] **Step 2: Write addrspacecast test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-addrspacecast.ll
; RUN: opt -passes=loop-tensorize -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; A 2D GEMM with addrspacecast (simulating GPU global→generic pointer cast).
; decomposePtrForDims must skip addrspacecast and extract k-dim stride.
;
; CHECK: Contraction (contractDim=0)

define void @gemm_addrspacecast(ptr addrspace(1) %A_global,
                                 ptr addrspace(1) %B_global,
                                 ptr %C,
                                 i64 %nb0_A, i64 %nb1_A,
                                 i64 %nb0_B, i64 %nb1_B,
                                 i64 %nb0_C, i64 %nb1_C) {
entry:
  ; Cast global pointers to generic address space
  %A = addrspacecast ptr addrspace(1) %A_global to ptr
  %B = addrspacecast ptr addrspace(1) %B_global to ptr
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  %j.off.B = mul i64 %j, %nb1_B
  %B.j = getelementptr inbounds i8, ptr %B, i64 %j.off.B
  br label %k.loop

k.loop:
  %k = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  %i.off.A = mul i64 %i, %nb1_A
  %A.i = getelementptr inbounds i8, ptr %A, i64 %i.off.A
  %k.off.A = mul i64 %k, %nb0_A
  ; addrspacecast inside the chain
  %A.i.as1 = addrspacecast ptr %A.i to ptr addrspace(1)
  %A.i.generic = addrspacecast ptr addrspace(1) %A.i.as1 to ptr
  %A.ik = getelementptr inbounds i8, ptr %A.i.generic, i64 %k.off.A

  %k.off.B = mul i64 %k, %nb0_B
  %B.kj = getelementptr inbounds i8, ptr %B.j, i64 %k.off.B

  %a.val = load float, ptr %A.ik, align 4
  %b.val = load float, ptr %B.kj, align 4
  %mul   = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C = mul i64 %j, %nb0_C
  %C.ij.i8 = getelementptr inbounds i8, ptr %C, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.ij.i8, i64 %j.off.C
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %exit, label %i.loop

exit:
  ret void
}
```

- [ ] **Step 3: Write loop-invariant PHI test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll
; RUN: opt -passes=loop-tensorize -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Batched GEMM where the outer batch loop uses a pointer-induction PHI
; (A_slice advances by batch_stride each iteration).
; decomposePtrForDims must follow the loop-invariant incoming value of
; the PHI (the preheader value) and extract i/k strides from inner GEPs.
;
; CHECK: Contraction (contractDim=0)

define void @gemm_phi_ptr(ptr %A_base, ptr %B_base, ptr %C_base,
                           i64 %batch_stride_A,
                           i64 %nb1_A, i64 %nb0_A,
                           i64 %nb1_B, i64 %nb0_B,
                           i64 %nb1_C, i64 %nb0_C) {
entry:
  br label %batch.loop

batch.loop:
  ; Pointer-induction PHI: A_slice = A_base + batch*batch_stride_A
  %A_slice = phi ptr [ %A_base, %entry ], [ %A_slice_next, %batch.latch ]
  %B_slice = phi ptr [ %B_base, %entry ], [ %B_slice_next, %batch.latch ]
  %C_slice = phi ptr [ %C_base, %entry ], [ %C_slice_next, %batch.latch ]
  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %batch.loop ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  %i.off = mul i64 %i, %nb1_A
  %k.off = mul i64 %k, %nb0_A
  ; GEP from the PHI pointer — decomposePtrForDims must follow PHI
  %A.i  = getelementptr inbounds i8, ptr %A_slice, i64 %i.off
  %A.ik = getelementptr inbounds i8, ptr %A.i,    i64 %k.off

  %j.off.B = mul i64 %j, %nb1_B
  %k.off.B = mul i64 %k, %nb0_B
  %B.j  = getelementptr inbounds i8, ptr %B_slice, i64 %j.off.B
  %B.kj = getelementptr inbounds i8, ptr %B.j,    i64 %k.off.B

  %a.val    = load float, ptr %A.ik, align 4
  %b.val    = load float, ptr %B.kj, align 4
  %mul      = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C = mul i64 %j, %nb0_C
  %C.i  = getelementptr inbounds i8, ptr %C_slice, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.i,    i64 %j.off.C
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %batch.latch, label %i.loop

batch.latch:
  %A_slice_next = getelementptr inbounds i8, ptr %A_slice, i64 %batch_stride_A
  %B_slice_next = getelementptr inbounds i8, ptr %B_slice, i64 %batch_stride_A
  %C_slice_next = getelementptr inbounds i8, ptr %C_slice, i64 %batch_stride_A
  %batch.next = add nuw i64 0, 1
  %batch.cond = icmp eq i64 %batch.next, 4
  br i1 %batch.cond, label %exit, label %batch.loop

exit:
  ret void
}
```

- [ ] **Step 4: Write srem non-affine base test**

```llvm
; llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll
; RUN: opt -passes=loop-tensorize -debug-only=tplan-lower -S < %s 2>&1 \
; RUN:   | FileCheck %s
; REQUIRES: asserts
;
; Batched GEMM where A uses srem-based broadcasting (A has fewer batches than
; B/C, so A_batch_idx = output_batch % A_ne_batch).
; decomposePtrForDims must stop at the srem GEP, set Base = that GEP result,
; and still extract i/k strides from the inner affine GEPs.
; The resulting intrinsic should be 2D (i, k only) with the srem-computed
; slice pointer as A's base.
;
; CHECK: Contraction (contractDim=0)
; CHECK-NOT: TPlanLowering: Contraction cannot find C pointer

define void @gemm_srem_broadcast(
    ptr %A_base, ptr %B_base, ptr %C_base,
    i64 %A_ne_batch,
    i64 %nb_batch_A, i64 %nb1_A, i64 %nb0_A,
    i64 %nb_batch_B, i64 %nb1_B, i64 %nb0_B,
    i64 %nb_batch_C, i64 %nb1_C, i64 %nb0_C) {
entry:
  br label %batch.loop

batch.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %batch.latch ]

  ; A: broadcast via srem
  %a.batch.idx = srem i64 %b, %A_ne_batch
  %a.batch.off = mul i64 %a.batch.idx, %nb_batch_A
  %A_slice = getelementptr inbounds i8, ptr %A_base, i64 %a.batch.off

  ; B, C: direct indexing
  %b.batch.off = mul i64 %b, %nb_batch_B
  %B_slice = getelementptr inbounds i8, ptr %B_base, i64 %b.batch.off
  %c.batch.off = mul i64 %b, %nb_batch_C
  %C_slice = getelementptr inbounds i8, ptr %C_base, i64 %c.batch.off

  br label %i.loop

i.loop:
  %i = phi i64 [ 0, %batch.loop ], [ %i.next, %j.latch ]
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %acc.next, %k.loop ]

  %i.off.A = mul i64 %i, %nb1_A
  %k.off.A = mul i64 %k, %nb0_A
  %A.i  = getelementptr inbounds i8, ptr %A_slice, i64 %i.off.A
  %A.ik = getelementptr inbounds i8, ptr %A.i,    i64 %k.off.A

  %j.off.B = mul i64 %j, %nb1_B
  %k.off.B = mul i64 %k, %nb0_B
  %B.j  = getelementptr inbounds i8, ptr %B_slice, i64 %j.off.B
  %B.kj = getelementptr inbounds i8, ptr %B.j,    i64 %k.off.B

  %a.val    = load float, ptr %A.ik, align 4
  %b.val    = load float, ptr %B.kj, align 4
  %mul      = fmul float %a.val, %b.val
  %acc.next = fadd float %acc, %mul

  %k.next = add nuw i64 %k, 1
  %k.cond = icmp eq i64 %k.next, 64
  br i1 %k.cond, label %k.latch, label %k.loop

k.latch:
  %i.off.C = mul i64 %i, %nb1_C
  %j.off.C = mul i64 %j, %nb0_C
  %C.i  = getelementptr inbounds i8, ptr %C_slice, i64 %i.off.C
  %C.ij = getelementptr inbounds i8, ptr %C.i,    i64 %j.off.C
  store float %acc.next, ptr %C.ij, align 4
  %j.next = add nuw i64 %j, 1
  %j.cond = icmp eq i64 %j.next, 64
  br i1 %j.cond, label %j.latch, label %j.loop

j.latch:
  %i.next = add nuw i64 %i, 1
  %i.cond = icmp eq i64 %i.next, 64
  br i1 %i.cond, label %batch.latch, label %i.loop

batch.latch:
  %b.next = add nuw i64 %b, 1
  %b.cond = icmp eq i64 %b.next, 8
  br i1 %b.cond, label %exit, label %batch.loop

exit:
  ret void
}
```

- [ ] **Step 5: Run tests to confirm they currently fail (expected)**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-addrspacecast.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll 2>&1 | tail -20
```
Expected: FAIL on Contraction/srem checks (implementation not yet wired in).

- [ ] **Step 6: Commit tests**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-*.ll
git commit -m "tplan-analysis: add failing lit tests for decomposePtrForDims (bitcast/addrspacecast/phi/srem)"
```

---

## Task 3: Implement `TPlanAnalysis.cpp`

**Files:**
- Create: `llvm/lib/Transforms/Vectorize/TPlanAnalysis.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/CMakeLists.txt`

- [ ] **Step 1: Create implementation**

```cpp
//===- TPlanAnalysis.cpp - Pointer decomposition for TPlan codegen --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/TPlanAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

using namespace llvm;

/// Strip a single layer of bitcast or addrspacecast, returning the operand.
/// Returns nullptr if \p V is not a cast instruction of these kinds.
static Value *stripPointerCast(Value *V) {
  if (auto *BC = dyn_cast<BitCastInst>(V))
    return BC->getOperand(0);
  if (auto *AC = dyn_cast<AddrSpaceCastInst>(V))
    return AC->getOperand(0);
  // Handle constant expressions (e.g. bitcast in globals).
  if (auto *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast ||
        CE->getOpcode() == Instruction::AddrSpaceCast)
      return CE->getOperand(0);
  }
  return nullptr;
}

/// Given a PHINode that is a pointer, return the incoming value that is
/// loop-invariant with respect to \p OutermostLoop.  Returns nullptr if no
/// such value exists or if more than one invariant incoming exists (ambiguous).
static Value *getLoopInvariantIncoming(PHINode *Phi, Loop *OutermostLoop,
                                        ScalarEvolution &SE) {
  if (!Phi || !OutermostLoop)
    return nullptr;
  Value *Candidate = nullptr;
  for (unsigned I = 0, E = Phi->getNumIncomingValues(); I < E; ++I) {
    Value *In = Phi->getIncomingValue(I);
    // A value is loop-invariant w.r.t. OutermostLoop if its SCEV does not
    // contain an AddRec for OutermostLoop or any loop nested inside it.
    const SCEV *S = SE.getSCEV(In);
    if (isa<SCEVCouldNotCompute>(S))
      continue;
    // Check whether S contains any AddRec whose loop is OutermostLoop or
    // a descendant of it.
    bool HasAddRec = false;
    struct Checker {
      Loop *L;
      bool Found = false;
      bool follow(const SCEV *S) {
        if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S))
          if (L->contains(AR->getLoop())) { Found = true; return false; }
        return !Found;
      }
      bool isDone() const { return Found; }
    } C{OutermostLoop};
    SCEVTraversal<Checker> T(C);
    T.visitAll(S);
    HasAddRec = C.Found;
    if (!HasAddRec) {
      if (Candidate)
        return nullptr; // More than one invariant incoming — ambiguous.
      Candidate = In;
    }
  }
  return Candidate;
}

PtrDecomposition llvm::decomposePtrForDims(
    Value *Ptr,
    const SmallBitVector &DimSet,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    Loop *OutermostGEMMLoop,
    ScalarEvolution &SE) {

  PtrDecomposition Result;
  Result.NonAffineDims.resize(DimSet.size());
  Result.AffineDims.resize(DimSet.size());

  // Build reverse map: Loop* → dim index, for the dims we care about.
  DenseMap<const Loop *, unsigned> LoopToDim;
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    auto It = DimToLoop.find(UD);
    if (It != DimToLoop.end())
      LoopToDim[It->second] = UD;
  }

  Value *Cur = Ptr;
  // Bound the walk to avoid pathological IR.
  unsigned MaxSteps = 64;

  while (Cur && MaxSteps-- > 0) {
    // 1. Transparent: strip bitcast / addrspacecast.
    if (Value *Stripped = stripPointerCast(Cur)) {
      Cur = Stripped;
      continue;
    }

    // 2. Transparent: follow loop-invariant PHI incoming.
    if (auto *Phi = dyn_cast<PHINode>(Cur)) {
      Value *Inv = getLoopInvariantIncoming(Phi, OutermostGEMMLoop, SE);
      if (Inv) {
        Cur = Inv;
        continue;
      }
      // No invariant incoming found — treat current ptr as base.
      break;
    }

    // 3. Single-index GEP: attempt stride extraction.
    auto *GEP = dyn_cast<GetElementPtrInst>(Cur);
    if (!GEP || GEP->getNumIndices() != 1) {
      // Multi-index GEP or non-GEP non-cast non-PHI — stop.
      break;
    }

    Value *Idx = GEP->getOperand(1);
    const SCEV *IdxSCEV = SE.getSCEV(Idx);

    if (isa<SCEVCouldNotCompute>(IdxSCEV)) {
      // Non-affine index (e.g. srem): this GEP result is the base.
      // Mark any dim whose loop this GEP's block is in as non-affine.
      // (We don't know exactly which dim caused it, so we leave
      // NonAffineDims empty here; the caller infers it from AffineDims.)
      Result.Base = Cur;
      return Result;
    }

    // Walk all AddRec nodes in IdxSCEV (handles both nested chains and
    // SCEVAdd-of-AddRecs via a worklist).
    SmallVector<const SCEV *, 8> Worklist;
    Worklist.push_back(IdxSCEV);
    bool FoundAffine = false;
    while (!Worklist.empty()) {
      const SCEV *S = Worklist.pop_back_val();
      if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
        for (const SCEV *Op : Add->operands())
          Worklist.push_back(Op);
        continue;
      }
      if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
        auto It = LoopToDim.find(AR->getLoop());
        if (It != LoopToDim.end()) {
          unsigned D = It->second;
          if (!Result.Strides.count(D)) {
            Result.Strides[D] = AR->getStepRecurrence(SE);
            Result.AffineDims.set(D);
            FoundAffine = true;
          }
        }
        // Recurse into start to handle nested AddRecs.
        Worklist.push_back(AR->getStart());
      }
      // Scalar / unknown terms: ignore (they contribute to the base offset).
    }

    if (!FoundAffine) {
      // This GEP's index is loop-invariant at all GEMM dims —
      // could be an outer-scope offset. Continue walking upward.
    }

    Cur = GEP->getPointerOperand();
  }

  Result.Base = Cur;

  // Populate NonAffineDims = DimSet - AffineDims.
  unsigned N = DimSet.size();
  Result.NonAffineDims.resize(N);
  Result.AffineDims.resize(N);
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    if (!Result.AffineDims.test(UD))
      Result.NonAffineDims.set(UD);
  }

  return Result;
}
```

- [ ] **Step 2: Add `TPlanAnalysis.cpp` to CMakeLists.txt**

In `llvm/lib/Transforms/Vectorize/CMakeLists.txt`, add `TPlanAnalysis.cpp` to the source list alongside the other TPlan files:

```cmake
  TPlan.cpp
  TPlanAnalysis.cpp      # ← add this line
  TPlanWidener.cpp
  TPRecipeMatcher.cpp
  TPlanLowering.cpp
```

- [ ] **Step 3: Build**

```bash
ninja -C /Users/yun-yugyeong/Dev/llvm/build opt 2>&1 | tail -5
```
Expected: compiles cleanly, `bin/opt` linked.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanAnalysis.cpp \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt
git commit -m "tplan-analysis: implement decomposePtrForDims with bitcast/addrspacecast/PHI/srem support"
```

---

## Task 4: Wire `decomposePtrForDims()` into `emitContraction()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

- [ ] **Step 1: Add include**

At the top of `TPlanLowering.cpp`, add:

```cpp
#include "llvm/Transforms/Vectorize/TPlanAnalysis.h"
```

- [ ] **Step 2: Replace ad-hoc pointer/stride extraction in `emitContraction()`**

Find the block starting at line ~324 (after `LHSLoad`/`RHSLoad` are obtained) and replace through the `OutputDimSet` build with:

```cpp
  // Build DimToLoop from State (reuse existing helper or inline).
  // OutermostGEMMLoop = outermost loop whose IV appears in LHSLoad's DimSet.
  Loop *OutermostGEMMLoop = nullptr;
  for (int D = static_cast<int>(LHSDR->DimSet.size()) - 1; D >= 0; --D) {
    if (!LHSDR->DimSet.test(static_cast<unsigned>(D))) continue;
    auto It = State.DimToLoop.find(static_cast<unsigned>(D));
    if (It != State.DimToLoop.end()) {
      OutermostGEMMLoop = It->second;
      break;
    }
  }

  // Decompose A and B pointer chains.
  PtrDecomposition ADecomp = decomposePtrForDims(
      LHSLoad->getInstruction()->getPointerOperand(),
      LHSDR->DimSet, State.DimToLoop, OutermostGEMMLoop, *State.SE);
  PtrDecomposition BDecomp = decomposePtrForDims(
      RHSLoad->getInstruction()->getPointerOperand(),
      RHSDR->DimSet, State.DimToLoop, OutermostGEMMLoop, *State.SE);

  Value *LHSPtr = ADecomp.Base;
  Value *RHSPtr = BDecomp.Base;
  if (!LHSPtr || !RHSPtr)
    return nullptr;

  // Restrict OutputDimSet to dims where at least one of A/B has an affine
  // stride.  NonAffine dims are handled by the outer scalar loops (peeled).
  SmallBitVector AffineCoverage = ADecomp.AffineDims;
  AffineCoverage |= BDecomp.AffineDims;
```

Then in the `OutputDimSet` computation (after ContractDim is computed):

```cpp
  // Build OutputDimSet = (A.DimSet | B.DimSet) - {ContractDim} - NonAffineDims.
  unsigned NBits = std::max({static_cast<unsigned>(LHSDR->DimSet.size()),
                              static_cast<unsigned>(RHSDR->DimSet.size()),
                              static_cast<unsigned>(ContractDim + 1)});
  SmallBitVector LHSBits = LHSDR->DimSet, RHSBits = RHSDR->DimSet;
  LHSBits.resize(NBits); RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));
  // Remove non-affine dims — they are handled by outer scalar loops.
  SmallBitVector NonAffine = ADecomp.NonAffineDims;
  NonAffine.resize(NBits);
  OutputDimSet &= ~NonAffine;
```

Replace the `getOperandStride` lambda to use `ADecomp.Strides`/`BDecomp.Strides` instead of `getMemStride`:

```cpp
  auto getAStride = [&](unsigned Dim) -> Value * {
    auto It = ADecomp.Strides.find(Dim);
    if (It == ADecomp.Strides.end()) return I64(0);
    return expandStride(It->second, Dim);
  };
  auto getBStride = [&](unsigned Dim) -> Value * {
    auto It = BDecomp.Strides.find(Dim);
    if (It == BDecomp.Strides.end()) return I64(0);
    return expandStride(It->second, Dim);
  };

  SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    CStrides.push_back(CStoreRecipe
        ? expandStride(CStoreRecipe->getMemStride(UD, State.Plan, *State.SE), UD)
        : I64(State.Plan.getDenseStrideForDim(UD)));
    AStrides.push_back(getAStride(UD));
    BStrides.push_back(getBStride(UD));
    OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  unsigned ContUD = static_cast<unsigned>(ContractDim);
  Value *AContractStride = getAStride(ContUD);
  Value *BContractStride = getBStride(ContUD);
  Value *K = I64(State.Plan.getPFForDim(ContUD));
```

- [ ] **Step 3: Expose `DimToLoop` in `TPTransformState`**

In `TPlanLowering.cpp`, find `TPTransformState` struct and add the field:

```cpp
  DenseMap<unsigned, Loop *> DimToLoop; // populated in TPlanLowering_lower
```

In `TPlanLowering_lower()`, populate it after calling `TPlanWidener_widen()`:

```cpp
  // Build DimToLoop for use in decomposePtrForDims.
  // Reuse the same logic as TPRecipeMatcher.
  State.DimToLoop = buildDimToLoopForLowering(Plan, LI);
```

Add a local helper `buildDimToLoopForLowering` (same logic as `buildDimToLoop` in TPRecipeMatcher):

```cpp
static DenseMap<unsigned, Loop *>
buildDimToLoopForLowering(TPlan &Plan, LoopInfo &LI) {
  DenseMap<unsigned, Loop *> DimToLoop;
  SmallVector<TPBlockBase *> Worklist;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry()) Worklist.push_back(
      const_cast<TPBlockBase *>(Plan.getEntry()));
  while (!Worklist.empty()) {
    auto *Blk = Worklist.pop_back_val();
    if (!Visited.insert(Blk).second) continue;
    if (auto *BB = dyn_cast<TPBasicBlock>(Blk))
      for (TPRecipeBase &R : *BB)
        if (auto *IV = dyn_cast<TPWidenInductionRecipe>(&R))
          if (Loop *L = LI.getLoopFor(IV->getIVPhi()->getParent()))
            DimToLoop[IV->getDimIndex()] = L;
    if (auto *Reg = dyn_cast<TPRegionBlock>(Blk))
      if (Reg->getEntry()) Worklist.push_back(Reg->getEntry());
    for (TPBlockBase *S : Blk->getSuccessors()) Worklist.push_back(S);
  }
  return DimToLoop;
}
```

- [ ] **Step 4: Build**

```bash
ninja -C /Users/yun-yugyeong/Dev/llvm/build opt 2>&1 | tail -5
```
Expected: clean build.

- [ ] **Step 5: Run tests**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-bitcast.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-addrspacecast.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-phi.ll \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-srem.ll 2>&1 | tail -10
```
Expected: all 4 PASS.

- [ ] **Step 6: Run existing regression tests to check no breakage**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/llvm-lit -v \
  /Users/yun-yugyeong/Dev/llvm/llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -20
```
Expected: all existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: wire decomposePtrForDims into emitContraction for general ptr/stride decomposition"
```

---

## Task 5: Push and verify

- [ ] **Step 1: Push branch**

```bash
git push yg LoopTensorizebyClaude
```

- [ ] **Step 2: Smoke test with ggml input**

```bash
/Users/yun-yugyeong/Dev/llvm/build/bin/opt \
  -passes="loop-tensorize" \
  /Users/yun-yugyeong/Dev/llvm/ggml_compute_forward_mul_mat.ll \
  -o /dev/null --debug-only=loop-tensorize,tplan-lower 2>&1 \
  | grep -E "Stage|Contraction|cannot find"
```
Expected: `Contraction (contractDim=0)` present, `cannot find C pointer` absent.
