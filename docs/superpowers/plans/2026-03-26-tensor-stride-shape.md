# Tensor Stride & Shape Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `getTPValueShape()` to all `TensorOpKind`s, introduce per-recipe stride storage, add a companion `getTPValueStrides()`, declare new `@llvm.tensor.matmul` and `@llvm.tensor.elementwise.*` intrinsics, and update all lowering paths to emit stride-aware tensor IR.

**Architecture:** Two-level stride model: `TPlan::getDenseStrideForDim()` provides the packed-layout default (product of PFs of dims with index < D, innermost-first); `TPSingleDefRecipe::MemStrides` stores per-tensor overrides. Phase 1 uses static `uint64_t` strides — note this is a deliberate scope reduction from the spec's `TPValue*` design; dynamic stride support is left for future work. New target-independent intrinsics are declared programmatically (no `Intrinsics.td` change in Phase 1). All lowering paths (`Contraction`, `ElementWise`, `OuterProduct`, `BroadcastBinary`, `PlainReduction`) use shape+strides.

**Tech Stack:** LLVM C++ (ADT, IR, IRBuilder), opt pipeline (`-passes=loop-tensorize`), FileCheck lit tests.

**Spec:** `docs/superpowers/specs/2026-03-26-tensor-stride-shape-design.md`

---

## File Map

| File | Role |
|------|------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `getDenseStrideForDim()` to `TPlan`; add `MemStrides` + `getMemStride()` to `TPSingleDefRecipe` |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | Declare `getTPValueStrides()` |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | Implement `getTPValueStrides()` |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Add intrinsic helpers; update all `TensorOpKind` execute() paths |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll` | **NEW** — Contraction emits `@llvm.tensor.matmul` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll` | **NEW** — 2D ElementWise emits `@llvm.tensor.elementwise.fadd.2d` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll` | **NEW** — 3D ElementWise emits `@llvm.tensor.elementwise.fadd.3d` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll` | **NEW** — OuterProduct emits `@llvm.tensor.matmul` with K=1 |
| `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll` | **MODIFY** — update CHECK to expect `@llvm.tensor.matmul` |

---

## Task 1: Add `getDenseStrideForDim()` to `TPlan`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

Dense stride for dim D = product of `getPFForDim(d)` for all `d < D` (innermost-first convention).
Dim 0 (innermost) always returns 1.

- [ ] **Step 1: Add `getDenseStrideForDim()` after `getPFForDim()` in `TPlan.h`**

Around line 1288, after `setDimPF()`:

```cpp
/// Returns the dense (packed) stride for dimension \p Dim.
/// Dense stride(D) = product of getPFForDim(d) for all d < D.
/// Dim 0 (innermost) always returns 1.
/// \p Dim uses DimIdx convention (innermost=0, outermost=Depth-1).
uint64_t getDenseStrideForDim(unsigned Dim) const {
  uint64_t Stride = 1;
  for (unsigned D = 0; D < Dim; ++D)
    Stride *= static_cast<uint64_t>(getPFForDim(D));
  return Stride;
}
```

- [ ] **Step 2: Build to verify no compile errors**

```bash
ninja -C build llvm-opt 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan: add getDenseStrideForDim() to TPlan"
```

---

## Task 2: Add `MemStrides` + `getMemStride()` to `TPSingleDefRecipe`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

> **Note on spec divergence:** The spec declares `MemStrides` as `DenseMap<unsigned, TPValue*>` and `getTPValueStrides` returning `SmallVector<TPValue*>` to support dynamic strides. Phase 1 uses `uint64_t` throughout — a deliberate scope reduction. Adding dynamic stride support later will require changing these types and all call sites.

- [ ] **Step 1: Add `MemStrides` and `getMemStride()` to `TPSingleDefRecipe`**

In `TPlan.h`, inside `class TPSingleDefRecipe` (after the `DimSet` field, around line 712):

```cpp
/// Per-dim memory stride overrides (load/store recipes only).
/// Key: dim index (innermost=0). Value: stride in elements.
/// Absent entry → use Plan.getDenseStrideForDim(D).
/// Phase 1: always empty; SCEV-based population is future work.
DenseMap<unsigned, uint64_t> MemStrides;

/// Returns the effective memory stride for \p Dim.
/// Returns MemStrides[Dim] if set, else Plan.getDenseStrideForDim(Dim).
uint64_t getMemStride(unsigned Dim, const TPlan &Plan) const {
  auto It = MemStrides.find(Dim);
  return It != MemStrides.end() ? It->second : Plan.getDenseStrideForDim(Dim);
}
```

- [ ] **Step 2: Build**

```bash
ninja -C build llvm-opt 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan: add MemStrides + getMemStride() to TPSingleDefRecipe"
```

---

## Task 3: Add `getTPValueStrides()` to TPRecipeMatcher

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`

- [ ] **Step 1: Declare in `TPRecipeMatcher.h`**

After the `getTPValueShape()` declaration:

```cpp
/// Returns the effective memory stride for each dim in V.DimSet (innermost
/// first). Each entry is V.getMemStride(D, Plan): a recipe override if set,
/// else the TPlan dense default. Only meaningful for load/store recipes;
/// arithmetic recipes return dense defaults (which may be incorrect for them).
SmallVector<uint64_t> getTPValueStrides(const TPSingleDefRecipe &V,
                                         const TPlan &Plan);
```

- [ ] **Step 2: Implement in `TPRecipeMatcher.cpp`**

After the existing `getTPValueShape()` body (around line 172):

```cpp
SmallVector<uint64_t> llvm::getTPValueStrides(const TPSingleDefRecipe &V,
                                               const TPlan &Plan) {
  SmallVector<uint64_t> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan));
  return Strides;
}
```

- [ ] **Step 3: Build**

```bash
ninja -C build llvm-opt 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h \
        llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp
git commit -m "tplan: add getTPValueStrides() companion to getTPValueShape()"
```

---

## Task 4: Add intrinsic declaration helpers to `TPlanLowering.cpp`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

Intrinsics are declared via `Module::getOrInsertFunction()` — no `Intrinsics.td` change in Phase 1.
The functions return `FunctionCallee` for use with `IRBuilder::CreateCall`.

- [ ] **Step 1: Add `#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"` to includes**

`getTPValueStrides` is declared there; `TPlanLowering.cpp` already includes `TPRecipeMatcher.h` — verify it is present. If missing, add it after the existing TPRecipeMatcher include.

```cpp
#include "llvm/Transforms/Vectorize/TPRecipeMatcher.h"
#include "llvm/IR/Module.h"   // add if not present
```

- [ ] **Step 2: Add helper functions before the first `static` function in the file**

After the `DEBUG_TYPE` define (around line 33), insert:

```cpp
//===----------------------------------------------------------------------===//
// Intrinsic declaration helpers
//===----------------------------------------------------------------------===//

/// Returns @llvm.tensor.matmul.<type> — void with 12 i64/ptr args:
///   (ptr C, i64 M, i64 N, i64 ldc,
///    ptr A, i64 M, i64 K, i64 lda,
///    ptr B, i64 K, i64 N, i64 ldb)
static FunctionCallee getTensorMatmulFn(Module &M, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  std::string Name = "llvm.tensor.matmul.";
  Name += ElemTy->isFloatTy() ? "f32" : "f64";
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(Ctx),
      {PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty,
       PtrTy, I64Ty, I64Ty, I64Ty},
      /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}

/// Returns @llvm.tensor.elementwise.<op>.<rank>d.<type> — void.
/// Signature: ((ptr, i64×Rank) × 3 tensors, i64×Rank dims)
static FunctionCallee getTensorElementwiseFn(Module &M, StringRef OpName,
                                              unsigned Rank, Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  std::string Name = "llvm.tensor.elementwise.";
  Name += OpName.str();
  Name += "." + std::to_string(Rank) + "d.";
  Name += ElemTy->isFloatTy() ? "f32" : "f64";
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  for (unsigned T = 0; T < 3; ++T) {          // C, A, B
    Params.push_back(PtrTy);
    for (unsigned R = 0; R < Rank; ++R)
      Params.push_back(I64Ty);               // strides
  }
  for (unsigned R = 0; R < Rank; ++R)
    Params.push_back(I64Ty);                 // dims
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params, false);
  return M.getOrInsertFunction(Name, FT);
}
```

- [ ] **Step 3: Build**

```bash
ninja -C build llvm-opt 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: add getTensorMatmulFn/getTensorElementwiseFn helpers"
```

---

## Task 5: Update Contraction lowering → `@llvm.tensor.matmul`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll`

> **LDA/LDB rule:** For a row-major 2D matrix, the leading dimension (lda) is the stride of the outermost dimension. In innermost-first DimSet order, the outermost dim is always the *last* entry. So `lda = LHSStrides.back()`, independent of which position K occupies.

- [ ] **Step 1: Write the failing lit test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; 3-level GEMM (16x16x16, static trip counts).
; Contraction must emit @llvm.tensor.matmul.f32.
;
; CHECK: call void @llvm.tensor.matmul.f32
; CHECK-SAME: i64 16, i64 16, i64 16

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_16x16x16(ptr %A, ptr %B, ptr %C) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k = phi i64 [ 0, %j.loop ], [ %k.next, %k.loop ]
  %ik = add i64 %i, %k
  %kj = add i64 %k, %j
  %ij = add i64 %i, %j
  %aptr = getelementptr float, ptr %A, i64 %ik
  %bptr = getelementptr float, ptr %B, i64 %kj
  %cptr = getelementptr float, ptr %C, i64 %ij
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %cv   = load float, ptr %cptr
  %mul  = fmul float %av, %bv
  %add  = fadd float %cv, %mul
  store float %add, ptr %cptr
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
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

- [ ] **Step 2: Run — expect FAIL**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll
```

Expected: FAIL.

- [ ] **Step 3: Replace `emitContraction()` body**

Replace the full `emitContraction()` function (lines ~97–170 of `TPlanLowering.cpp`) with:

```cpp
static Value *emitContraction(const TPRecipeBase *FusedMul,
                               const TPRecipeBase *ReductionUpdate,
                               TPTransformState &State) {
  if (!FusedMul || FusedMul->operands().size() < 2)
    return nullptr;

  auto *LHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(0));
  auto *RHSDR = dyn_cast<TPSingleDefRecipe>(FusedMul->getOperand(1));
  if (!LHSDR || !RHSDR)
    return nullptr;
  if (LHSDR->DimSet.count() != 2 || RHSDR->DimSet.count() != 2) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction requires 2D operands\n");
    return nullptr;
  }

  Value *LHS = State.getValue(LHSDR);
  Value *RHS = State.getValue(RHSDR);
  if (!LHS || !RHS)
    return nullptr;

  SmallVector<unsigned>  LHSShape   = getTPValueShape(*LHSDR, State.Plan);
  SmallVector<unsigned>  RHSShape   = getTPValueShape(*RHSDR, State.Plan);
  SmallVector<uint64_t>  LHSStrides = getTPValueStrides(*LHSDR, State.Plan);
  SmallVector<uint64_t>  RHSStrides = getTPValueStrides(*RHSDR, State.Plan);

  int ContractDim = State.getContractDim(ReductionUpdate);
  auto findPos = [](const SmallBitVector &DS, int Dim) -> unsigned {
    unsigned Pos = 0;
    for (int D = DS.find_first(); D >= 0; D = DS.find_next(D), ++Pos)
      if (D == Dim) return Pos;
    return 0;
  };
  unsigned LHSPos = findPos(LHSDR->DimSet, ContractDim);
  unsigned RHSPos = findPos(RHSDR->DimSet, ContractDim);

  uint64_t M = LHSShape[1 - LHSPos];
  uint64_t K = LHSShape[LHSPos];
  uint64_t N = RHSShape[1 - RHSPos];
  // lda/ldb: leading dimension = stride of the outermost dim (last in
  // innermost-first DimSet order), regardless of where K sits.
  uint64_t LDA = LHSStrides.back();
  uint64_t LDB = RHSStrides.back();
  uint64_t LDC = State.Plan.getDenseStrideForDim(
      LHSDR->DimSet.find_last() + 1u);

  Type *ElemTy = LHS->getType()->getScalarType();
  if (!ElemTy->isFloatTy() && !ElemTy->isDoubleTy())
    return nullptr;

  // Locate the C accumulator pointer from the store recipe that consumes
  // the reduction result.
  Value *CPtr = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(1));
        if (PtrDR)
          CPtr = State.getValue(PtrDR);
        break;
      }
    }
  }
  if (!CPtr)
    CPtr = Constant::getNullValue(PointerType::getUnqual(
        State.Builder.getContext()));

  Module *Mod = State.Builder.GetInsertBlock()->getModule();
  auto MatmulFn = getTensorMatmulFn(*Mod, ElemTy);
  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };

  return B.CreateCall(MatmulFn,
      {CPtr, I64(M), I64(N), I64(LDC),
       LHS,  I64(M), I64(K), I64(LDA),
       RHS,  I64(K), I64(N), I64(LDB)});
}
```

- [ ] **Step 4: Update `matrix-multiply-emit.ll` to expect the new intrinsic**

Replace its `CHECK` line:

```llvm
; CHECK: call void @llvm.tensor.matmul.f32
; CHECK-NOT: @llvm.matrix.multiply
```

- [ ] **Step 5: Run both lit tests**

```bash
./build/bin/llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll \
  llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
```

Expected: both PASS.

- [ ] **Step 6: Run full suite — no regressions**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll \
        llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
git commit -m "tplan-lower: Contraction emits @llvm.tensor.matmul.f32"
```

---

## Task 6: Update ElementWise lowering → `@llvm.tensor.elementwise.*`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll`

- [ ] **Step 1: Write the failing lit tests**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; 2D elementwise fadd: C[i][j] = A[i][j] + B[i][j]
; dim0=j (innermost, PF=16), dim1=i (outermost, PF=8).
; Dense strides: dim0=1, dim1=16.
; Expected call: @llvm.tensor.elementwise.fadd.2d.f32(
;   ptr C, i64 1, i64 16,   ptr A, i64 1, i64 16,   ptr B, i64 1, i64 16,
;   i64 16, i64 8)
;
; CHECK: call void @llvm.tensor.elementwise.fadd.2d.f32
; CHECK-SAME: i64 1, i64 16
; CHECK-SAME: i64 1, i64 16
; CHECK-SAME: i64 1, i64 16
; CHECK-SAME: i64 16, i64 8

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @eltwise_fadd_2d(ptr %A, ptr %B, ptr %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %ij = add i64 %i, %j
  %aptr = getelementptr float, ptr %A, i64 %ij
  %bptr = getelementptr float, ptr %B, i64 %ij
  %cptr = getelementptr float, ptr %C, i64 %ij
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %cv = fadd float %av, %bv
  store float %cv, ptr %cptr
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %outer.latch, label %inner
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %outer
exit:
  ret void
}
```

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; 3D elementwise fadd: D[i][j][k] = A[i][j][k] + B[i][j][k]
; dim0=k (PF=4), dim1=j (PF=8), dim2=i (PF=16).
; Dense strides: dim0=1, dim1=4, dim2=32.
;
; CHECK: call void @llvm.tensor.elementwise.fadd.3d.f32
; CHECK-SAME: i64 1, i64 4, i64 32
; CHECK-SAME: i64 1, i64 4, i64 32
; CHECK-SAME: i64 1, i64 4, i64 32
; CHECK-SAME: i64 4, i64 8, i64 16

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @eltwise_fadd_3d(ptr %A, ptr %B, ptr %D) {
entry:
  br label %l.i
l.i:
  %i = phi i64 [ 0, %entry ], [ %i.next, %l.i.latch ]
  br label %l.j
l.j:
  %j = phi i64 [ 0, %l.i ], [ %j.next, %l.j.latch ]
  br label %l.k
l.k:
  %k = phi i64 [ 0, %l.j ], [ %k.next, %l.k.latch ]
  %ijk = add i64 %i, %j
  %ijk2 = add i64 %ijk, %k
  %aptr = getelementptr float, ptr %A, i64 %ijk2
  %bptr = getelementptr float, ptr %B, i64 %ijk2
  %dptr = getelementptr float, ptr %D, i64 %ijk2
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %dv = fadd float %av, %bv
  store float %dv, ptr %dptr
  br label %l.k.latch
l.k.latch:
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 4
  br i1 %k.done, label %l.j.latch, label %l.k
l.j.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 8
  br i1 %j.done, label %l.i.latch, label %l.j
l.i.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %exit, label %l.i
exit:
  ret void
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
./build/bin/llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll \
  llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll
```

Expected: FAIL.

- [ ] **Step 3: Replace the `ElementWise` case in `TPWidenRecipe::execute()`**

Use a local lambda for early-return to avoid any `goto`-across-initialization issues:

```cpp
case TensorOpKind::ElementWise: {
  auto tryVectorize = [&]() -> bool {
    auto *ADR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
    auto *BDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
    if (!ADR || !BDR) return false;

    unsigned Rank = ADR->DimSet.count();
    if (Rank < 1 || Rank > 3) return false; // TODO: support rank > 3

    // Determine op name from the BinaryOperator opcode.
    StringRef OpName;
    if (auto *BO = dyn_cast<BinaryOperator>(Inst)) {
      switch (BO->getOpcode()) {
      case Instruction::FAdd: OpName = "fadd"; break;
      case Instruction::FSub: OpName = "fsub"; break;
      case Instruction::FMul: OpName = "fmul"; break;
      default: return false;
      }
    } else {
      return false;
    }

    Value *APtr = State.getValue(ADR);
    Value *BPtr = State.getValue(BDR);
    if (!APtr || !BPtr) return false;

    // Find C pointer from the store recipe that uses this recipe's result.
    Value *CPtr = nullptr;
    if (auto *DefVal = this->getDefinedValue()) {
      for (TPUser *U : DefVal->users()) {
        auto *RB = dyn_cast<TPRecipeBase>(U);
        if (!RB) continue;
        if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
          auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(1));
          if (PtrDR) CPtr = State.getValue(PtrDR);
          break;
        }
      }
    }
    if (!CPtr) return false;

    SmallVector<unsigned>  Shape    = getTPValueShape(*ADR, State.Plan);
    SmallVector<uint64_t>  AStrides = getTPValueStrides(*ADR, State.Plan);
    SmallVector<uint64_t>  BStrides = getTPValueStrides(*BDR, State.Plan);
    SmallVector<uint64_t>  CStrides = AStrides; // same DimSet for elementwise

    Type *ElemTy = APtr->getType()->getScalarType();
    Module *Mod  = State.Builder.GetInsertBlock()->getModule();
    auto EltFn   = getTensorElementwiseFn(*Mod, OpName, Rank, ElemTy);
    IRBuilder<> &B = State.Builder;
    auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };

    SmallVector<Value *> Args;
    for (auto &[Ptr, Strides] : {std::pair{CPtr, &CStrides},
                                  std::pair{APtr, &AStrides},
                                  std::pair{BPtr, &BStrides}}) {
      Args.push_back(Ptr);
      for (uint64_t S : *Strides) Args.push_back(I64(S));
    }
    for (unsigned D : Shape) Args.push_back(I64(D));
    B.CreateCall(EltFn, Args);
    return true;
  };

  if (tryVectorize()) return;

  // Scalar fallback.
  auto *Clone = Inst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[Inst] = Result;
  State.setValue(this, Result);
  return;
}
```

- [ ] **Step 4: Run the lit tests**

```bash
./build/bin/llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll \
  llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll
```

Expected: PASS.

- [ ] **Step 5: Run existing eltwise test — no regression**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/pf-dimset-eltwise.ll
```

Expected: PASS (that test uses `fdiv` which hits the scalar fallback).

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-2d.ll \
        llvm/test/Transforms/LoopTensorize/basic/tensor-eltwise-stride-3d.ll
git commit -m "tplan-lower: ElementWise emits @llvm.tensor.elementwise.fadd.{2,3}d"
```

---

## Task 7: Update OuterProduct lowering → `@llvm.tensor.matmul` with K=1

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll`

- [ ] **Step 1: Write the failing lit test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; Outer product: C[i][j] = A[i] * B[j]
; LHS DimSet={dim1(i)} shape=[8], RHS DimSet={dim0(j)} shape=[16].
; Disjoint DimSets → OuterProduct → tensor.matmul(M=8, K=1, N=16).
;
; CHECK: call void @llvm.tensor.matmul.f32
; CHECK-SAME: i64 8, i64 1
; CHECK-SAME: i64 1, i64 16

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @outer_product(ptr %A, ptr %B, ptr %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %inner
inner:
  %j = phi i64 [ 0, %outer ], [ %j.next, %inner.latch ]
  %aptr = getelementptr float, ptr %A, i64 %i
  %bptr = getelementptr float, ptr %B, i64 %j
  %cptr = getelementptr float, ptr %C, i64 %i
  %av = load float, ptr %aptr
  %bv = load float, ptr %bptr
  %cv = fmul float %av, %bv
  store float %cv, ptr %cptr
  br label %inner.latch
inner.latch:
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %outer.latch, label %inner
outer.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 8
  br i1 %i.done, label %exit, label %outer
exit:
  ret void
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll
```

Expected: FAIL.

- [ ] **Step 3: Replace the `OuterProduct` case using a lambda**

```cpp
case TensorOpKind::OuterProduct: {
  auto tryVectorize = [&]() -> bool {
    auto *LHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
    auto *RHSDR = dyn_cast<TPSingleDefRecipe>(getOperand(1));
    if (!LHSDR || !RHSDR) return false;

    SmallVector<unsigned> LHSShape = getTPValueShape(*LHSDR, State.Plan);
    SmallVector<unsigned> RHSShape = getTPValueShape(*RHSDR, State.Plan);
    if (LHSShape.empty() || RHSShape.empty()) return false;

    uint64_t M = 1, N = 1;
    for (unsigned D : LHSShape) M *= D;
    for (unsigned D : RHSShape) N *= D;

    Value *LHS = State.getValue(LHSDR);
    Value *RHS = State.getValue(RHSDR);
    if (!LHS || !RHS) return false;

    Value *CPtr = nullptr;
    if (auto *DefVal = this->getDefinedValue()) {
      for (TPUser *U : DefVal->users()) {
        auto *RB = dyn_cast<TPRecipeBase>(U);
        if (!RB) continue;
        if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
          auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(1));
          if (PtrDR) CPtr = State.getValue(PtrDR);
          break;
        }
      }
    }
    if (!CPtr) return false;

    Type *ElemTy = LHS->getType()->getScalarType();
    Module *Mod  = State.Builder.GetInsertBlock()->getModule();
    auto MatmulFn = getTensorMatmulFn(*Mod, ElemTy);
    IRBuilder<> &B = State.Builder;
    auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };

    // OuterProduct = matmul(col-vec[M×1], row-vec[1×N]) → C[M×N]
    B.CreateCall(MatmulFn,
        {CPtr, I64(M), I64(N), I64(N),    // C, M, N, ldc=N (dense)
         LHS,  I64(M), I64(1), I64(1),    // A: M×1 col-vector, lda=1
         RHS,  I64(1), I64(N), I64(N)});  // B: 1×N row-vector, ldb=N
    return true;
  };

  if (tryVectorize()) return;

  // Scalar fallback.
  auto *Clone = Inst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[Inst] = Result;
  State.setValue(this, Result);
  return;
}
```

- [ ] **Step 4: Run the lit test**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-outer-product.ll
git commit -m "tplan-lower: OuterProduct emits @llvm.tensor.matmul with K=1"
```

---

## Task 8: `BroadcastBinary` and `PlainReduction` — shape+stride fallback note

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

Both ops currently clone the scalar instruction. They already call `getTPValueShape`/`getTPValueStrides` through the scalar-clone path in Phase 1 — no intrinsic exists for them yet. Add a debug log that prints the shape/strides so the information is at least visible.

- [ ] **Step 1: Add shape/stride debug logging to BroadcastBinary and PlainReduction**

In `TPWidenRecipe::execute()`, replace the `BroadcastBinary` case:

```cpp
case TensorOpKind::BroadcastBinary: {
  LLVM_DEBUG({
    auto *DR = dyn_cast<TPSingleDefRecipe>(getOperand(0));
    if (DR) {
      auto Shape   = getTPValueShape(*DR, State.Plan);
      auto Strides = getTPValueStrides(*DR, State.Plan);
      dbgs() << "BroadcastBinary shape=[";
      for (unsigned D : Shape) dbgs() << D << " ";
      dbgs() << "] strides=[";
      for (uint64_t S : Strides) dbgs() << S << " ";
      dbgs() << "] — scalar fallback (TODO: broadcast intrinsic)\n";
    }
  });
  auto *Clone = Inst->clone();
  State.remapClone(Clone);
  Value *Result = State.Builder.Insert(Clone);
  applyFlags(*cast<Instruction>(Result));
  State.EmittedMap[Inst] = Result;
  State.setValue(this, Result);
  return;
}
```

Apply the same pattern to `PlainReduction`.

- [ ] **Step 2: Build**

```bash
ninja -C build llvm-opt 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: add shape/stride debug logging to BroadcastBinary and PlainReduction"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run the full LoopTensorize lit suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all pass, no regressions.

- [ ] **Step 2: Check new tests are all present and discovered**

```bash
./build/bin/llvm-lit --show-tests llvm/test/Transforms/LoopTensorize/basic/ \
  | grep tensor
```

Expected output includes:
```
tensor-matmul-emit.ll
tensor-eltwise-stride-2d.ll
tensor-eltwise-stride-3d.ll
tensor-outer-product.ll
```

- [ ] **Step 3: Final commit (if any outstanding changes)**

```bash
git status
# commit any remaining unstaged changes
```

---

## Notes on Future Work (out of scope)

- **SCEV-based `MemStrides` population:** in `TPlanWidener_widen()`, inspect GEP index coefficients per loop dim to detect non-dense strides and populate `TPSingleDefRecipe::MemStrides`. Change value type from `uint64_t` to `TPValue*` for dynamic strides, and update `getTPValueStrides` return type accordingly.
- **Dynamic dims:** extend `@llvm.tensor.matmul` backend lowering to dispatch to ARM SME / BLAS / tiled loop when dims are runtime values.
- **Rank > 3 elementwise:** add `.4d`, `.5d` variants or adopt a variadic design.
- **Register in `Intrinsics.td`:** promote programmatically-declared intrinsics to the official LLVM intrinsic registry.
- **`BroadcastBinary` / `PlainReduction` vectorization:** design and emit proper intrinsics once the shape/stride infrastructure is validated.
