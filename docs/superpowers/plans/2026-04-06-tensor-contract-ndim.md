# N-Dim Tensor Contraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `emitContraction()` from 2D-only to rank 1–4 per operand and introduce the `llvm.tensor.contract.<Ra>d.<Rb>d.<type>` intrinsic family, enabling GEMV, batched GEMM, and multi-head attention patterns.

**Architecture:** Add `getTensorContractFn()` helper, rewrite `emitContraction()` to build output-order stride vectors, update existing GEMM tests to use the new intrinsic name, and add new GEMV and batched-GEMM tests.

**Tech Stack:** LLVM C++ (IR builder, SCEV, SmallBitVector), lit/FileCheck

---

## File Map

| File | Role |
|------|------|
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Add `getTensorContractFn()`; rewrite `emitContraction()` |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll` | Update CHECK `matmul` → `contract.2d.2d` |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll` | Update CHECK (already fixed by GEP-fix plan) |
| `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll` | Update CHECK (already fixed by GEP-fix plan) |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll` | New: 1D×2D GEMV |
| `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll` | New: 3D×3D batched GEMM |

---

### Task 1: Add `getTensorContractFn()` and rewrite `emitContraction()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

- [ ] **Step 1: Write the failing lit test for 2D×2D GEMM with new intrinsic name**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 2D GEMM using the new contract intrinsic (replaces tensor.matmul).
; A[i*%K+k], B[k*%N+j], C[i*%N+j]. Runtime strides %K and %N.
; CHECK: call void @llvm.tensor.contract.2d.2d.f32(
; CHECK-SAME: i64 0
; CHECK-SAME: i64 %K
; CHECK-SAME: i64 %N

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_2d(ptr %A, ptr %B, ptr %C, i64 %K, i64 %N) {
entry:
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  %ai   = mul i64 %i, %K
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, %N
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ci   = mul i64 %i, %N
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr
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

- [ ] **Step 2: Run the test — verify it fails (still emitting `tensor.matmul`)**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll
```
Expected: FAIL (output contains `llvm.tensor.matmul.f32`, not `contract.2d.2d`)

- [ ] **Step 3: Add `getTensorContractFn()` before `getTensorElementwiseFn` (after line 138)**

In `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`, insert after the closing `}` of
`getTensorMatmulFn()` (line 138):

```cpp
/// Returns (creating if needed) @llvm.tensor.contract.<Ra>d.<Rb>d.<type>.
/// Signature (RankC = |OutputDimSet| = |(A|B).DimSet - {ContractDim}|):
///   void(ptr C, i64×RankC C_strides,
///        ptr A, i64×RankC A_strides, i64 A_contract_stride,
///        ptr B, i64×RankC B_strides, i64 B_contract_stride,
///        i64 K,
///        i64×RankC output_dims)
/// A/B strides are in output-dim order; 0 encodes broadcast (A/B ∌ that dim).
static FunctionCallee getTensorContractFn(Module &M, unsigned RankA,
                                           unsigned RankB, unsigned RankC,
                                           Type *ElemTy) {
  LLVMContext &Ctx = M.getContext();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  assert(!TypeSuffix.empty() && "unsupported element type for contract");
  std::string Name = (Twine("llvm.tensor.contract.") + Twine(RankA) + "d." +
                      Twine(RankB) + "d." + TypeSuffix).str();
  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  SmallVector<Type *> Params;
  Params.push_back(PtrTy);                                         // C
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);  // C strides
  Params.push_back(PtrTy);                                         // A
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);  // A strides
  Params.push_back(I64Ty);                                         // A contract stride
  Params.push_back(PtrTy);                                         // B
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);  // B strides
  Params.push_back(I64Ty);                                         // B contract stride
  Params.push_back(I64Ty);                                         // K
  for (unsigned i = 0; i < RankC; ++i) Params.push_back(I64Ty);  // output dims
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), Params,
                                        /*isVarArg=*/false);
  return M.getOrInsertFunction(Name, FT);
}
```

- [ ] **Step 4: Replace `emitContraction()` body**

Replace the entire body of `emitContraction()` (lines 282–421) with:

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

  unsigned RankA = LHSDR->DimSet.count();
  unsigned RankB = RHSDR->DimSet.count();
  if (RankA < 1 || RankA > 4 || RankB < 1 || RankB > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction rank out of [1,4]\n");
    return nullptr;
  }

  auto *LHSLoad = dyn_cast<TPWidenLoadRecipe>(LHSDR);
  auto *RHSLoad = dyn_cast<TPWidenLoadRecipe>(RHSDR);
  if (!LHSLoad || !RHSLoad)
    return nullptr;

  auto *LHSPtrDR = dyn_cast<TPSingleDefRecipe>(LHSLoad->getOperand(0));
  auto *RHSPtrDR = dyn_cast<TPSingleDefRecipe>(RHSLoad->getOperand(0));
  if (!LHSPtrDR || !RHSPtrDR)
    return nullptr;

  Value *LHSPtr = State.getValue(LHSPtrDR);
  Value *RHSPtr = State.getValue(RHSPtrDR);
  if (!LHSPtr || !RHSPtr)
    return nullptr;

  Type *ElemTy = LHSLoad->getInstruction()->getType()->getScalarType();
  StringRef TypeSuffix = getTypeSuffix(ElemTy);
  if (TypeSuffix.empty())
    return nullptr;

  int ContractDim = State.getContractDim(ReductionUpdate);
  if (ContractDim < 0 ||
      !LHSDR->DimSet.test(static_cast<unsigned>(ContractDim)) ||
      !RHSDR->DimSet.test(static_cast<unsigned>(ContractDim))) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction dim not in DimSets\n");
    return nullptr;
  }

  // Build OutputDimSet = (A.DimSet | B.DimSet) − {ContractDim}.
  unsigned NBits = std::max({LHSDR->DimSet.size(), RHSDR->DimSet.size(),
                              static_cast<unsigned>(ContractDim + 1)});
  SmallBitVector LHSBits = LHSDR->DimSet, RHSBits = RHSDR->DimSet;
  LHSBits.resize(NBits); RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));

  unsigned RankC = OutputDimSet.count();
  if (RankC < 1 || RankC > 4) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction output rank out of [1,4]\n");
    return nullptr;
  }

  // Locate C accumulator pointer (primary: recipe users; fallback: IR users).
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
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
  if (!CPtr) {
    if (auto *WR = dyn_cast<TPWidenRecipe>(ReductionUpdate)) {
      for (User *U : WR->getInstruction()->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          Value *P = SI->getPointerOperand();
          CPtr = isa<GetElementPtrInst>(P)
                     ? cast<GetElementPtrInst>(P)->getPointerOperand()
                     : P;
          break;
        }
      }
    }
  }
  if (!CPtr) {
    LLVM_DEBUG(dbgs() << "TPlanLowering: Contraction cannot find C pointer\n");
    return nullptr;
  }

  IRBuilder<> &B = State.Builder;
  auto I64 = [&](uint64_t V) -> Value * { return B.getInt64(V); };
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return I64(State.Plan.getDenseStrideForDim(Dim));
  };
  // Returns A/B stride for output dim Dim: 0 if operand doesn't span it.
  auto getOperandStride = [&](const TPSingleDefRecipe *DR,
                               unsigned Dim) -> Value * {
    if (!DR->DimSet.test(Dim))
      return I64(0);
    return expandStride(DR->getMemStride(Dim, State.Plan, *State.SE), Dim);
  };

  // Build stride/dim vectors in output-dim order.
  SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    // C strides: from store recipe if available, else dense default.
    if (CStoreRecipe && State.SE)
      CStrides.push_back(expandStride(
          CStoreRecipe->getMemStride(UD, State.Plan, *State.SE), UD));
    else
      CStrides.push_back(I64(State.Plan.getDenseStrideForDim(UD)));
    AStrides.push_back(getOperandStride(LHSDR, UD));
    BStrides.push_back(getOperandStride(RHSDR, UD));
    OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  // Contraction dim strides and K.
  unsigned ContUD = static_cast<unsigned>(ContractDim);
  Value *AContractStride = expandStride(
      LHSDR->getMemStride(ContUD, State.Plan, *State.SE), ContUD);
  Value *BContractStride = expandStride(
      RHSDR->getMemStride(ContUD, State.Plan, *State.SE), ContUD);
  Value *K = I64(State.Plan.getPFForDim(ContUD));

  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee ContractFn =
      getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy);

  SmallVector<Value *> Args;
  Args.push_back(CPtr);
  Args.append(CStrides.begin(), CStrides.end());
  Args.push_back(LHSPtr);
  Args.append(AStrides.begin(), AStrides.end());
  Args.push_back(AContractStride);
  Args.push_back(RHSPtr);
  Args.append(BStrides.begin(), BStrides.end());
  Args.push_back(BContractStride);
  Args.push_back(K);
  Args.append(OutDims.begin(), OutDims.end());

  return B.CreateCall(ContractFn, Args);
}
```

- [ ] **Step 5: Build**

```bash
ninja -C build LoopTensorize 2>&1 | tail -20
```
Expected: compiles without errors or warnings.

- [ ] **Step 6: Run the 2D GEMM test — verify it passes**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll
```
Expected: PASS (`contract.2d.2d.f32` present, `i64 0` for A_stride[j],
`i64 %K` for A_contract_stride, `i64 %N` for B_contract_stride)

- [ ] **Step 7: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemm-2d.ll
git commit -m "tplan-lower: replace emitContraction 2D-only check with N-dim contract intrinsic"
```

---

### Task 2: Update existing GEMM tests to use the new intrinsic name

**Files:**
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll`
- Modify: `llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll`

- [ ] **Step 1: Run all three tests — confirm they now fail**

```bash
llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll \
  llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll \
  llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
```
Expected: all FAIL (`llvm.tensor.matmul.f32` no longer emitted by Contraction path)

- [ ] **Step 2: Update `tplan-strided-matmul.ll`**

Change the two CHECK lines that reference `tensor.matmul.f32`:

```diff
-; CHECK: call void @llvm.tensor.matmul.f32(
+; CHECK: call void @llvm.tensor.contract.2d.2d.f32(
 ; CHECK-SAME: i64 %lda
 ; CHECK-SAME: i64 %ldb
```

(Apply to both `@gemm_strided` and `@gemm_dense` CHECK blocks.)

For `@gemm_dense`, `%lda` corresponds to `A_contract_stride=%K` and `%ldb`
corresponds to `B_contract_stride=%N`. The CHECK-SAME lines match substrings,
so they remain valid — `%K` and `%N` still appear in the new intrinsic call.

- [ ] **Step 3: Update `tensor-matmul-emit.ll`**

```diff
-; CHECK: call void @llvm.tensor.matmul.f32(ptr {{.*}}, i64 256, i64 256, i64 %N,
-; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 %K,
-; CHECK-SAME: ptr {{.*}}, i64 256, i64 256, i64 %N)
+; CHECK: call void @llvm.tensor.contract.2d.2d.f32(
+; CHECK-SAME: i64 0
+; CHECK-SAME: i64 %K
+; CHECK-SAME: i64 %N
```

(Note: this test was already updated by the GEP-fix plan to use runtime `%K`/`%N`;
only the intrinsic name changes here.)

- [ ] **Step 4: Update `matrix-multiply-emit.ll`**

```diff
-; CHECK: call void @llvm.tensor.matmul.f32
+; CHECK: call void @llvm.tensor.contract.2d.2d.f32
 ; CHECK-NOT: @llvm.matrix.multiply
```

- [ ] **Step 5: Run all three tests — verify they pass**

```bash
llvm-lit -v \
  llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll \
  llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll \
  llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add \
  llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll \
  llvm/test/Transforms/LoopTensorize/basic/tensor-matmul-emit.ll \
  llvm/test/Transforms/LoopTensorize/basic/matrix-multiply-emit.ll
git commit -m "test: update GEMM tests from tensor.matmul to tensor.contract.2d.2d"
```

---

### Task 3: Add GEMV test — `contract.1d.2d.f32`

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll`

GEMV: `C[j] += A[k] * B[k][j]`
- Loop: `j` outer (dim 1), `k` inner + reduction (dim 0)
- A: DimSet={0}, RankA=1
- B: DimSet={0,1}, RankB=2
- OutputDimSet={1}, RankC=1, output dim order=[j]
- A_strides[j]=0 (A∌j), A_contract_stride=1 (A[k] dense)
- B_strides[j]=1 (B innermost), B_contract_stride=%N (stride per k in B[k*N+j])

- [ ] **Step 1: Write the failing test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; GEMV: C[j] += A[k] * B[k][j]
; A is 1D (DimSet={dim0=k}), B is 2D (DimSet={dim0=k, dim1=j}).
; OutputDimSet = {dim1=j}, RankC=1.
; Emits @llvm.tensor.contract.1d.2d.f32 with A_stride[j]=0.
;
; CHECK: call void @llvm.tensor.contract.1d.2d.f32(
; CHECK-SAME: i64 0, i64 1
; CHECK-SAME: i64 %N

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemv(ptr %A, ptr %B, ptr %C, i64 %N) {
entry:
  br label %j.loop
j.loop:
  %j   = phi i64   [ 0,   %entry  ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  %aptr = getelementptr float, ptr %A, i64 %k
  %bk   = mul i64 %k, %N
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %cptr = getelementptr float, ptr %C, i64 %j
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %exit, label %j.loop
exit:
  ret void
}
```

- [ ] **Step 2: Run — verify FAIL (before Task 1 was done, it fell back to scalar)**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll
```
Expected: FAIL (no `contract.1d.2d.f32` in output)

Note: if Task 1 is already complete, this test may already pass. Skip to Step 4.

- [ ] **Step 3: Verify Task 1 is complete (implementation already done)**

After Task 1's implementation, the 1D×2D case should be handled without
additional code changes — the new `emitContraction()` supports RankA=1.

- [ ] **Step 4: Run — verify PASS**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll
```
Expected: PASS

The key checks:
- `i64 0, i64 1` — A_stride[j]=0 (A doesn't span j), A_contract_stride=1
- `i64 %N` — B_contract_stride=%N (stride per k in B[k*N+j])

- [ ] **Step 5: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-gemv.ll
git commit -m "test: add GEMV (1D×2D) contract.1d.2d.f32 lit test"
```

---

### Task 4: Add batched GEMM test — `contract.3d.3d.f32`

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll`

Batched GEMM: `C[b][i][j] += A[b][i][k] * B[b][k][j]`
- Loops: `b` outermost (dim 3), `i` (dim 2), `j` (dim 1), `k` innermost+reduction (dim 0)
- A: DimSet={0,2,3}, RankA=3
- B: DimSet={0,1,3}, RankB=3
- OutputDimSet={1,2,3}, RankC=3, output dim order=[j, i, b]
- Shared free dim: 3 (b) — both A and B span it, both strides non-zero
- A_strides in output order [j,i,b] = [0, %K, %IK] (A∌j → 0)
- B_strides in output order [j,i,b] = [1, 0, %KN]  (B∌i → 0)

GEP expressions (with runtime params %K=cols-of-A, %N=cols-of-B,
%IK=rows×cols-of-A, %KN=rows×cols-of-B, %IN=rows×cols-of-C):
- A[b][i][k] → `b*%IK + i*%K + k`
- B[b][k][j] → `b*%KN + k*%N + j`
- C[b][i][j] → `b*%IN + i*%N + j`

- [ ] **Step 1: Write the failing test**

Create `llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; Batched GEMM: C[b][i][j] += A[b][i][k] * B[b][k][j]
; dim0=k (inner+reduction), dim1=j, dim2=i, dim3=b (outer).
; A: DimSet={0,2,3} RankA=3. B: DimSet={0,1,3} RankB=3.
; OutputDimSet={1,2,3} RankC=3. Output order=[j,i,b].
; Shared free dim b: A_stride[b]=%IK, B_stride[b]=%KN (both non-zero).
;
; CHECK: call void @llvm.tensor.contract.3d.3d.f32(
; CHECK-SAME: i64 0
; CHECK-SAME: i64 %K
; CHECK-SAME: i64 0, i64 %KN
; CHECK-SAME: i64 %N

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @batched_gemm(ptr %A, ptr %B, ptr %C,
                           i64 %K, i64 %N,
                           i64 %IK, i64 %KN, i64 %IN) {
entry:
  br label %b.loop
b.loop:
  %b = phi i64 [ 0, %entry ], [ %b.next, %i.latch ]
  br label %i.loop
i.loop:
  %i = phi i64 [ 0, %b.loop ], [ %i.next, %j.latch ]
  br label %j.loop
j.loop:
  %j = phi i64 [ 0, %i.loop ], [ %j.next, %k.latch ]
  br label %k.loop
k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; A[b*%IK + i*%K + k]
  %ab   = mul i64 %b, %IK
  %ai   = mul i64 %i, %K
  %aib  = add i64 %ab, %ai
  %aibk = add i64 %aib, %k
  %aptr = getelementptr float, ptr %A, i64 %aibk
  ; B[b*%KN + k*%N + j]
  %bb   = mul i64 %b, %KN
  %bk   = mul i64 %k, %N
  %bbk  = add i64 %bb, %bk
  %bbkj = add i64 %bbk, %j
  %bptr = getelementptr float, ptr %B, i64 %bbkj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 16
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  ; C[b*%IN + i*%N + j]
  %cb   = mul i64 %b, %IN
  %ci   = mul i64 %i, %N
  %cbi  = add i64 %cb, %ci
  %cbij = add i64 %cbi, %j
  %cptr = getelementptr float, ptr %C, i64 %cbij
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 16
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 16
  br i1 %i.done, label %i.latch, label %i.loop
i.latch:
  %b.next = add i64 %b, 1
  %b.done = icmp eq i64 %b.next, 4
  br i1 %b.done, label %exit, label %b.loop
exit:
  ret void
}
```

- [ ] **Step 2: Run — verify FAIL (before Task 1, 3D falls back to scalar)**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll
```
Expected: FAIL

- [ ] **Step 3: Run after Task 1 is complete — verify PASS**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll
```
Expected: PASS

Key checks:
- `i64 0` — A_stride[j]=0 (A doesn't span j)
- `i64 %K` — A_stride[i]=%K (A[b,i,k]: stride per i)
- `i64 0, i64 %KN` — B_stride[i]=0 (B∌i), B_stride[b]=%KN
- `i64 %N` — B_contract_stride=%N (stride per k in B[b,k,j])

- [ ] **Step 4: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-batched.ll
git commit -m "test: add batched GEMM (3D×3D) contract.3d.3d.f32 lit test"
```

---

### Task 5: Run full suite and push

- [ ] **Step 1: Run all LoopTensorize tests**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/
```
Expected: all pass (no regressions)

- [ ] **Step 2: Confirm OuterProduct path still uses `tensor.matmul`**

```bash
llvm-lit -v llvm/test/Transforms/LoopTensorize/ 2>&1 | grep -i matmul
```
The OuterProduct lowering (inside `tryVectorize` for `BroadcastBinary`) still
calls `getTensorMatmulFn`, so any existing OuterProduct tests still pass.

- [ ] **Step 3: Push**

```bash
git push yg LoopTensorizebyClaude
```
