# TPlan SCEV-Based Dynamic Stride Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the always-dense `MemStrides` stub with SCEV-derived per-dimension strides so that `llvm.tensor.matmul` and `llvm.tensor.elementwise` receive correct runtime leading-dimension arguments for non-packed tensors.

**Architecture:** During `TPRecipePatternMatcher_match()`, walk each load/store recipe's GEP index using ScalarEvolution to extract per-loop AddRec steps, store them as `const SCEV*` in `MemStrides`. During lowering, `SCEVExpander` materializes each SCEV to a `Value*` that is passed directly to the intrinsic call.

**Tech Stack:** LLVM ScalarEvolution (`SE.getSCEV`, `SCEVAddRecExpr::getStepRecurrence`), `SCEVExpander` (`llvm/Transforms/Utils/ScalarEvolutionExpander.h`), FileCheck lit tests.

---

## File Map

| File | Role |
|------|------|
| `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Data model: `MemStrides` type, `getMemStride()` decl, `TPWidenStoreRecipe` extension, `TPTransformState` fields |
| `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Non-inline `getMemStride()` definitions (both load and store recipes) |
| `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h` | Updated `getTPValueStrides()` + `TPRecipePatternMatcher_match()` signatures |
| `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp` | `populateSCEVStrides()` helpers, updated `getTPValueStrides()`, updated `TPRecipePatternMatcher_match()` |
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Wire SE/Expander into State, update all stride call sites, fix LDC |
| `llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll` | Lit test: strided GEMM verifies runtime `%lda`/`%ldb` args in call |

---

## Task 1: Write the failing lit test

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll`

- [ ] **Step 1: Write the test**

```llvm
; RUN: opt -passes=loop-tensorize -S %s | FileCheck %s
;
; Verify that a GEMM with non-dense leading dimensions (%lda, %ldb) produces
; a llvm.tensor.matmul call with runtime stride arguments, not constant
; dense defaults.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: @gemm_strided
; CHECK: call void @llvm.tensor.matmul.f32(
; CHECK-SAME: i64 %lda
; CHECK-SAME: i64 %ldb
;
; Also verify the dense case: A[i*K+k] → LDA=%K, B[k*N+j] → LDB=%N, LDC=%N.
; (Before SCEV strides the dense default was K*N, not K.)
; CHECK-LABEL: @gemm_dense
; CHECK: call void @llvm.tensor.matmul.f32(
; CHECK-SAME: i64 %N
; CHECK-SAME: i64 %K
; CHECK-SAME: i64 %N

; Strided GEMM: A[i][k] with leading dim lda, B[k][j] with leading dim ldb,
; C[i][j] dense (leading dim N).
define void @gemm_strided(ptr %A, ptr %B, ptr %C,
                           i64 %M, i64 %N, i64 %K,
                           i64 %lda, i64 %ldb) {
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
  ; A[i*lda + k] — outer stride is %lda (non-dense when lda > K)
  %ai  = mul i64 %i, %lda
  %ak  = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av  = load float, ptr %aptr, align 4
  ; B[k*ldb + j] — outer stride is %ldb (non-dense when ldb > N)
  %bk  = mul i64 %k, %ldb
  %bj  = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv  = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  ; C[i*N + j] — dense (stride == N)
  %ci  = mul i64 %i, %N
  %cj  = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  %cv  = load float, ptr %cptr, align 4
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

; Dense GEMM: A[i*K+k], B[k*N+j], C[i*N+j].
; Expected after SCEV: LDA=%K, LDB=%N, LDC=%N.
define void @gemm_dense(ptr %A, ptr %B, ptr %C,
                         i64 %M, i64 %N, i64 %K) {
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
  %ai  = mul i64 %i, %K
  %ak  = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %av  = load float, ptr %aptr, align 4
  %bk  = mul i64 %k, %N
  %bj  = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %bv  = load float, ptr %bptr, align 4
  %prod = fmul float %av, %bv
  %ci  = mul i64 %i, %N
  %cj  = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  %cv  = load float, ptr %cptr, align 4
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

- [ ] **Step 2: Verify the test fails (expected: missing %lda/%ldb in intrinsic args)**

```bash
cd /Users/yun-yugyeong/Dev/llvm
ninja -C build opt 2>&1 | tail -5
build/bin/opt -passes=loop-tensorize -S \
  llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll \
  | grep "tensor.matmul"
```

Expected: the matmul call is emitted but stride args are constant integers (dense defaults), NOT `%lda`/`%ldb`. The FileCheck will fail. That's correct — this is TDD.

- [ ] **Step 3: Commit the test**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll
git commit -m "test: add failing lit test for SCEV-based strided matmul strides"
```

---

## Task 2: Data model changes in TPlan.h and TPlan.cpp

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

### TPlan.h changes

- [ ] **Step 1: Change `MemStrides` type and `getMemStride()` declaration in `TPSingleDefRecipe` (around line 728)**

Replace:
```cpp
  /// Per-dim memory stride overrides (load/store recipes only).
  /// Key: dim index (innermost=0). Value: stride in elements.
  /// Absent entry → use Plan.getDenseStrideForDim(D).
  /// Phase 1: always empty; SCEV-based population is future work.
  /// Note: spec uses TPValue* for dynamic strides; uint64_t is a deliberate
  /// Phase 1 scope reduction.
  DenseMap<unsigned, uint64_t> MemStrides;

  /// Returns the effective memory stride for \p Dim.
  /// Returns MemStrides[Dim] if set, else Plan.getDenseStrideForDim(Dim).
  uint64_t getMemStride(unsigned Dim, const TPlan &Plan) const;
```

With:
```cpp
  /// Per-dim memory stride overrides (load/store recipes only).
  /// Key: dim index (innermost=0). Value: SCEV stride expression in elements.
  /// Absent entry → dense default expressed as a SCEV constant.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  /// Returns MemStrides[Dim] if set, else SE.getConstant(getDenseStrideForDim(Dim)).
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;
```

- [ ] **Step 2: Add `DimSet`, `MemStrides`, and `getMemStride()` to `TPWidenStoreRecipe` (around line 1181)**

Replace the existing class body:
```cpp
class TPWidenStoreRecipe : public TPRecipeBase {
public:
  TPWidenStoreRecipe(Instruction *Store, TPValue *PtrOp, TPValue *ValOp)
      : TPRecipeBase(TPWidenStoreSC, {PtrOp, ValOp}), StoreInst(Store) {}

  Instruction *getInstruction() const { return StoreInst; }

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenStoreSC;
  }

private:
  Instruction *StoreInst;
};
```

With:
```cpp
class TPWidenStoreRecipe : public TPRecipeBase {
public:
  TPWidenStoreRecipe(Instruction *Store, TPValue *PtrOp, TPValue *ValOp)
      : TPRecipeBase(TPWidenStoreSC, {PtrOp, ValOp}), StoreInst(Store) {}

  Instruction *getInstruction() const { return StoreInst; }

  /// Dimensions this store participates in. Copied from the stored-value
  /// operand's DimSet by TPRecipePatternMatcher_match().
  SmallBitVector DimSet;

  /// Per-dim memory stride overrides in elements.
  /// Populated by TPRecipePatternMatcher_match() via SCEV GEP-index analysis.
  DenseMap<unsigned, const SCEV *> MemStrides;

  /// Returns the effective memory stride for \p Dim as a SCEV expression.
  const SCEV *getMemStride(unsigned Dim, const TPlan &Plan,
                            ScalarEvolution &SE) const;

  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
  void execute(TPTransformState &State) const override;

  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenStoreSC;
  }

private:
  Instruction *StoreInst;
};
```

- [ ] **Step 3: Add `SE` and `Expander` to `TPTransformState` (around line 1384)**

Replace the existing struct body:
```cpp
struct TPTransformState {
  IRBuilder<> &Builder;
  const TPlan &Plan;
  const RecipeClassMap *ClassMap = nullptr;
  DenseMap<const TPRecipeValue *, Value *> ValueMap;
  DenseMap<Value *, Value *> EmittedMap;

  TPTransformState(IRBuilder<> &B, const TPlan &P) : Builder(B), Plan(P) {}
```

With:
```cpp
struct TPTransformState {
  IRBuilder<> &Builder;
  const TPlan &Plan;
  const RecipeClassMap *ClassMap = nullptr;
  DenseMap<const TPRecipeValue *, Value *> ValueMap;
  DenseMap<Value *, Value *> EmittedMap;
  /// Set by TPlanLowering_lower() before execute() loop.
  ScalarEvolution *SE = nullptr;
  /// Set by TPlanLowering_lower() before execute() loop. Owned by the caller.
  SCEVExpander *Expander = nullptr;

  TPTransformState(IRBuilder<> &B, const TPlan &P) : Builder(B), Plan(P) {}
```

For `SCEVExpander` to be forward-declarable in TPlan.h, add a forward declaration near the other class forward declarations (around line 33):
```cpp
class SCEVExpander;
```

- [ ] **Step 4: Remove the old inline `getMemStride()` definition (around line 1432)**

Remove these lines entirely:
```cpp
inline uint64_t TPSingleDefRecipe::getMemStride(unsigned Dim,
                                               const TPlan &Plan) const {
  auto It = MemStrides.find(Dim);
  return It != MemStrides.end() ? It->second : Plan.getDenseStrideForDim(Dim);
}
```

The definition moves to TPlan.cpp in the next step.

### TPlan.cpp changes

- [ ] **Step 5: Add both `getMemStride()` definitions to TPlan.cpp**

After the existing `using namespace llvm;` line, add (TPlan.cpp already includes `ScalarEvolution.h`):

```cpp
const SCEV *TPSingleDefRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                             ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}

const SCEV *TPWidenStoreRecipe::getMemStride(unsigned Dim, const TPlan &Plan,
                                              ScalarEvolution &SE) const {
  auto It = MemStrides.find(Dim);
  if (It != MemStrides.end())
    return It->second;
  return SE.getConstant(APInt(64, Plan.getDenseStrideForDim(Dim)));
}
```

- [ ] **Step 6: Build to confirm compilation errors are only in dependents (not TPlan itself)**

```bash
ninja -C build LLVMVectorize 2>&1 | grep "error:" | head -20
```

Expected: errors in TPRecipeMatcher.cpp and TPlanLowering.cpp (old callers of the old API). No errors in TPlan.cpp itself.

---

## Task 3: Update TPRecipeMatcher.h and TPRecipeMatcher.cpp

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h`
- Modify: `llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp`

### Header changes

- [ ] **Step 1: Update TPRecipeMatcher.h with new signatures and forward decls**

Replace the entire file content:
```cpp
//===- TPRecipeMatcher.h - Pattern matching for TPlan recipes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// Declares TPRecipePatternMatcher_match(), getTPValueShape(), and
/// getTPValueStrides(). Requires TPlanWidener_widen() to have been called first.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
#define LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Vectorize/TPlanTypes.h"

namespace llvm {
class LoopInfo;
class SCEV;
class ScalarEvolution;
class TPlan;
class TPSingleDefRecipe;

/// Returns the tensor shape of \p V: { Plan.getPFForDim(d) for d in V.DimSet }.
/// Returns {} for scalar (empty DimSet) values.
/// Requires TPlanWidener_widen() to have been called first.
SmallVector<unsigned> getTPValueShape(const TPSingleDefRecipe &V,
                                       const TPlan &Plan);

/// Returns the effective memory stride for each dim in V.DimSet (innermost
/// first) as SCEV expressions in element units. Each entry is
/// V.getMemStride(D, Plan, SE): a SCEV from MemStrides if populated by
/// TPRecipePatternMatcher_match(), else a dense-default SCEV constant.
SmallVector<const SCEV *> getTPValueStrides(const TPSingleDefRecipe &V,
                                             const TPlan &Plan,
                                             ScalarEvolution &SE);

/// Classify every recipe in \p Plan into a TensorOpKind, and populate
/// MemStrides on load/store recipes using SCEV GEP-index analysis.
/// Requires TPlanWidener_widen() to have been called first.
/// Results are written into \p Out (existing entries are overwritten).
void TPRecipePatternMatcher_match(TPlan &Plan, RecipeClassMap &Out,
                                   ScalarEvolution &SE, LoopInfo &LI);

} // namespace llvm
#endif // LLVM_TRANSFORMS_VECTORIZE_TRECIPEMATCHER_H
```

Note: `TPlan &Plan` is now non-const because the matcher mutates recipe `MemStrides`.

### Implementation changes

- [ ] **Step 2: Add required includes to TPRecipeMatcher.cpp**

After the existing includes, add:
```cpp
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Instructions.h"   // already present, confirms GetElementPtrInst
```

- [ ] **Step 3: Update `getTPValueStrides()` implementation**

Replace (around line 174):
```cpp
SmallVector<uint64_t> llvm::getTPValueStrides(const TPSingleDefRecipe &V,
                                               const TPlan &Plan) {
  SmallVector<uint64_t> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan));
  return Strides;
}
```

With:
```cpp
SmallVector<const SCEV *> llvm::getTPValueStrides(const TPSingleDefRecipe &V,
                                                   const TPlan &Plan,
                                                   ScalarEvolution &SE) {
  SmallVector<const SCEV *> Strides;
  for (int D = V.DimSet.find_first(); D >= 0; D = V.DimSet.find_next(D))
    Strides.push_back(V.getMemStride(static_cast<unsigned>(D), Plan, SE));
  return Strides;
}
```

- [ ] **Step 4: Add `populateSCEVStrides` helpers before `collectBBs` (around line 182)**

```cpp
/// Build a map from TPlan dimension index → Loop* by scanning IV recipes
/// in the plan's header block.
static DenseMap<unsigned, Loop *>
buildDimToLoop(TPlan &Plan, LoopInfo &LI) {
  DenseMap<unsigned, Loop *> DimToLoop;
  if (!Plan.getEntry())
    return DimToLoop;
  // Walk the entry block (preheader) and all descendants looking for
  // TPWidenInductionRecipe nodes. Each one wraps a loop PHI that identifies
  // the Loop* for its dimension.
  SmallVector<TPBlockBase *, 8> Worklist;
  SmallPtrSet<TPBlockBase *, 8> Seen;
  Worklist.push_back(const_cast<TPBlockBase *>(Plan.getEntry()));
  while (!Worklist.empty()) {
    TPBlockBase *Blk = Worklist.pop_back_val();
    if (!Seen.insert(Blk).second)
      continue;
    if (auto *BB = dyn_cast<TPBasicBlock>(Blk)) {
      for (TPRecipeBase &R : *BB) {
        if (auto *IV = dyn_cast<TPWidenInductionRecipe>(&R)) {
          auto *Phi = cast<PHINode>(IV->getInstruction());
          if (Loop *L = LI.getLoopFor(Phi->getParent()))
            DimToLoop[IV->getDimIdx()] = L;
        }
      }
    }
    if (auto *Reg = dyn_cast<TPRegionBlock>(Blk))
      if (Reg->getEntry())
        Worklist.push_back(Reg->getEntry());
    for (TPBlockBase *Succ : Blk->getSuccessors())
      Worklist.push_back(Succ);
  }
  return DimToLoop;
}

/// Extract per-dimension element-count strides from \p GEPIdx (the flat index
/// expression of a single-index GEP) and store them in \p MemStrides.
/// Uses the AddRec step of each loop's recurrence as the stride in elements.
/// Absent dimensions are left unset → getMemStride() uses the dense default.
static void populateSCEVStridesFromIndex(
    DenseMap<unsigned, const SCEV *> &MemStrides,
    const SmallBitVector &DimSet,
    Value *GEPIdx,
    const DenseMap<unsigned, Loop *> &DimToLoop,
    ScalarEvolution &SE) {
  const SCEV *IdxSCEV = SE.getSCEV(GEPIdx);
  // Walk nested AddRecs to collect loop → step mappings.
  DenseMap<const Loop *, const SCEV *> LoopStep;
  const SCEV *S = IdxSCEV;
  while (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    LoopStep[AR->getLoop()] = AR->getStepRecurrence(SE);
    S = AR->getStart();
  }
  // Map dimension index → SCEV step.
  for (int D = DimSet.find_first(); D >= 0; D = DimSet.find_next(D)) {
    auto DIt = DimToLoop.find(static_cast<unsigned>(D));
    if (DIt == DimToLoop.end())
      continue;
    auto SIt = LoopStep.find(DIt->second);
    if (SIt != LoopStep.end())
      MemStrides[static_cast<unsigned>(D)] = SIt->second;
    // Absent → getMemStride() falls back to dense SCEV constant.
  }
}

/// Populate MemStrides on a load recipe by analysing the GEP index of
/// its load instruction's pointer operand.
static void populateSCEVStrides(TPWidenLoadRecipe &LR,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  auto *LI = cast<LoadInst>(LR.getInstruction());
  auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
  // Only handle the common flat-index pattern: getelementptr T, ptr base, i64 idx
  if (!GEP || GEP->getNumIndices() != 1)
    return;
  populateSCEVStridesFromIndex(LR.MemStrides, LR.DimSet,
                                GEP->getOperand(1), DimToLoop, SE);
}

/// Populate DimSet + MemStrides on a store recipe. DimSet is copied from
/// the stored-value operand's DimSet; strides come from the store's GEP index.
static void populateSCEVStrides(TPWidenStoreRecipe &SR,
                                 const DenseMap<unsigned, Loop *> &DimToLoop,
                                 ScalarEvolution &SE) {
  // Copy DimSet from stored-value operand (operand 1).
  if (auto *ValDR = dyn_cast<TPSingleDefRecipe>(SR.getOperand(1)))
    SR.DimSet = ValDR->DimSet;
  if (SR.DimSet.none())
    return;
  auto *SI = cast<StoreInst>(SR.getInstruction());
  auto *GEP = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
  if (!GEP || GEP->getNumIndices() != 1)
    return;
  populateSCEVStridesFromIndex(SR.MemStrides, SR.DimSet,
                                GEP->getOperand(1), DimToLoop, SE);
}
```

- [ ] **Step 5: Update `TPRecipePatternMatcher_match()` signature and body**

Replace (around line 198):
```cpp
void llvm::TPRecipePatternMatcher_match(const TPlan &Plan,
                                         RecipeClassMap &Out) {
  SmallVector<const TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);

  for (const TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      RecipeClassification C;
      if (isReductionUpdate(&R)) {
        C = classifyReduction(R, Plan);
      } else if (R.getTPRecipeID() == TPRecipeBase::TPWidenSC &&
                 isa<BinaryOperator>(
                     cast<TPWidenRecipe>(R).getInstruction()) &&
                 R.operands().size() == 2) {
        C.Kind = classifyBinaryOp(R);
      }
      // else: load, store, cast, PHI, canonical IV → default Scalar
      Out[&R] = C;
    }
  }

  // Second pass: mark each FusedMulRecipe of a Contraction as Contraction too.
  SmallVector<std::pair<TPRecipeBase *, int>> FusedMuls;
  for (auto &[R, C] : Out) {
    if (C.Kind == TensorOpKind::Contraction && C.FusedMulRecipe)
      FusedMuls.push_back({C.FusedMulRecipe, C.ContractDim});
  }
  for (auto [MulR, Dim] : FusedMuls) {
    Out[MulR].Kind = TensorOpKind::Contraction;
    Out[MulR].ContractDim = Dim;
  }
}
```

With:
```cpp
void llvm::TPRecipePatternMatcher_match(TPlan &Plan, RecipeClassMap &Out,
                                         ScalarEvolution &SE, LoopInfo &LI) {
  // Build dim→Loop mapping from IV recipes before classifying.
  DenseMap<unsigned, Loop *> DimToLoop = buildDimToLoop(Plan, LI);

  SmallVector<const TPBasicBlock *, 32> AllBBs;
  SmallPtrSet<TPBlockBase *, 32> Visited;
  if (Plan.getEntry())
    collectBBs(const_cast<TPBlockBase *>(Plan.getEntry()), AllBBs, Visited);

  for (const TPBasicBlock *BB : AllBBs) {
    for (const TPRecipeBase &R : *BB) {
      // Populate SCEV strides on load/store recipes before classification.
      if (auto *LR = dyn_cast<TPWidenLoadRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenLoadRecipe *>(LR), DimToLoop, SE);
      else if (auto *SR = dyn_cast<TPWidenStoreRecipe>(&R))
        populateSCEVStrides(*const_cast<TPWidenStoreRecipe *>(SR), DimToLoop, SE);

      RecipeClassification C;
      if (isReductionUpdate(&R)) {
        C = classifyReduction(R, Plan);
      } else if (R.getTPRecipeID() == TPRecipeBase::TPWidenSC &&
                 isa<BinaryOperator>(
                     cast<TPWidenRecipe>(R).getInstruction()) &&
                 R.operands().size() == 2) {
        C.Kind = classifyBinaryOp(R);
      }
      // else: load, store, cast, PHI, canonical IV → default Scalar
      Out[&R] = C;
    }
  }

  // Second pass: mark each FusedMulRecipe of a Contraction as Contraction too.
  SmallVector<std::pair<TPRecipeBase *, int>> FusedMuls;
  for (auto &[R, C] : Out) {
    if (C.Kind == TensorOpKind::Contraction && C.FusedMulRecipe)
      FusedMuls.push_back({C.FusedMulRecipe, C.ContractDim});
  }
  for (auto [MulR, Dim] : FusedMuls) {
    Out[MulR].Kind = TensorOpKind::Contraction;
    Out[MulR].ContractDim = Dim;
  }
}
```

- [ ] **Step 6: Build to confirm TPRecipeMatcher compiles**

```bash
ninja -C build LLVMVectorize 2>&1 | grep "TPRecipeMatcher" | head -10
```

Expected: no errors from TPRecipeMatcher.cpp. Remaining errors should be only in TPlanLowering.cpp.

- [ ] **Step 7: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPRecipeMatcher.h \
        llvm/lib/Transforms/Vectorize/TPRecipeMatcher.cpp \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan-stride: data model + SCEV stride population in pattern matcher"
```

---

## Task 4: Update TPlanLowering.cpp

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

### 4a: Add SCEVExpander include

- [ ] **Step 1: Add the SCEVExpander include** (after the existing ScalarEvolution include, around line 22)

```cpp
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
```

### 4b: Wire SE and Expander into TPlanLowering_lower()

- [ ] **Step 2: Update `TPplanLowering_lower()` to pass SE/LI to matcher and wire State**

Replace (around line 600):
```cpp
bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);
  ...
  // 2. Classify every recipe by DimSet patterns.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM);
  ...
  // 3. Lower: walk block CFG in construction order.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;
```

With:
```cpp
bool llvm::TPlanLowering_lower(TPlan &Plan, Function &F, LoopInfo &LI,
                                ScalarEvolution &SE, DominatorTree &DT) {
  // 1. Propagate DimSets via BFS.
  TPlanWidener_widen(Plan);
  ...
  // 2. Classify every recipe and populate SCEV strides on load/store recipes.
  RecipeClassMap CM;
  TPRecipePatternMatcher_match(Plan, CM, SE, LI);
  ...
  // 3. Lower: walk block CFG in construction order.
  IRBuilder<> Builder(F.getContext());
  if (!F.empty())
    Builder.SetInsertPoint(&F.getEntryBlock().front());

  SCEVExpander Expander(SE, F.getParent()->getDataLayout(), "tplan.stride");

  TPTransformState State(Builder, Plan);
  State.ClassMap = &CM;
  State.SE = &SE;
  State.Expander = &Expander;
```

Keep all other lines in the function unchanged.

### 4c: Add `expandStride` helper and update `emitContraction`

- [ ] **Step 3: Add `expandStride` lambda and update the matmul call in `emitContraction`**

In `emitContraction`, after the existing `auto I64 = ...` lambda (around line 260), add:

```cpp
  // Expand a SCEV stride expression to a Value* at the current insert point.
  // Falls back to the dense default constant if expansion is unsafe.
  auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
    if (State.Expander && State.Expander->isSafeToExpand(S))
      return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                            &*B.GetInsertPoint());
    return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
  };
```

Locate the line that finds `CPtr` via the store recipe (around line 221). Save the store recipe reference alongside `CPtr`:

Replace:
```cpp
  Value *CPtr = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
        if (PtrDR)
          CPtr = State.getValue(PtrDR);
        break;
      }
    }
  }
```

With:
```cpp
  Value *CPtr = nullptr;
  TPWidenStoreRecipe *CStoreRecipe = nullptr;
  if (auto *DefVal = ReductionUpdate->getDefinedValue()) {
    for (TPUser *U : DefVal->users()) {
      auto *RB = dyn_cast<TPRecipeBase>(U);
      if (!RB) continue;
      if (auto *SR = dyn_cast<TPWidenStoreRecipe>(RB)) {
        CStoreRecipe = SR;
        auto *PtrDR = dyn_cast<TPSingleDefRecipe>(SR->getOperand(0));
        if (PtrDR)
          CPtr = State.getValue(PtrDR);
        break;
      }
    }
  }
```

Then replace the matmul call construction (around line 262):
```cpp
  return B.CreateCall(MatmulFn,
      {CPtr,    I64(M), I64(N), I64(LDC),
       LHSPtr,  I64(M), I64(K), I64(LDA),
       RHSPtr,  I64(K), I64(N), I64(LDB)});
```

With:
```cpp
  // Compute LDA, LDB from SCEV strides; LDC from store recipe's MemStrides.
  // lastDim is the outermost dimension in LHS's DimSet = the leading-dim axis.
  unsigned LHSLastDim = static_cast<unsigned>(LHSDR->DimSet.find_last());
  unsigned RHSLastDim = static_cast<unsigned>(RHSDR->DimSet.find_last());
  Value *VDA = expandStride(LHSStrides.back(), LHSLastDim);
  Value *VDB = expandStride(RHSStrides.back(), RHSLastDim);
  Value *VDC;
  if (CStoreRecipe && State.SE) {
    // Use the outermost dim in C's DimSet for the leading dimension.
    int CLastDim = CStoreRecipe->DimSet.find_last();
    const SCEV *LDC_SCEV = CLastDim >= 0
        ? CStoreRecipe->getMemStride(static_cast<unsigned>(CLastDim),
                                      State.Plan, *State.SE)
        : State.SE->getConstant(APInt(64, 1));
    VDC = expandStride(LDC_SCEV, CLastDim >= 0 ? static_cast<unsigned>(CLastDim) : 0);
  } else {
    VDC = I64(State.Plan.getDenseStrideForDim(LHSLastDim + 1));
  }

  return B.CreateCall(MatmulFn,
      {CPtr,   I64(M), I64(N), VDC,
       LHSPtr, I64(M), I64(K), VDA,
       RHSPtr, I64(K), I64(N), VDB});
```

Also remove the now-unused `LDA`, `LDB`, `LDC` local variables (lines 210–213) since they are replaced by `VDA`, `VDB`, `VDC`.

Remove:
```cpp
  uint64_t LDA = LHSStrides.back();
  uint64_t LDB = RHSStrides.back();
  uint64_t LDC = State.Plan.getDenseStrideForDim(
      static_cast<unsigned>(LHSDR->DimSet.find_last()) + 1u);
```

And update the `LHSStrides`/`RHSStrides` variable types by changing the two lines at ~184–185:

Replace:
```cpp
  SmallVector<uint64_t>  LHSStrides = getTPValueStrides(*LHSDR, State.Plan);
  SmallVector<uint64_t>  RHSStrides = getTPValueStrides(*RHSDR, State.Plan);
```

With:
```cpp
  SmallVector<const SCEV *> LHSStrides =
      getTPValueStrides(*LHSDR, State.Plan, *State.SE);
  SmallVector<const SCEV *> RHSStrides =
      getTPValueStrides(*RHSDR, State.Plan, *State.SE);
```

### 4d: Update elementwise stride call sites

- [ ] **Step 4: Update the elementwise execute() (around line 418)**

Add the `expandStride` lambda at the top of the elementwise block (inside the `case TensorOpKind::ElementWise:` body, after `auto I64 = ...`):

```cpp
      auto expandStride = [&](const SCEV *S, unsigned Dim) -> Value * {
        if (State.Expander && State.Expander->isSafeToExpand(S))
          return State.Expander->expandCodeFor(S, B.getInt64Ty(),
                                               &*B.GetInsertPoint());
        return B.getInt64(State.Plan.getDenseStrideForDim(Dim));
      };
```

Replace:
```cpp
      SmallVector<uint64_t>  AStrides = getTPValueStrides(*ADR, State.Plan);
      SmallVector<uint64_t>  BStrides = getTPValueStrides(*BDR, State.Plan);
      // C shares the same DimSet as A/B in an elementwise op, so its dense
      // strides match AStrides. Phase 1 does not yet populate MemStrides for C
      // independently; when strided-C support is added, query the store recipe.
      SmallVector<uint64_t>  CStrides = AStrides;
```

With:
```cpp
      SmallVector<const SCEV *> AStrides =
          getTPValueStrides(*ADR, State.Plan, *State.SE);
      SmallVector<const SCEV *> BStrides =
          getTPValueStrides(*BDR, State.Plan, *State.SE);
      // Derive C strides from the store recipe's MemStrides.
      SmallVector<const SCEV *> CStrides;
      // Find the store recipe for C (same approach as emitContraction).
      if (auto *DefVal = this->getDefinedValue()) {
        for (TPUser *U : DefVal->users()) {
          if (auto *SR = dyn_cast<TPWidenStoreRecipe>(U)) {
            for (int D = SR->DimSet.find_first(); D >= 0;
                 D = SR->DimSet.find_next(D))
              CStrides.push_back(
                  SR->getMemStride(static_cast<unsigned>(D), State.Plan, *State.SE));
            break;
          }
        }
      }
      if (CStrides.empty())
        CStrides = AStrides; // fallback: use A's strides
```

Replace the Args-building loop:
```cpp
      SmallVector<Value *> Args;
      for (auto &[Ptr, Strides] : {std::pair{CPtr, &CStrides},
                                    std::pair{APtr, &AStrides},
                                    std::pair{BPtr, &BStrides}}) {
        Args.push_back(Ptr);
        for (uint64_t S : *Strides) Args.push_back(I64(S));
      }
```

With:
```cpp
      // Build args: for each operand, push ptr then one stride per dim.
      // appendStrideArgs iterates DimSet in find_first order (innermost-first),
      // matching the order in which Strides was built by getTPValueStrides().
      auto appendStrideArgs = [&](SmallVector<const SCEV *> &Strides,
                                   const SmallBitVector &DimBV) {
        int D = DimBV.find_first();
        for (const SCEV *S : Strides) {
          Args.push_back(expandStride(S, D >= 0 ? static_cast<unsigned>(D) : 0));
          if (D >= 0) D = DimBV.find_next(D);
        }
      };
      SmallVector<Value *> Args;
      // C
      Args.push_back(CPtr);
      appendStrideArgs(CStrides, CStoreRecipe ? CStoreRecipe->DimSet : ADR->DimSet);
      // A
      Args.push_back(APtr);
      appendStrideArgs(AStrides, ADR->DimSet);
      // B
      Args.push_back(BPtr);
      appendStrideArgs(BStrides, BDR->DimSet);

### 4e: Update debug-path call sites

- [ ] **Step 5: Update the two debug-path `getTPValueStrides` calls (BroadcastBinary ~line 472, PlainReduction ~line 577)**

For both occurrences, change:
```cpp
        auto Strides = getTPValueStrides(*DR, State.Plan);
```

To:
```cpp
        auto Strides = State.SE
            ? getTPValueStrides(*DR, State.Plan, *State.SE)
            : SmallVector<const SCEV *>{};
```

And update the corresponding debug print loop from `for (uint64_t S : Strides)` to `for (const SCEV *S : Strides)`:
```cpp
        for (const SCEV *S : Strides) {
          if (S) S->print(dbgs());
          else dbgs() << "?";
          dbgs() << " ";
        }
```

- [ ] **Step 6: Build everything**

```bash
ninja -C build opt 2>&1 | grep "error:" | head -20
```

Expected: clean build, zero errors.

- [ ] **Step 7: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-stride: wire SCEVExpander into lowering; update all stride call sites"
```

---

## Task 5: Verify test passes and run existing tests

- [ ] **Step 1: Run the new strided matmul lit test**

```bash
build/bin/opt -passes=loop-tensorize -S \
  llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll \
  | grep "tensor.matmul"
```

Expected output (something like):
```
  call void @llvm.tensor.matmul.f32(ptr %C, i64 %M, i64 %N, i64 %N, ptr %A, i64 %M, i64 %K, i64 %lda, ptr %B, i64 %K, i64 %N, i64 %ldb)
```

The key check: `%lda` and `%ldb` appear as stride args instead of constant integers.

- [ ] **Step 2: Run FileCheck on the new test**

```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll
```

Expected: `PASS`.

- [ ] **Step 3: Run the existing tplan-print test to confirm no regression**

```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
```

Expected: `PASS`.

- [ ] **Step 4: Run the full LoopTensorize test suite**

```bash
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass.

- [ ] **Step 5: Final commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tplan-strided-matmul.ll
git commit -m "tplan-stride: SCEV-based dynamic stride analysis complete; all tests passing"
```
