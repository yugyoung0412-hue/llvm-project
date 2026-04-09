# Outer-Dim Tensorization: 3-Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `tensor.contract.2d.2d.2d.f32` (full M×N×K tiling) for ggml-style GEMMs where A's row pointer is computed in a separate outer-loop body block, by (1) fixing the `buildInitial()` ordering bug, (2) pre-building all tiling loop block skeletons before `execute()`, and (3) simplifying `emitContraction()` to just fill in the pre-built body.

**Architecture:**
- **Section 1**: Fix `buildInitial()` in `TPlan.cpp` — emit outer body blocks into ValueMap *before* recursing into inner region, so preheader-split GEPs (e.g. `%a.row = gep %A, %i*K` between M-loop and N-loop) get correct DimSet propagation.
- **Section 2**: Add `PrebuiltTilingInfo` to `TPTransformState` and a `preBuildTilingBlocks()` function in `TPlanLowering.cpp`. After `createTensorizedLoopSkeleton()` (guard) but **before** `execute()`, call `preBuildTilingBlocks()` to emit M-tile, N-tile, K-tile loop block skeletons using `emitTilingLoop()` / `emitFixedCountingLoop()` in Policy order. Store in `State.PrebuiltTiling`.
- **Section 3**: Strip loop-creation code from `emitContraction()`; when `State.PrebuiltTiling` is set, move insert point to the pre-built body BB, compute pointer offsets with pre-built IVs, emit `tensor.contract` call, and close the pre-built loops.

**Tech Stack:** C++ (LLVM), `IRBuilder<>`, SCEV, LLVM lit tests.

---

## Background & Root Causes

### Section 1 — `buildInitial()` ordering bug (`TPlan.cpp`)

`BuildRegion(Idx)` processes one loop level. For non-innermost loops:

```
Line 846: if (Idx + 1 < AllLoops.size()) {
Line 850:   auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);
Line 851:   auto *Child = BuildRegion(Idx + 1);   ← recurse FIRST   ← BUG
...
Line 864:   InnerPH->setInsertionBB(L->getHeader());
Line 866:   for (BasicBlock *BB : L->blocks()) {
Line 867:     if (BB == L->getHeader() || BB == L->getLoopLatch()) continue;
Line 868:     if (InnerLoop->contains(BB)) continue;
Line 869:     EmitBlock(BB, InnerPH);               ← too late: inner already built
```

Any value computed in a body block between the outer loop header and the inner loop (e.g. `%a.row = gep %A, %i * K` in `i.body` between `i.loop` and `j.loop`) is NOT in `ValueMap` when `BuildRegion(Idx+1)` recurses. Those downstream GEP recipes fall back to `ir<%a.row>` (a live-in with `DimSet={}`), so BFS never propagates the M-dim bit through the A GEP chain.

**Fix**: Move `InnerPH->setInsertionBB()` + `EmitBlock` loop to **before** the recursive call (line 851).

### Sections 2/3 — Architectural separation of structure from content

Currently `emitContraction()` both:
- **Creates** tiling loop blocks (via `emitTilingLoop()` / `emitFixedCountingLoop()`)
- **Fills** them with pointer offsets and the `tensor.contract` call

The policy (which dims to tile, PF, TC) is already decided before `execute()`. Building the loop structure inside `emitContraction()` (during `execute()`) mixes structure with content and makes the code harder to reason about.

**Target**: Policy → `preBuildTilingBlocks()` (structure) → `execute()` / `emitContraction()` (content).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll` | Failing lit test (Section 1 regression) |
| Modify | `llvm/lib/Transforms/Vectorize/TPlan.cpp` lines 846–870 | Section 1: fix `BuildRegion` ordering |
| Modify | `llvm/include/llvm/Transforms/Vectorize/TPlan.h` near `TPTransformState` | Section 2: add `PrebuiltTilingInfo` struct + field |
| Modify | `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Section 2: add `preBuildTilingBlocks()`; Section 3: simplify `emitContraction()` |

---

## Task 1: Write the Failing Lit Test (Section 1)

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll`

- [ ] **Step 1: Write the test file**

The test uses a 3-loop GEMM (M=N=K=16, static) where A's row pointer is computed in a **separate `i.body` block** between the M-loop header and the N-loop. Without the buildInitial() fix, `%a.row` is a live-in and A-load gets `DimSet={0}` (K only), producing `tensor.contract.1d.2d.1d.f32`. After the fix, A-load gets `DimSet={0,2}` (K, M), OutputDimSet={1,2}, RankC=2 → `tensor.contract.2d.2d.2d.f32`.

```llvm
; llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
; RUN: opt -passes=loop-tensorize -S --disable-verify < %s | FileCheck %s
;
; 3-loop GEMM (M=N=K=16) where A's row pointer is computed in a separate
; i.body block between the M-loop header and the N-loop. This triggers the
; buildInitial() ordering bug: without the fix, i.body is emitted AFTER
; BuildRegion(j.loop) recurses, so %a.row is seen as ir<> (DimSet={}) in
; the k-loop body. After the fix, i.body is emitted first and %a.row
; gets DimSet={2} (M-dim), propagating to A-load DimSet={0,2}.
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
  ; A's row pointer computed in a SEPARATE block — between M-loop header
  ; and N-loop. This is the pattern that triggers the buildInitial() bug.
  %a.row.off = mul i64 %i, 16
  %a.row = getelementptr float, ptr %A, i64 %a.row.off
  br label %j.loop

j.loop:
  %j = phi i64 [ 0, %i.body ], [ %j.next, %k.latch ]
  br label %k.loop

k.loop:
  %k   = phi i64   [ 0,   %j.loop ], [ %k.next, %k.loop ]
  %acc = phi float [ 0.0, %j.loop ], [ %sum,    %k.loop ]
  ; A[i][k]: uses %a.row from i.body (cross-block outer-loop body reference)
  %a.ptr = getelementptr float, ptr %a.row, i64 %k
  ; B[k][j]: flat 2-D layout, K×N stride
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

Expected: **FAIL** — `tensor.contract.2d.2d.2d.f32` not found (currently emits a `1d` variant or bails out).

- [ ] **Step 3: Commit the failing test**

```bash
cd /Users/yun-yugyeong/Dev/llvm
git add llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
git commit -m "$(cat <<'EOF'
test: add failing lit for preheader-split GEP outer-dim DimSet propagation

The test has A's row pointer in a separate i.body block between the M-loop
header and the N-loop. Without the buildInitial() ordering fix, i.body is
emitted after BuildRegion recurses into j.loop, so %a.row becomes a live-in
ir<> in the k-loop body, giving A-load DimSet={K only} and wrong contract rank.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fix `buildInitial()` Ordering (Section 1)

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp` lines 846–870

- [ ] **Step 1: Apply the fix**

In `TPlan.cpp`, find the block at line 846 (`if (Idx + 1 < AllLoops.size())`). The current code calls `BuildRegion(Idx + 1)` at line 851 **before** the `EmitBlock` loop at lines 864–870.

Replace this entire block:

```cpp
    if (Idx + 1 < AllLoops.size()) {
      // Non-innermost: create inner preheader + recurse.
      unsigned ChildLevel = Level - 1;
      std::string ChildStr = std::to_string(ChildLevel);
      auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);
      auto *Child = BuildRegion(Idx + 1);
      auto *MiddleBB = P.createTPBasicBlock("middle.block" + ChildStr);
```

With (move `setInsertionBB` + `EmitBlock` loop to BEFORE the recursive call):

```cpp
    if (Idx + 1 < AllLoops.size()) {
      // Non-innermost: create inner preheader + recurse.
      unsigned ChildLevel = Level - 1;
      std::string ChildStr = std::to_string(ChildLevel);
      auto *InnerPH = P.createTPBasicBlock("tensor.ph" + ChildStr);

      // Emit outer body blocks BEFORE recursing into the inner region so that
      // any values computed between the outer loop header and the inner loop
      // preheader (e.g. A's row GEP in ggml-style GEMM: %a.row = gep %A,
      // %i*K, in an i.body block between i.loop header and j.loop) are in
      // ValueMap when BuildRegion processes inner-loop GEP recipes.
      // Without this ordering those GEPs fall back to ir<> live-ins (DimSet={})
      // and BFS never propagates the outer-dim bit.
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

Then **delete** the old `InnerPH->setInsertionBB` + `EmitBlock` loop that was at lines 864–870 (they are now duplicate — only remove those two statements, leaving the rest of the block unchanged).

- [ ] **Step 2: Build**

```bash
cd /Users/yun-yugyeong/Dev/llvm
ninja -C build LLVMVectorize 2>&1 | tail -20
```

Expected: build succeeds.

- [ ] **Step 3: Run the new test — verify it now PASSES**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
```

Expected: **PASS** — `tensor.contract.2d.2d.2d.f32` found, no `tensor.contract.1d.` present.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "$(cat <<'EOF'
tplan: emit outer body blocks before recursing in buildInitial()

BuildRegion(Idx) previously called BuildRegion(Idx+1) at line 851 before
the EmitBlock loop (lines 864-870) that adds outer body blocks to ValueMap.
Blocks between the outer loop header and the inner loop (e.g. A's row-ptr
GEP in ggml-style GEMM) were not in ValueMap when the inner-loop GEP
recipes were built, so those recipes fell back to ir<> live-ins (DimSet={})
and BFS skipped them. A-load ended up K-only instead of {K, M}.

Fix: move InnerPH->setInsertionBB() + EmitBlock loop to before the
recursive BuildRegion call. Now %a.row is resolved to a TP recipe with
DimSet={M-dim} before k-loop body processing begins, so BFS correctly
gives A-load DimSet={K,M} → tensor.contract.2d.2d.2d.f32.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `PrebuiltTilingInfo` to `TPTransformState` (Section 2 prep)

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h` — `TPTransformState` struct (around line 1496)

- [ ] **Step 1: Add the struct and field**

Inside `TPTransformState`, after the `EmittedContractions` / `ContractionResults` fields and before the constructor, add:

```cpp
  /// Pre-built tiling loop skeleton populated by preBuildTilingBlocks()
  /// before execute(). When set, emitContraction() skips loop creation and
  /// instead moves to BodyBB to emit the tensor.contract call.
  struct PrebuiltTilingInfo {
    /// One entry per StaticTiled output dim, ordered outermost-first
    /// (same order as Policy.Specs iteration over non-Inline dims
    ///  excluding the contract dim).
    SmallVector<TilingLoopInfo, 4>   StaticLoops;
    /// Set when the contract dim (K) is DynamicTiled.
    Optional<FixedCountLoopInfo>     DynamicLoop;
    /// Innermost body block — where emitContraction() inserts pointer
    /// offsets and the tensor.contract call.
    BasicBlock                      *BodyBB = nullptr;
    /// Expanded TC values parallel to StaticLoops (used for pointer offset
    /// ActualSize computation in emitContraction).
    SmallVector<Value *, 4>          TCValues;
  };
  Optional<PrebuiltTilingInfo> PrebuiltTiling;
```

`TilingLoopInfo` and `FixedCountLoopInfo` are defined in `TPlanLowering.cpp`. Forward-declare them in `TPlan.h` if needed, or move the struct definitions to `TPlanLowering.h` / keep `PrebuiltTilingInfo` in `TPlanLowering.cpp` and add a forward pointer in `TPTransformState` using `void *`.

**Simpler alternative** (preferred — avoids header dependency): keep `PrebuiltTilingInfo` as a file-local struct in `TPlanLowering.cpp`, add only a `void *PrebuiltTilingPtr = nullptr` opaque pointer to `TPTransformState`, and cast it in `preBuildTilingBlocks()` / `emitContraction()`. This keeps the TPlan header free of IR types.

Concrete approach: add this to `TPTransformState` in `TPlan.h`:

```cpp
  /// Opaque pointer to a PrebuiltTilingInfo allocated by preBuildTilingBlocks()
  /// (defined in TPlanLowering.cpp). Null when not pre-built.
  void *PrebuiltTilingPtr = nullptr;
```

- [ ] **Step 2: Build to verify no compilation errors**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build (field is just a null pointer, unused yet).

- [ ] **Step 3: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "$(cat <<'EOF'
tplan: add PrebuiltTilingPtr opaque field to TPTransformState

Opaque void* to hold a PrebuiltTilingInfo allocated by the upcoming
preBuildTilingBlocks() function in TPlanLowering.cpp. Keeping it opaque
avoids pulling IR types (BasicBlock*, PHINode*) into TPlan.h.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement `preBuildTilingBlocks()` (Section 2)

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

`preBuildTilingBlocks()` is called in `TPlanLowering_lower()` AFTER `createTensorizedLoopSkeleton()` (so the guard + tensor path entry exist) but BEFORE `execute()`. It builds the tiling loop skeleton inside the tensor path and stores it in `State.PrebuiltTilingPtr`.

- [ ] **Step 1: Define `PrebuiltTilingInfo` struct (file-local, above `emitContraction`)**

Add this near the `TilingLoopInfo` struct definition (around line 404):

```cpp
/// Tiling loop skeleton pre-built by preBuildTilingBlocks() before execute().
/// Stored via State.PrebuiltTilingPtr (opaque void*); cast in callers.
struct PrebuiltTilingInfo {
  /// Static tiling loops for output dims (M, N, ...), outermost first.
  SmallVector<TilingLoopInfo, 4>  StaticLoops;
  /// Fixed-count tiling loop for the dynamic contract dim (K).
  Optional<FixedCountLoopInfo>    DynamicLoop;
  /// Innermost body BB — where tensor.contract is emitted.
  BasicBlock                     *BodyBB    = nullptr;
  /// Expanded TC value* parallel to StaticLoops (i64).
  SmallVector<Value *, 4>         TCValues;
  /// PF parallel to StaticLoops (used for ActualSize arg).
  SmallVector<unsigned, 4>        PFs;
  /// Dim indices parallel to StaticLoops.
  SmallVector<unsigned, 4>        Dims;
  /// Dynamic loop dim index and PF (for K pointer offset).
  unsigned                        DynDim = 0;
  unsigned                        DynPF  = 0;

  ~PrebuiltTilingInfo() = default;
};
```

- [ ] **Step 2: Implement `preBuildTilingBlocks()`**

Add this function before `TPlanLowering_lower()` (around line 1725):

```cpp
/// Scans the ClassMap for Contraction recipes, determines OutputDimSet and
/// ContractDim, then builds tiling loop block skeletons for all non-Inline
/// Policy dims BEFORE execute(). On success, allocates a PrebuiltTilingInfo
/// and stores it in State.PrebuiltTilingPtr.
///
/// \param InsertBB  The block to use as preheader for the outermost tiling
///                  loop. Must be on the tensor path (after the guard) and
///                  have no terminator yet (or the caller appends after it).
static void preBuildTilingBlocks(BasicBlock *InsertBB,
                                  TPTransformState &State,
                                  const TPlan &Plan,
                                  const RecipeClassMap &CM,
                                  ScalarEvolution &SE) {
  if (!InsertBB)
    return;

  // Step A: Find any Contraction recipe to get OutputDimSet + ContractDim.
  int ContractDim = -1;
  SmallBitVector LHSBits, RHSBits;
  for (const auto &[R, Class] : CM) {
    if (Class.Kind != TensorOpKind::Contraction)
      continue;
    ContractDim = Class.ContractDim;
    if (auto *FMul = Class.FusedMulRecipe) {
      if (auto *L = dyn_cast<TPSingleDefRecipe>(FMul->getOperand(0)))
        LHSBits = L->DimSet;
      if (auto *R2 = dyn_cast<TPSingleDefRecipe>(FMul->getOperand(1)))
        RHSBits = R2->DimSet;
    }
    break; // one contraction per nest for now
  }
  if (ContractDim < 0)
    return; // nothing to pre-build

  // Build OutputDimSet = (LHS | RHS) - {ContractDim}.
  unsigned NBits = std::max(LHSBits.size(), RHSBits.size());
  LHSBits.resize(NBits); RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));

  // Step B: Collect StaticTiled output dims and DynamicTiled contract dim.
  SCEVExpander Exp(SE, "tplan.prebuild");
  IRBuilder<> B(InsertBB, InsertBB->end());
  auto *Info = new PrebuiltTilingInfo();

  // Output dims (StaticTiled), outermost first (highest DimIdx first).
  for (int D = static_cast<int>(NBits) - 1; D >= 0; --D) {
    if (!OutputDimSet.test(static_cast<unsigned>(D))) continue;
    const DimEmissionSpec *Spec = State.Policy.getSpec(static_cast<unsigned>(D));
    if (!Spec || Spec->Mode != DimEmitMode::StaticTiled) continue;
    const SCEV *BTC = Plan.getTCForDim(static_cast<unsigned>(D));
    if (!BTC) continue;
    Value *BTCVal = Exp.expandCodeFor(BTC, B.getInt64Ty(), &*B.GetInsertPoint());
    Value *TCVal  = B.CreateAdd(BTCVal, B.getInt64(1), "tc.d" + Twine(D));
    Value *PFVal  = B.getInt64(Spec->PF);
    TilingLoopInfo LI = emitTilingLoop(B, TCVal, PFVal,
                                        "tile.d" + Twine(D));
    Info->StaticLoops.push_back(LI);
    Info->TCValues.push_back(TCVal);
    Info->PFs.push_back(Spec->PF);
    Info->Dims.push_back(static_cast<unsigned>(D));
    // B is now inside the body of this loop; next iteration nests inside.
  }

  // Contract dim (DynamicTiled).
  {
    const DimEmissionSpec *Spec = State.Policy.getSpec(
        static_cast<unsigned>(ContractDim));
    if (Spec && Spec->Mode == DimEmitMode::DynamicTiled) {
      const SCEV *BTC = Plan.getTCForDim(static_cast<unsigned>(ContractDim));
      if (BTC) {
        Value *BTCVal = Exp.expandCodeFor(BTC, B.getInt64Ty(),
                                           &*B.GetInsertPoint());
        Value *TCVal = B.CreateAdd(BTCVal, B.getInt64(1), "k.tc");
        unsigned PF = Spec->PF;
        Info->DynamicLoop = emitFixedCountingLoop(B, TCVal, PF, "tensor.body");
        Info->DynDim = static_cast<unsigned>(ContractDim);
        Info->DynPF  = PF;
        // B is now in tensor.body.body.
      }
    }
  }

  Info->BodyBB = B.GetInsertBlock();
  State.PrebuiltTilingPtr = Info;
}
```

- [ ] **Step 3: Call `preBuildTilingBlocks()` in `TPlanLowering_lower()`**

In `TPlanLowering_lower()`, after the existing `createTensorizedLoopSkeleton()` block (lines 1756–1810) and before the `execute()` loop (line 1813), add:

```cpp
  // Section 2: Pre-build tiling loop skeletons inside the tensor path before
  // execute() runs. emitContraction() will later fill in the body block.
  if (State.Policy.needsGuard()) {
    // Find the tensor path entry (outermost loop preheader on tensor side).
    // After createTensorizedLoopSkeleton(), OutermostLoop's preheader is on
    // the tensor path and has been redirected by the guard.
    Loop *OutermostLoop = nullptr;
    unsigned MaxDim = 0;
    for (const auto &[D, L] : State.DimToLoop) {
      if (!OutermostLoop || D > MaxDim) { MaxDim = D; OutermostLoop = L; }
    }
    if (OutermostLoop) {
      BasicBlock *TensorPH = OutermostLoop->getLoopPreheader();
      preBuildTilingBlocks(TensorPH, State, Plan, CM, SE);
    }
  }
```

- [ ] **Step 4: Build and verify no crashes**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -10
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/outer-dim-preheader-gep.ll
```

Expected: build clean; test still passes (emitContraction() not yet simplified, old path still active for now).

- [ ] **Step 5: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "$(cat <<'EOF'
tplan-lower: pre-build tiling loop skeletons before execute() (Section 2)

Add preBuildTilingBlocks() which scans the ClassMap for a Contraction recipe,
derives OutputDimSet + ContractDim, then calls emitTilingLoop() for each
StaticTiled output dim (M, N, outermost-first) and emitFixedCountingLoop()
for the DynamicTiled contract dim (K). The resulting loop skeleton is stored
in State.PrebuiltTilingPtr before execute() begins.

emitContraction() is not yet simplified — that is Task 5.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Simplify `emitContraction()` to Use Pre-Built Body (Section 3)

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` — `emitContraction()` (lines 535–1174)

When `State.PrebuiltTilingPtr` is set, `emitContraction()` skips loop creation and directly uses the pre-built body BB.

- [ ] **Step 1: Add early-exit path at the start of `emitContraction()`'s tiling section**

After the `NeedsTiling` / `NeedsStaticTiling` / `NeedsDynamicTiling` booleans are computed (around line 768) and before the static/dynamic tiling paths (line 871), add:

```cpp
  // ----------------------------------------------------------------
  // Section 3 fast path: tiling skeleton was pre-built by preBuildTilingBlocks().
  // Move to the pre-built body BB and emit only pointer offsets + call.
  // ----------------------------------------------------------------
  if (NeedsTiling && State.PrebuiltTilingPtr) {
    auto *Info = static_cast<PrebuiltTilingInfo *>(State.PrebuiltTilingPtr);
    if (!Info->BodyBB)
      goto scalar_fallback; // defensive

    // Erase the current block's original terminator (scalar loop exit branch).
    OrigTerm->eraseFromParent();
    // Remove self-loop PHI incoming from the header (same as existing paths).
    for (PHINode &Phi : LoopHeaderBB->phis()) {
      int Idx = Phi.getBasicBlockIndex(LoopHeaderBB);
      if (Idx >= 0)
        Phi.removeIncomingValue(Idx, /*DeletePHIIfEmpty=*/false);
    }

    // Connect preheader to the outermost pre-built loop (or directly to BodyBB
    // if only a DynamicLoop was built).
    if (!Info->StaticLoops.empty()) {
      // StaticLoops[0] was built with LoopHeaderBB as preheader in preBuildTilingBlocks.
      // Just set B to the inner-most static body (which is also inside DynamicLoop if present).
    }

    // Move B to the body.
    B.SetInsertPoint(Info->BodyBB);

    // STEP C (from static path): compute per-StaticTiled-dim offset pointers.
    Value *TiledCPtr = CPtr, *TiledAPtr = LHSPtr, *TiledBPtr = RHSPtr;
    for (unsigned I = 0; I < Info->StaticLoops.size(); ++I) {
      unsigned Dim  = Info->Dims[I];
      Value   *IV   = Info->StaticLoops[I].IV;
      auto OffsetPtr = [&](Value *Base, Value *Stride) -> Value * {
        if (auto *CI = dyn_cast<ConstantInt>(Stride); CI && CI->isZero())
          return Base;
        return B.CreateGEP(ElemTy, Base,
                           B.CreateMul(IV, Stride, "tile.off"), "tile.ptr");
      };
      TiledAPtr = OffsetPtr(TiledAPtr, getAStride(Dim));
      TiledBPtr = OffsetPtr(TiledBPtr, getBStride(Dim));
      TiledCPtr = OffsetPtr(TiledCPtr, getCStride(Dim));
    }

    // STEP D (from static path): If dynamic K loop, compute K-offset inside body.
    if (Info->DynamicLoop) {
      Value *KIV    = Info->DynamicLoop->IV;
      Value *AKOff  = B.CreateMul(KIV, CachedAContractStride, "tensor.body.a.off");
      TiledAPtr     = B.CreateGEP(ElemTy, TiledAPtr, AKOff, "tensor.body.a.ptr");
      Value *BKOff  = B.CreateMul(KIV, CachedBContractStride, "tensor.body.b.off");
      TiledBPtr     = B.CreateGEP(ElemTy, TiledBPtr, BKOff, "tensor.body.b.ptr");
    }

    // Build args and emit tensor.contract call.
    SmallVector<Value *> Args;
    Args.push_back(TiledCPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].CStr);
    Args.push_back(TiledAPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].AStr);
    Args.push_back(CachedAContractStride);
    Args.push_back(TiledBPtr);
    for (unsigned SI = 0; SI < OutputStrides.size(); ++SI)
      Args.push_back(OutputStrides[SI].BStr);
    Args.push_back(CachedBContractStride);
    // K size: fixed PF for DynamicTiled, else getRealDim.
    Value *KArg = Info->DynamicLoop
                      ? B.getInt64(Info->DynPF)
                      : getRealDim(static_cast<unsigned>(ContractDim));
    Args.push_back(KArg);
    for (int D = OutputDimSet.find_first(); D >= 0; D = OutputDimSet.find_next(D))
      Args.push_back(Info->StaticLoops.empty()
                         ? getRealDim(static_cast<unsigned>(D))
                         : Info->StaticLoops[/* find index for D */0].ActualSize);
    Value *Call = B.CreateCall(ContractFn, Args);

    // Close dynamic loop (br to latch), move to exit.
    if (Info->DynamicLoop) {
      B.CreateBr(Info->DynamicLoop->LatchBB);
      B.SetInsertPoint(Info->DynamicLoop->ExitBB);
      // scalar.block remainder handled by existing code in Info (if needed).
    }

    // Close static loops in reverse (innermost first).
    for (int I = static_cast<int>(Info->StaticLoops.size()) - 1; I >= 0; --I) {
      B.CreateBr(Info->StaticLoops[I].LatchBB);
      B.SetInsertPoint(Info->StaticLoops[I].ExitBB);
    }

    if (OrigSuccessor)
      B.CreateBr(OrigSuccessor);

    return Call;
  }
  scalar_fallback:
```

> **Note:** The exact ActualSize indexing and scalar-block remainder wiring for the dynamic path will need to be filled in during implementation based on how `FixedCountLoopInfo` exposes its exit IV. Reference the existing DynamicTiledPath (lines 970–1171) for the scalar.block pattern.

- [ ] **Step 2: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 3: Run all LoopTensorize tests**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass including `outer-dim-preheader-gep.ll`.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "$(cat <<'EOF'
tplan-lower: simplify emitContraction() to use pre-built body BB (Section 3)

When State.PrebuiltTilingPtr is set (pre-built by preBuildTilingBlocks()),
emitContraction() skips emitTilingLoop() / emitFixedCountingLoop() and
instead moves directly to the pre-built BodyBB. It computes tile pointer
offsets using the pre-built loop IVs (from StaticLoops + DynamicLoop) and
emits the tensor.contract call, then closes the pre-built loops in reverse.

This separates tiling structure (decided by Policy before execute) from
tiling content (tensor.contract args decided by DimSet during execute).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full Regression Suite

- [ ] **Step 1: Run the full LoopTensorize lit suite**

```bash
cd /Users/yun-yugyeong/Dev/llvm
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Key tests to watch for regressions:

| Test | What it covers |
|------|---------------|
| `basic/tensor-contract-gemm-2d.ll` | Flat GEP 2D GEMM → `2d.2d.2d` |
| `basic/pf-dimset-gemm.ll` | DimSet propagation baseline |
| `basic/ptr-decompose-srem.ll` | srem stop + outer body GEP (batched) |
| `basic/tensor-contract-batched.ll` | Batched contraction ranks |
| `basic/skeleton-guard.ll` | Guard + dynamic-K tiling |
| `basic/tiling-dynamic-multidim.ll` | Multi-dim tiling |
| `basic/tiling-pf8.ll` | PF=8 static tiling |
| `remainder/non-divisible-tripcount.ll` | Non-divisible remainder path |
| `basic/outer-dim-preheader-gep.ll` | New test (must pass) |

- [ ] **Step 2: If a DimSet-related test fails, diagnose with debug output**

```bash
./build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  -S --disable-verify \
  llvm/test/Transforms/LoopTensorize/basic/<failing>.ll 2>&1 | head -100
```

Look for:
- `WIDEN-GEP` DimSet after BFS — check M/N dims appear
- `Contraction (contractDim=X)` — ContractDim must be in both A and B DimSets
- `TPlanLowering: dim=X mode=... PF=...` — Policy classification correct?
- `PrebuiltTilingInfo: BodyBB=null` — pre-build failed, check CM scan

- [ ] **Step 3: Commit any regression fixes found**

```bash
git add <fixed-files>
git commit -m "$(cat <<'EOF'
tplan-lower: fix regression in <test> after pre-build + buildInitial changes

<Describe the specific issue and fix>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- [x] Section 1 (buildInitial ordering fix) → Task 2
- [x] Section 2 (pre-build tiling skeletons before execute) → Tasks 3–4
- [x] Section 3 (emitContraction simplified) → Task 5
- [x] TDD: failing test first → Task 1, passes after Task 2
- [x] Regression gate → Task 6

**Placeholder scan:**
- Task 5 Step 1 contains a note about `ActualSize` indexing needing care during implementation — the pattern is fully referenced to existing code (lines 970–1171 DynamicTiledPath). Not a blocker; it is a genuine implementation detail to resolve during coding.

**Type consistency:**
- `PrebuiltTilingInfo` defined in Task 4 Step 1, used in Task 5 Step 1 — same struct name/fields throughout.
- `TilingLoopInfo`, `FixedCountLoopInfo` — used in Task 4 and Task 5, defined at lines 405 / 333 of `TPlanLowering.cpp`.
- `State.PrebuiltTilingPtr` — added in Task 3, populated in Task 4, consumed in Task 5.
