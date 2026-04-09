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
  /// Floating preheader block used during pre-build; replaced by LoopHeaderBB
  /// and erased in emitContraction(). Its only instruction is br FirstHeader.
  BasicBlock                     *AnchorBB  = nullptr;
  /// Expanded TC value* parallel to StaticLoops (i64).
  SmallVector<Value *, 4>         TCValues;
  /// PF parallel to StaticLoops (used for ActualSize arg).
  SmallVector<unsigned, 4>        PFs;
  /// Dim indices parallel to StaticLoops.
  SmallVector<unsigned, 4>        Dims;
  /// Dynamic loop dim index and PF (for K pointer offset).
  unsigned                        DynDim = 0;
  unsigned                        DynPF  = 0;
  /// Expanded real trip-count for the dynamic dim (needed for scalar.block).
  Value                          *DynTCVal = nullptr;

  ~PrebuiltTilingInfo() = default;
};
```

- [ ] **Step 2: Implement `preBuildTilingBlocks()`**

Add this function before `TPlanLowering_lower()` (around line 1725):

```cpp
/// Scans the ClassMap for Contraction recipes, determines OutputDimSet and
/// ContractDim, then builds FLOATING tiling loop block skeletons for all
/// non-Inline Policy dims BEFORE execute(). On success, allocates a
/// PrebuiltTilingInfo and stores it in State.PrebuiltTilingPtr.
///
/// Design — AnchorBB pattern:
///   1. SCEV expansion instructions land in InsertBB (TensorPH), BEFORE its
///      terminator. InsertBB is already in the dominator tree, so these Values
///      dominate all tiling loop blocks once emitContraction() connects them.
///   2. A floating AnchorBB (no predecessors yet) is the temporary preheader
///      for emitTilingLoop / emitFixedCountingLoop. AnchorBB's only instruction
///      is the br to the first tiling header.
///   3. In emitContraction(): B.CreateBr(FirstHeader) + replaceIncomingBlockWith
///      (AnchorBB → LoopHeaderBB) + AnchorBB->eraseFromParent() wires the
///      pre-built structure into the live CFG.
///
/// \param InsertBB  Tensor path preheader (OutermostLoop->getLoopPreheader()).
///                  Must have a terminator; SCEV expansions are inserted before
///                  it. The tiling loops themselves go into a separate AnchorBB.
static void preBuildTilingBlocks(BasicBlock *InsertBB,
                                  TPTransformState &State,
                                  const TPlan &Plan,
                                  const RecipeClassMap &CM) {
  if (!InsertBB)
    return;
  Instruction *InsertBefore = InsertBB->getTerminator();
  if (!InsertBefore)
    return; // InsertBB must be terminated (it still has br OutermostLoop)
  Function *F = InsertBB->getParent();

  // Step A: Find any Contraction recipe to get OutputDimSet + ContractDim.
  int ContractDim = -1;
  SmallBitVector LHSBits, RHSBits;
  for (const auto &[R, Class] : CM) {
    if (Class.Kind != TensorOpKind::Contraction)
      continue;
    ContractDim = Class.ContractDim;
    if (Class.FusedMulRecipe) {
      if (auto *L = dyn_cast<TPSingleDefRecipe>(
              Class.FusedMulRecipe->getOperand(0)))
        LHSBits = L->DimSet;
      if (auto *R2 = dyn_cast<TPSingleDefRecipe>(
              Class.FusedMulRecipe->getOperand(1)))
        RHSBits = R2->DimSet;
    }
    break; // one contraction per nest for now
  }
  if (ContractDim < 0)
    return; // nothing to pre-build

  unsigned NBits = std::max(LHSBits.size(), RHSBits.size());
  if (NBits == 0) return;
  LHSBits.resize(NBits); RHSBits.resize(NBits);
  SmallBitVector OutputDimSet = LHSBits;
  OutputDimSet |= RHSBits;
  OutputDimSet.reset(static_cast<unsigned>(ContractDim));

  // Step B: Expand all trip-count SCEVs into InsertBB BEFORE its terminator.
  // For SCEVConstant / SCEVUnknown (function args), expandCodeFor returns the
  // constant or the original Value* with no new instructions. For AddRec-
  // derived BTC (e.g. %K-1 for a loop bound), it emits a sub into InsertBB.
  // Either way, the result dominates all tiling loop blocks in the final CFG
  // (InsertBB = TensorPH dominates the entire tensor path).
  auto *Info = new PrebuiltTilingInfo();
  LLVMContext &Ctx = F->getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  IRBuilder<> PH(InsertBefore); // inserts before InsertBefore (the br)

  // Output dims (StaticTiled), outermost first (highest DimIdx first).
  for (int D = static_cast<int>(NBits) - 1; D >= 0; --D) {
    if (!OutputDimSet.test(static_cast<unsigned>(D))) continue;
    const DimEmissionSpec *Spec =
        State.Policy.getSpec(static_cast<unsigned>(D));
    if (!Spec || Spec->Mode != DimEmitMode::StaticTiled) continue;
    const SCEV *BTC = Plan.getTCForDim(static_cast<unsigned>(D));
    if (!BTC) continue;
    Value *BTCVal =
        State.Expander->expandCodeFor(BTC, I64, InsertBefore);
    Value *TCVal = PH.CreateAdd(BTCVal, PH.getInt64(1),
                                "tc.d" + Twine(D));
    Info->TCValues.push_back(TCVal);
    Info->PFs.push_back(Spec->PF);
    Info->Dims.push_back(static_cast<unsigned>(D));
  }

  // Dynamic contract dim (K).
  Value *DynTCVal = nullptr;
  {
    const DimEmissionSpec *Spec =
        State.Policy.getSpec(static_cast<unsigned>(ContractDim));
    if (Spec && Spec->Mode == DimEmitMode::DynamicTiled) {
      const SCEV *BTC = Plan.getTCForDim(static_cast<unsigned>(ContractDim));
      if (BTC) {
        Value *BTCVal =
            State.Expander->expandCodeFor(BTC, I64, InsertBefore);
        DynTCVal = PH.CreateAdd(BTCVal, PH.getInt64(1), "k.tc");
        Info->DynDim    = static_cast<unsigned>(ContractDim);
        Info->DynPF     = Spec->PF;
        Info->DynTCVal  = DynTCVal;
      }
    }
  }

  // Step C: Build floating tiling loop skeletons from a temporary AnchorBB.
  // AnchorBB has no predecessors. Its only instruction will be the br to the
  // first tiling header, created by emitTilingLoop(). In emitContraction(),
  // LoopHeaderBB's new br replaces this edge and AnchorBB is deleted.
  BasicBlock *AnchorBB = BasicBlock::Create(Ctx, "tiling.anchor", F);
  Info->AnchorBB = AnchorBB;
  IRBuilder<> TileB(AnchorBB);

  // Static loops — outermost first (Dims list is already highest-D-first).
  for (unsigned I = 0; I < Info->Dims.size(); ++I) {
    std::string Name = (Twine("tile.d") + Twine(Info->Dims[I])).str();
    Value *TileSize  = TileB.getInt64(Info->PFs[I]);
    TilingLoopInfo LI =
        emitTilingLoop(TileB, Info->TCValues[I], TileSize, Name);
    Info->StaticLoops.push_back(LI);
    // TileB is now in LI's body BB; next iteration nests inside.
  }

  // Dynamic loop (K), if applicable.
  if (DynTCVal) {
    FixedCountLoopInfo FCI =
        emitFixedCountingLoop(TileB, DynTCVal, Info->DynPF, "tensor.body");
    Info->DynamicLoop = FCI;
    // TileB is now in tensor.body.body.
  }

  Info->BodyBB = TileB.GetInsertBlock();
  State.PrebuiltTilingPtr = Info;
}
```

- [ ] **Step 3: Call `preBuildTilingBlocks()` in `TPlanLowering_lower()`**

In `TPlanLowering_lower()`, after the existing `createTensorizedLoopSkeleton()` block (lines 1756–1810) and before the `execute()` loop (line 1813), add:

```cpp
  // Section 2: Pre-build floating tiling loop skeletons before execute().
  // State.Expander is already initialized; InsertBB is the tensor path
  // preheader (OutermostLoop->getLoopPreheader()). SCEV expansions land
  // in InsertBB (before its terminator); loop blocks float off AnchorBB
  // until emitContraction() wires them into the live CFG.
  if (State.Policy.needsGuard()) {
    Loop *OutermostLoop = nullptr;
    unsigned MaxDim = 0;
    for (const auto &[D, L] : State.DimToLoop) {
      if (!OutermostLoop || D > MaxDim) { MaxDim = D; OutermostLoop = L; }
    }
    if (OutermostLoop) {
      BasicBlock *TensorPH = OutermostLoop->getLoopPreheader();
      preBuildTilingBlocks(TensorPH, State, Plan, CM);
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

- [ ] **Step 1: Add pre-built path at the start of `emitContraction()`'s tiling section**

After the `NeedsTiling` / `NeedsStaticTiling` / `NeedsDynamicTiling` booleans are computed (around line 768) and after `OrigTerm` / `OrigSuccessor` are set up (around line 869), add the pre-built path **before** the `if (NeedsStaticTiling && !NeedsDynamicTiling)` check (line 871).

```cpp
  // ----------------------------------------------------------------
  // Section 3 fast path: tiling skeleton was pre-built by preBuildTilingBlocks().
  // Wire AnchorBB → LoopHeaderBB, move to BodyBB, emit pointer offsets + call,
  // close pre-built loops, emit scalar.block remainder if dynamic K.
  // ----------------------------------------------------------------
  {
    auto *Info = NeedsTiling
                     ? static_cast<PrebuiltTilingInfo *>(State.PrebuiltTilingPtr)
                     : nullptr;
    if (Info && Info->BodyBB) {
      // Erase the scalar loop terminator (same as existing tiling paths).
      OrigTerm->eraseFromParent();
      // Remove k-loop self-edge PHI incomings from LoopHeaderBB.
      for (PHINode &Phi : LoopHeaderBB->phis()) {
        int SelfIdx = Phi.getBasicBlockIndex(LoopHeaderBB);
        if (SelfIdx >= 0)
          Phi.removeIncomingValue(SelfIdx, /*DeletePHIIfEmpty=*/false);
      }

      // Connect LoopHeaderBB → first tiling header by replacing AnchorBB's role.
      // The first tiling header's IV PHI has incoming(0, AnchorBB); we redirect
      // that to LoopHeaderBB and erase AnchorBB (which held only the br).
      if (!Info->StaticLoops.empty()) {
        BasicBlock *FirstHeader = Info->StaticLoops[0].IV->getParent();
        B.CreateBr(FirstHeader);
        Info->StaticLoops[0].IV->replaceIncomingBlockWith(Info->AnchorBB,
                                                           LoopHeaderBB);
      } else {
        // Dynamic-only case: connect directly to tensor.body header.
        assert(Info->DynamicLoop.has_value());
        BasicBlock *DynHeader = Info->DynamicLoop->IV->getParent();
        B.CreateBr(DynHeader);
        Info->DynamicLoop->IV->replaceIncomingBlockWith(Info->AnchorBB,
                                                         LoopHeaderBB);
      }
      Info->AnchorBB->eraseFromParent(); // safe: no predecessors remain

      // Position B in the pre-built innermost body block.
      B.SetInsertPoint(Info->BodyBB);

      // Build dim → StaticLoops index mapping for ActualSize lookup below.
      DenseMap<unsigned, unsigned> DimToLoopIdx;
      for (unsigned I = 0; I < Info->Dims.size(); ++I)
        DimToLoopIdx[Info->Dims[I]] = I;

      // STEP C': Compute tile-offset base pointers using static loop IVs.
      // For each StaticTiled output dim I: tiled_ptr = base + IV[I] * stride[dim].
      Value *TiledCPtr = CPtr, *TiledAPtr = LHSPtr, *TiledBPtr = RHSPtr;
      for (unsigned I = 0; I < Info->StaticLoops.size(); ++I) {
        unsigned Dim = Info->Dims[I];
        Value   *IV  = Info->StaticLoops[I].IV;
        auto OffsetPtr = [&](Value *Base, Value *Stride) -> Value * {
          if (auto *CI = dyn_cast<ConstantInt>(Stride); CI && CI->isZero())
            return Base;
          Value *Off = B.CreateMul(IV, Stride, "tile.off");
          return B.CreateGEP(ElemTy, Base, Off, "tile.ptr");
        };
        TiledAPtr = OffsetPtr(TiledAPtr, getAStride(Dim));
        TiledBPtr = OffsetPtr(TiledBPtr, getBStride(Dim));
        TiledCPtr = OffsetPtr(TiledCPtr, getCStride(Dim));
      }

      // STEP D': If dynamic K loop, offset A/B by K-tile IV inside body.
      if (Info->DynamicLoop) {
        Value *KIV   = Info->DynamicLoop->IV;
        Value *AKOff = B.CreateMul(KIV, CachedAContractStride,
                                    "tensor.body.a.off");
        TiledAPtr    = B.CreateGEP(ElemTy, TiledAPtr, AKOff,
                                    "tensor.body.a.ptr");
        Value *BKOff = B.CreateMul(KIV, CachedBContractStride,
                                    "tensor.body.b.off");
        TiledBPtr    = B.CreateGEP(ElemTy, TiledBPtr, BKOff,
                                    "tensor.body.b.ptr");
      }

      // Build tensor.contract args.
      SmallVector<Value *> Args;
      Args.push_back(TiledCPtr);
      for (auto &S : OutputStrides) Args.push_back(S.CStr);
      Args.push_back(TiledAPtr);
      for (auto &S : OutputStrides) Args.push_back(S.AStr);
      Args.push_back(CachedAContractStride);
      Args.push_back(TiledBPtr);
      for (auto &S : OutputStrides) Args.push_back(S.BStr);
      Args.push_back(CachedBContractStride);
      // K: fixed DynPF for dynamic loop, else full real dim.
      Args.push_back(Info->DynamicLoop ? B.getInt64(Info->DynPF)
                                       : getRealDim(ContUD));
      // Output dims: ActualSize from the tiling loop if that dim was statically
      // tiled (min(PF, TC - IV)); otherwise the full real dim.
      for (int D = OutputDimSet.find_first(); D >= 0;
           D = OutputDimSet.find_next(D)) {
        unsigned UD = static_cast<unsigned>(D);
        auto It = DimToLoopIdx.find(UD);
        Args.push_back(It != DimToLoopIdx.end()
                           ? Info->StaticLoops[It->second].ActualSize
                           : getRealDim(UD));
      }
      Value *Call = B.CreateCall(ContractFn, Args);

      // Close dynamic K loop and emit scalar.block remainder.
      if (Info->DynamicLoop) {
        B.CreateBr(Info->DynamicLoop->LatchBB);
        B.SetInsertPoint(Info->DynamicLoop->ExitBB);

        // scalar.block: iterate the remaining K elements one-by-one.
        // EpiStart = ExitIV of the fixed-count loop = number of K elements
        // consumed by full tiles. TcVal = total K trip count.
        Value *EpiStart = Info->DynamicLoop->ExitIV;
        Value *TcVal    = Info->DynTCVal; // pre-expanded in preBuildTilingBlocks
        Value *ScRem    = B.CreateSub(TcVal, EpiStart, "scalar.rem");
        Value *HasSc    = B.CreateICmpUGT(ScRem, B.getInt64(0), "scalar.guard");

        BasicBlock *ScPHBB   = B.GetInsertBlock();
        BasicBlock *ScBodyBB = BasicBlock::Create(
            B.getContext(), "scalar.block", ScPHBB->getParent());
        BasicBlock *ScExitBB = BasicBlock::Create(
            B.getContext(), "scalar.block.exit", ScPHBB->getParent());
        B.CreateCondBr(HasSc, ScBodyBB, ScExitBB);

        B.SetInsertPoint(ScBodyBB);
        PHINode *ScIV = B.CreatePHI(B.getInt64Ty(), 2, "scalar.iv");
        ScIV->addIncoming(EpiStart, ScPHBB);
        // Load C, A[k], B[k]; fmul; fadd; store.
        // TiledCPtr / TiledAPtr / TiledBPtr are the M/N-tile-offset pointers
        // computed above (already reflect the current M and N tile offsets).
        Value *CVal  = B.CreateLoad(ElemTy, TiledCPtr, "scalar.c");
        Value *SAOff = B.CreateMul(ScIV, CachedAContractStride, "scalar.a.off");
        Value *SAPtr = B.CreateGEP(ElemTy, TiledAPtr, SAOff, "scalar.a.ptr");
        Value *SAVal = B.CreateLoad(ElemTy, SAPtr, "scalar.a");
        Value *SBOff = B.CreateMul(ScIV, CachedBContractStride, "scalar.b.off");
        Value *SBPtr = B.CreateGEP(ElemTy, TiledBPtr, SBOff, "scalar.b.ptr");
        Value *SBVal = B.CreateLoad(ElemTy, SBPtr, "scalar.b");
        Value *SProd = B.CreateFMul(SAVal, SBVal, "scalar.mul");
        Value *SSum  = B.CreateFAdd(CVal, SProd, "scalar.sum");
        B.CreateStore(SSum, TiledCPtr);
        Value *ScNext = B.CreateAdd(ScIV, B.getInt64(1), "scalar.next");
        ScIV->addIncoming(ScNext, ScBodyBB);
        Value *ScDone = B.CreateICmpUGE(ScNext, TcVal, "scalar.done");
        B.CreateCondBr(ScDone, ScExitBB, ScBodyBB);

        B.SetInsertPoint(ScExitBB);
        Call = nullptr; // C updated in-place; no SSA value to return.
      }

      // Close static loops in reverse (innermost first).
      for (int I = static_cast<int>(Info->StaticLoops.size()) - 1; I >= 0;
           --I) {
        B.CreateBr(Info->StaticLoops[I].LatchBB);
        B.SetInsertPoint(Info->StaticLoops[I].ExitBB);
      }

      if (OrigSuccessor)
        B.CreateBr(OrigSuccessor);

      return Call;
    } // end if (Info && Info->BodyBB)
  } // end pre-built block
  // Fall through to existing static / dynamic tiling paths if no pre-built info.
```

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
