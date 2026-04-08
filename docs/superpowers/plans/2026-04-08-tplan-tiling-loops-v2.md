# TPlan Tiling Loop Generation Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When TripCount > PF for any dimension, emit tiling loops (`for tile = 0; tile < TC; tile += PF`) around PF-sized `tensor.contract` calls, with `min(PF, TC - tile)` remainder handling.

**Architecture:** Four tasks in dependency order. Task 1 adds `hasDimPF()` to TPlan and fixes the widener so pre-set PFs are respected. Task 2 adds a `-loop-tensorize-pf` CLI override for testing. Task 3 implements the tiling loop emission in `emitContraction()`, with a critical ordering rule: expand all TripCounts to `Value *` before creating any loop BBs. Task 4 adds TDD-style tests (write failing test → implement → verify).

**Tech Stack:** LLVM IR (IRBuilder), ScalarEvolution (SCEV, SCEVExpander), TPlan infrastructure (`TPlanLowering.cpp`, `TPlan.h`, `TPlan.cpp`, `TPlanWidener.cpp`, `LoopTensorize.cpp`)

**Key design decisions (from v1 review):**
- `hasDimPF()` guard in widener: pre-set PFs are not overwritten by the default PF=256.
- Tiling decision block placed **after** stride lambdas (~line 508), not before.
- All TripCounts expanded to `Value *` in one pass **before** any `emitTilingLoop()` call. This avoids SCEVExpander inserting code into the wrong BB.
- `const SCEV *` lifetime is safe: SCEVs are valid throughout a single-pass invocation of LoopTensorize; SE is not invalidated between `buildInitial()` and lowering.
- Tests use named-block CHECKs (`tile.d0.header:`, etc.) instead of `CHECK-NOT: br i1`.

---

### Task 1: Add `hasDimPF()` + `DimBackedgeTakenMap` to TPlan; fix widener

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h:1322-1394`
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp:738-740`
- Modify: `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp:63-70`

#### Why this order?

The widener (`TPlanWidener.cpp:69`) unconditionally calls `Plan.setDimPF(Dim, 256u)`, which overwrites any PF set before `TPlanLowering_lower()`. Adding `hasDimPF()` lets the widener skip dims that already have a PF, enabling external overrides (Task 2).

Separately, `TPlan.cpp:739-740` has a `(void)TC;` placeholder for future TripCount storage. We fill it in here. We store `const SCEV *` (backedge-taken count, i.e., real trip count minus 1); the +1 correction is applied at expansion time in Task 3.

- [ ] **Step 1: Add `hasDimPF()`, `setDimTC()`, `getTCForDim()` to TPlan.h**

In `llvm/include/llvm/Transforms/Vectorize/TPlan.h`, after `setDimPF()` (line 1327), add:

```cpp
  /// Returns true if a parallel factor has been explicitly set for \p Dim.
  /// Used by TPlanWidener to avoid overwriting caller-supplied overrides.
  bool hasDimPF(unsigned Dim) const { return DimPFMap.count(Dim) > 0; }

  /// Stores the backedge-taken count SCEV for dimension \p Dim.
  /// The real trip count is getTCForDim(Dim) + 1.
  /// \p Dim uses the DimIdx convention (innermost=0, outermost=Depth-1).
  void setDimTC(unsigned Dim, const SCEV *BTC) { DimBackedgeTakenMap[Dim] = BTC; }

  /// Returns the backedge-taken count SCEV for dimension \p Dim,
  /// or nullptr if not available. Real TC = returned value + 1.
  const SCEV *getTCForDim(unsigned Dim) const {
    auto It = DimBackedgeTakenMap.find(Dim);
    return It != DimBackedgeTakenMap.end() ? It->second : nullptr;
  }
```

In the `private:` section (line 1380), after `DimPFMap`:

```cpp
  DenseMap<unsigned, const SCEV *> DimBackedgeTakenMap; ///< dim → backedge-taken count (real TC = value+1). nullptr if unknown.
```

- [ ] **Step 2: Store TripCount in `buildInitial()` (TPlan.cpp:739-740)**

Replace lines 739-740 in `llvm/lib/Transforms/Vectorize/TPlan.cpp`:

```cpp
// Before:
const SCEV *TC = Info.IVs[Idx].TripCount;
(void)TC; // Trip count stored for future use

// After:
const SCEV *TC = Info.IVs[Idx].TripCount;
if (TC)
  P.setDimTC(DimIdx, TC); // Stores backedge-taken count; real TC = TC+1.
```

- [ ] **Step 3: Guard the widener's default PF=256 with `hasDimPF()` (TPlanWidener.cpp:69)**

Replace line 69 in `llvm/lib/Transforms/Vectorize/TPlanWidener.cpp`:

```cpp
// Before:
Plan.setDimPF(Dim, 256u);

// After:
if (!Plan.hasDimPF(Dim))  // Don't overwrite caller-supplied PF override.
  Plan.setDimPF(Dim, 256u);
```

- [ ] **Step 4: Build and verify**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && ninja -j$(nproc) opt 2>&1 | tail -5
```

Expected: Build succeeds with no errors.

- [ ] **Step 5: Run existing LoopTensorize tests to verify no regressions**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -20
```

Expected: All tests PASS. No behavior change yet (hasDimPF() is only consulted when a PF override is pre-set, which no caller does yet).

- [ ] **Step 6: Commit**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
          llvm/lib/Transforms/Vectorize/TPlan.cpp \
          llvm/lib/Transforms/Vectorize/TPlanWidener.cpp && \
  git commit -m "tplan-tiling: add hasDimPF/setDimTC to TPlan; guard widener default PF"
```

---

### Task 2: Add `-loop-tensorize-pf` CLI override

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp:1-59`

This lets tests force a small PF (e.g., `--loop-tensorize-pf=8`) so that TC=16 > PF=8 triggers the tiling path. Because `hasDimPF()` now guards the widener, setting PF before `TPlanLowering_lower()` is safe.

- [ ] **Step 1: Add `cl::opt` and apply override before lowering**

At the top of `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp`, after existing includes and before the first function, add:

```cpp
#include "llvm/Support/CommandLine.h"

static cl::opt<unsigned> OverridePF(
    "loop-tensorize-pf",
    cl::desc("Override parallel factor (tile size) for all dimensions. "
             "0 = use default (256). Intended for testing only."),
    cl::init(0));
```

Then, inside the loop over `collectLoopNests` (after `TPlan Plan = TPlan::buildInitial(*InfoOpt);`, currently line 51), add before the call to `TPlanLowering_lower()` (line 59):

```cpp
    // Apply command-line PF override for testing.
    // hasDimPF() in the widener will preserve these values.
    if (OverridePF > 0)
      for (unsigned D = 0; D < InfoOpt->Depth; ++D)
        Plan.setDimPF(D, OverridePF);

    TPlanLowering_lower(Plan, F, LI, SE, DT);
```

- [ ] **Step 2: Build and verify**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && ninja -j$(nproc) opt 2>&1 | tail -5
```

Expected: Build succeeds.

- [ ] **Step 3: Run existing tests to verify no regressions**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -20
```

Expected: All tests PASS (OverridePF defaults to 0 = disabled).

- [ ] **Step 4: Commit**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/lib/Transforms/Vectorize/LoopTensorize.cpp && \
  git commit -m "tplan-tiling: add -loop-tensorize-pf CLI override for testing"
```

---

### Task 3: Implement tiling loop emission in `emitContraction()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:328-544`

This is the core task. We insert two sections into `emitContraction()`:
1. **Before line 329** (`emitContraction` start): the `emitTilingLoop()` helper struct+function.
2. **After line 508** (after stride lambdas): tiling decision + loop emission, replacing the original single-call path at lines 510-543.

**Critical ordering rule:** Expand ALL TripCount SCEVs to `Value *` FIRST (before any `emitTilingLoop()` call). This ensures SCEVExpander inserts code at the current BB, not inside a newly-created loop body.

**CFG structure for a 2-dim tiled case (output dim D0, contraction dim K):**

```
[original preheader BB]
  │ B.CreateBr(D0.header)
  ▼
[D0.header]: IV0 = phi [0, preheader], [IV0.next, D0.latch]
  │ icmp uge IV0, TC0 → exit0 : D0.body
  ▼
[D0.body]: ActualSize0 = umin(PF0, TC0 - IV0)
  │ B.CreateBr(K.header)   ← emitTilingLoop sets up D0.body → K chain
  ▼
[K.header]: IVK = phi [0, D0.body], [IVK.next, K.latch]
  │ icmp uge IVK, TCK → K.exit : K.body
  ▼
[K.body]: ActualSizeK = umin(PFK, TCK - IVK)
  │ [tensor.contract call]
  │ B.CreateBr(K.latch)    ← loop-close reversal, I=1 (innermost)
  ▼
[K.latch]: IVK.next = IVK + PFK → K.header
[K.exit]:  ← B.SetInsertPoint here after closing K
  │ B.CreateBr(D0.latch)   ← loop-close reversal, I=0 (outermost)
  ▼
[D0.latch]: IV0.next = IV0 + PF0 → D0.header
[D0.exit]:  ← B.SetInsertPoint here → returned to caller
```

- [ ] **Step 1: Add `TilingLoopInfo` struct and `emitTilingLoop()` helper before `emitContraction()`**

Insert the following immediately before the `static Value *emitContraction(...)` function definition in `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`:

```cpp
/// Describes the structure of one tiling loop emitted by emitTilingLoop().
struct TilingLoopInfo {
  BasicBlock *LatchBB;  ///< Back-edge block: IV += TileSize, br Header.
  BasicBlock *ExitBB;   ///< Fall-through after loop completes.
  PHINode    *IV;       ///< i64 induction: 0, TileSize, 2*TileSize, ...
  Value      *ActualSize; ///< min(TileSize, TripCount - IV) for remainder.
};

/// Emits a counting loop:
///   header: IV = phi [0, preheader], [IV+TileSize, latch]
///           if IV >= TripCount goto exit else goto body
///   body:   ActualSize = umin(TileSize, TripCount - IV)
///           [caller inserts code here, then calls B.CreateBr(info.LatchBB)]
///   latch:  IV.next = IV + TileSize; br header
///   exit:   [B insertion point left here by loop-close reversal]
///
/// On return, B's insertion point is in `body`. The caller MUST branch to
/// LatchBB after inserting body code, then set the insertion point to ExitBB.
/// Use the loop-close reversal pattern (see emitContraction tiling path).
static TilingLoopInfo emitTilingLoop(IRBuilder<> &B, Value *TripCount,
                                      Value *TileSize, const Twine &Name) {
  Function    *F   = B.GetInsertBlock()->getParent();
  LLVMContext &Ctx = F->getContext();
  Type        *I64 = Type::getInt64Ty(Ctx);

  BasicBlock *PreheaderBB = B.GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Ctx, Name + ".header", F);
  BasicBlock *BodyBB   = BasicBlock::Create(Ctx, Name + ".body",   F);
  BasicBlock *LatchBB  = BasicBlock::Create(Ctx, Name + ".latch",  F);
  BasicBlock *ExitBB   = BasicBlock::Create(Ctx, Name + ".exit",   F);

  // Preheader → Header.
  B.CreateBr(HeaderBB);

  // Header: phi + exit-check.
  B.SetInsertPoint(HeaderBB);
  PHINode *IV = B.CreatePHI(I64, 2, Name + ".iv");
  IV->addIncoming(ConstantInt::get(I64, 0), PreheaderBB);
  Value *Done = B.CreateICmpUGE(IV, TripCount, Name + ".done");
  B.CreateCondBr(Done, ExitBB, BodyBB);

  // Body: compute remainder-safe tile size.
  B.SetInsertPoint(BodyBB);
  Value *Remaining  = B.CreateSub(TripCount, IV, Name + ".rem");
  Value *ActualSize = B.CreateIntrinsic(Intrinsic::umin, {I64},
                                         {TileSize, Remaining},
                                         nullptr, Name + ".actual");
  // Insertion point remains in BodyBB for the caller.

  // Latch: advance IV and loop back. Built with a separate builder so the
  // caller's builder stays in BodyBB.
  IRBuilder<> LB(LatchBB);
  Value *NextIV = LB.CreateAdd(IV, TileSize, Name + ".next");
  IV->addIncoming(NextIV, LatchBB);
  LB.CreateBr(HeaderBB);

  return TilingLoopInfo{LatchBB, ExitBB, IV, ActualSize};
}
```

- [ ] **Step 2: Add tiling decision block after stride lambdas (after line 508)**

In `emitContraction()`, after the `getCStride` lambda definition ends (after line 508, the `};` closing getCStride), and before the existing line 510 (`// Build stride/dim vectors...`), insert:

```cpp
  // --- Tiling decision ---
  // Collect dimensions where the real TripCount > PF. These dimensions need
  // tiling loops. Dimensions where TC ≤ PF (or TC is unknown) use today's
  // single-call path with the full PF as the tile size.
  //
  // NB: getTCForDim() stores a backedge-taken count (BTC). Real TC = BTC+1.
  // We do the +1 here once, at the decision point.
  struct TileDimInfo {
    unsigned Dim;        // DimIdx (innermost=0)
    uint64_t TC;         // Real trip count (BTC + 1); 0 = dynamic (always tile)
    unsigned PF;         // Tile size (parallel factor)
    const SCEV *BTCSCEV; // Backedge-taken count SCEV for runtime expansion.
  };
  SmallVector<TileDimInfo, 4> TiledDims;

  auto checkDim = [&](unsigned D) {
    const SCEV *BTC = State.Plan.getTCForDim(D);
    if (!BTC)
      return; // Unknown TC — no tiling for this dim.
    unsigned PF = State.Plan.getPFForDim(D);
    if (auto *C = dyn_cast<SCEVConstant>(BTC)) {
      uint64_t RealTC = C->getValue()->getZExtValue() + 1;
      if (RealTC > static_cast<uint64_t>(PF))
        TiledDims.push_back({D, RealTC, PF, BTC});
    } else {
      // Dynamic TC: always tile conservatively.
      TiledDims.push_back({D, 0, PF, BTC});
    }
  };

  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    checkDim(static_cast<unsigned>(D));
  checkDim(static_cast<unsigned>(ContractDim));

  bool NeedsTiling = !TiledDims.empty();
```

- [ ] **Step 3: Replace the original single-call path (lines 510-543) with branching paths**

Delete lines 510-543 (the existing `// Build stride/dim vectors...` block through `return B.CreateCall(ContractFn, Args);`) and replace with:

```cpp
  // Common setup used by both paths.
  unsigned ContUD = static_cast<unsigned>(ContractDim);
  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee ContractFn =
      getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy);

  if (!NeedsTiling) {
    // ----------------------------------------------------------------
    // Fast path: TC ≤ PF for all dims — emit a single tensor.contract.
    // ----------------------------------------------------------------
    SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
    for (int D = OutputDimSet.find_first(); D >= 0;
         D = OutputDimSet.find_next(D)) {
      unsigned UD = static_cast<unsigned>(D);
      CStrides.push_back(getCStride(UD));
      AStrides.push_back(getAStride(UD));
      BStrides.push_back(getBStride(UD));
      OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
    }
    Value *AContractStride = getAStride(ContUD);
    Value *BContractStride = getBStride(ContUD);
    Value *K = I64(State.Plan.getPFForDim(ContUD));

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

  // ----------------------------------------------------------------
  // Tiling path: emit nested loops around a PF-sized tensor.contract.
  //
  // STEP A — Expand ALL TripCount SCEVs to Value* BEFORE creating any
  //   loop BBs. This keeps SCEVExpander's inserted code in the current
  //   preheader BB, not inside a loop body.
  // ----------------------------------------------------------------
  SmallVector<Value *> TCValues; // parallel to TiledDims
  for (auto &TD : TiledDims) {
    // expandCodeFor requires the SCEV to be safe to expand here.
    // getTCForDim() stores backedge-taken count; add 1 for real TC.
    Value *BTC = State.Expander->expandCodeFor(
        TD.BTCSCEV, B.getInt64Ty(), &*B.GetInsertPoint());
    TCValues.push_back(B.CreateAdd(BTC, B.getInt64(1), "tc.real"));
  }

  // STEP B — Emit tiling loops (outermost first, matching TiledDims order).
  SmallVector<TilingLoopInfo, 4> LoopInfos;
  for (unsigned I = 0; I < TiledDims.size(); ++I) {
    std::string Name = "tile.d" + std::to_string(TiledDims[I].Dim);
    Value *TileSize  = B.getInt64(TiledDims[I].PF);
    LoopInfos.push_back(emitTilingLoop(B, TCValues[I], TileSize, Name));
    // B's insertion point is now in the body of loop I.
    // The next emitTilingLoop() call will nest inside it correctly.
  }

  // STEP C — Compute offset base pointers inside the innermost body.
  // For each tiled dim, offset = IV * stride (in elements, via GEP).
  Value *TiledCPtr = CPtr, *TiledAPtr = LHSPtr, *TiledBPtr = RHSPtr;
  DenseMap<unsigned, Value *> ActualSizes; // dim → min(PF, TC-IV)
  for (unsigned I = 0; I < TiledDims.size(); ++I) {
    unsigned Dim    = TiledDims[I].Dim;
    Value   *IV     = LoopInfos[I].IV;
    ActualSizes[Dim] = LoopInfos[I].ActualSize;

    auto offsetPtr = [&](Value *Base, Value *Stride) -> Value * {
      Value *Off = B.CreateMul(IV, Stride, "tile.off");
      return B.CreateGEP(ElemTy, Base, Off, "tile.ptr");
    };
    Value *AStr = getAStride(Dim);
    Value *BStr = getBStride(Dim);
    Value *CStr = getCStride(Dim);
    // Only offset when the operand actually spans this dim (stride != 0).
    auto nonZero = [](Value *V) -> bool {
      auto *CI = dyn_cast<ConstantInt>(V);
      return !CI || !CI->isZero();
    };
    if (nonZero(AStr)) TiledAPtr = offsetPtr(TiledAPtr, AStr);
    if (nonZero(BStr)) TiledBPtr = offsetPtr(TiledBPtr, BStr);
    if (nonZero(CStr)) TiledCPtr = offsetPtr(TiledCPtr, CStr);
  }

  // STEP D — Build tile-sized tensor.contract arguments and call.
  SmallVector<Value *> Args;
  Args.push_back(TiledCPtr);
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getCStride(static_cast<unsigned>(D)));
  Args.push_back(TiledAPtr);
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getAStride(static_cast<unsigned>(D)));
  Args.push_back(getAStride(ContUD));
  Args.push_back(TiledBPtr);
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getBStride(static_cast<unsigned>(D)));
  Args.push_back(getBStride(ContUD));
  // K dimension size: use per-tile actual size if K is tiled, else full PF.
  Args.push_back(ActualSizes.count(ContUD) ? ActualSizes[ContUD]
                                            : I64(State.Plan.getPFForDim(ContUD)));
  // Output dimension sizes.
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    Args.push_back(ActualSizes.count(UD) ? ActualSizes[UD]
                                          : I64(State.Plan.getPFForDim(UD)));
  }
  Value *Call = B.CreateCall(ContractFn, Args);

  // STEP E — Close tiling loops in reverse (innermost first).
  // Pattern: branch from current body BB to LatchBB, then set insert point
  // to ExitBB (which becomes the body of the next-outer loop, or the
  // continuation point after all loops).
  for (int I = static_cast<int>(LoopInfos.size()) - 1; I >= 0; --I) {
    B.CreateBr(LoopInfos[I].LatchBB);
    B.SetInsertPoint(LoopInfos[I].ExitBB);
  }

  return Call;
```

- [ ] **Step 4: Build and verify**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && ninja -j$(nproc) opt 2>&1 | tail -10
```

Expected: Build succeeds with no errors.

- [ ] **Step 5: Run existing tests (no tiling should occur with default PF=256)**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -20
```

Expected: All existing tests PASS. The tiling path is gated on `NeedsTiling`, which is false when TC ≤ PF=256 (all existing tests use small trip counts).

- [ ] **Step 6: Commit**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp && \
  git commit -m "tplan-tiling: emit tiling loops in emitContraction() when TripCount > PF"
```

---

### Task 4: Tests

**Files:**
- Create: `llvm/test/Transforms/LoopTensorize/basic/tiling-no-tile.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tiling-pf8.ll`
- Create: `llvm/test/Transforms/LoopTensorize/basic/tiling-remainder.ll`

Tests follow TDD order: write the test file, run it to confirm the expected behavior, then commit.

#### Test A: Baseline — no tiling when TC ≤ PF

- [ ] **Step 1: Write baseline test**

Create `llvm/test/Transforms/LoopTensorize/basic/tiling-no-tile.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; GEMM 64×64×64 with default PF=256. All dimensions fit in one tile.
; Expected: single tensor.contract call, no tile.d* blocks.
;
; CHECK-LABEL: @gemm_64(
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: tile.d

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_64(ptr %A, ptr %B, ptr %C) {
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
  %ai   = mul i64 %i, 64
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 64
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 64
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ci   = mul i64 %i, 64
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 64
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 64
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
```

- [ ] **Step 2: Run baseline test**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tiling-no-tile.ll
```

Expected: PASS — single contract call, no `tile.d` labels.

- [ ] **Step 3: Commit baseline test**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/test/Transforms/LoopTensorize/basic/tiling-no-tile.ll && \
  git commit -m "tplan-tiling: baseline test — single contract when TC ≤ PF"
```

#### Test B: Tiling path with PF=8 override

- [ ] **Step 4: Write tiling test**

Create `llvm/test/Transforms/LoopTensorize/basic/tiling-pf8.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; GEMM 16×16×16 with forced PF=8. TripCount=16 > PF=8 for all dims.
; Expected: tiling loop structure with min(8, TC-IV) remainder handling.
;
; CHECK-LABEL: @gemm_16_tiled(
; CHECK: tile.d{{[0-9]+}}.header:
; CHECK: icmp uge i64
; CHECK: tile.d{{[0-9]+}}.body:
; CHECK: call i64 @llvm.umin.i64(
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tile.d{{[0-9]+}}.latch:
; CHECK: tile.d{{[0-9]+}}.exit:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_16_tiled(ptr %A, ptr %B, ptr %C) {
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
  %ai   = mul i64 %i, 16
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 16
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
  %ci   = mul i64 %i, 16
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

- [ ] **Step 5: Run tiling test**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tiling-pf8.ll
```

Expected: PASS — `tile.d*.header`, `tile.d*.body`, `tile.d*.latch`, `tile.d*.exit` blocks present; `@llvm.umin.i64` for remainder; single contract call per tile.

- [ ] **Step 6: Commit tiling test**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/test/Transforms/LoopTensorize/basic/tiling-pf8.ll && \
  git commit -m "tplan-tiling: end-to-end test with -loop-tensorize-pf=8 (TC=16 > PF=8)"
```

#### Test C: Remainder handling — TC not divisible by PF

- [ ] **Step 7: Write remainder test**

Create `llvm/test/Transforms/LoopTensorize/basic/tiling-remainder.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; GEMM 17×17×17 with PF=8. 17 is not divisible by 8 (2 full tiles + 1 rem).
; Expected: umin(8, 17-IV) produces correct remainder tile of size 1.
;
; CHECK-LABEL: @gemm_17_remainder(
; CHECK: tile.d{{[0-9]+}}.body:
; CHECK: %{{.*}} = call i64 @llvm.umin.i64(i64 8,
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tile.d{{[0-9]+}}.exit:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_17_remainder(ptr %A, ptr %B, ptr %C) {
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
  %ai   = mul i64 %i, 17
  %ak   = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 17
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, 17
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ci   = mul i64 %i, 17
  %cj   = add i64 %ci, %j
  %cptr = getelementptr float, ptr %C, i64 %cj
  store float %sum, ptr %cptr
  %j.next = add i64 %j, 1
  %j.done = icmp eq i64 %j.next, 17
  br i1 %j.done, label %j.latch, label %j.loop
j.latch:
  %i.next = add i64 %i, 1
  %i.done = icmp eq i64 %i.next, 17
  br i1 %i.done, label %exit, label %i.loop
exit:
  ret void
}
```

- [ ] **Step 8: Run remainder test**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tiling-remainder.ll
```

Expected: PASS — `umin(i64 8, ...)` appears for remainder; contract call present.

- [ ] **Step 9: Run full test suite**

```bash
cd /Users/yun-yugyeong/Dev/llvm/build && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -30
```

Expected: All tests PASS.

- [ ] **Step 10: Commit remainder test**

```bash
cd /Users/yun-yugyeong/Dev/llvm && \
  git add llvm/test/Transforms/LoopTensorize/basic/tiling-remainder.ll && \
  git commit -m "tplan-tiling: test non-divisible TC remainder (17 / PF=8)"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | `hasDimPF()` + `DimBackedgeTakenMap` + widener guard | `TPlan.h`, `TPlan.cpp`, `TPlanWidener.cpp` |
| 2 | `-loop-tensorize-pf` CLI override | `LoopTensorize.cpp` |
| 3 | Tiling loop emission in `emitContraction()` | `TPlanLowering.cpp` |
| 4 | Tests: baseline / tiling / remainder | `test/Transforms/LoopTensorize/basic/` |

**v1 → v2 key fixes:**
- `hasDimPF()` guard prevents widener from overwriting CLI/TTI PF overrides.
- Tiling decision placed **after** stride lambdas (line 508+), not before.
- All `expandCodeFor()` calls batched **before** any `emitTilingLoop()` call.
- Tests use `tile.d*.header` label checks instead of broken `CHECK-NOT: br i1`.
- `DimBackedgeTakenMap` naming makes the backedge-taken-count semantics explicit.
