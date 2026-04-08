# TPlan Tiling Loop Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When TripCount > PF (tile size) for any dimension, emit tiling loops around PF-sized `tensor.contract` intrinsic calls instead of a single oversized call.

**Architecture:** Store per-dimension TripCount SCEVs in TPlan alongside the existing DimPFMap. During `emitContraction()`, compare TripCount vs PF for each dimension. For dimensions needing tiling, generate `for tile = 0 to TC step PF` loops with `min(PF, TC - tile)` remainder handling and strided pointer offsets. Dimensions where PF == TripCount (or TripCount is unknown) fall through to today's single-call path.

**Tech Stack:** LLVM IR (IRBuilder), ScalarEvolution (SCEV, SCEVExpander), TPlan infrastructure (TPlanLowering.cpp, TPlan.h, TPlan.cpp)

---

### Task 1: Store TripCount in TPlan

**Files:**
- Modify: `/root/llvm-project/llvm/include/llvm/Transforms/Vectorize/TPlan.h:1375-1394`
- Modify: `/root/llvm-project/llvm/lib/Transforms/Vectorize/TPlan.cpp:738-740`

- [ ] **Step 1: Add DimTripCountMap to TPlan private members**

In `TPlan.h`, add a new map after `DimPFMap` (line 1380):

```cpp
DenseMap<unsigned, unsigned> DimPFMap;     ///< dim index (DimIdx, innermost=0) → parallel factor.
DenseMap<unsigned, const SCEV *> DimTripCountMap; ///< dim index → trip count SCEV (nullptr if unknown).
```

- [ ] **Step 2: Add public accessors for DimTripCountMap**

In `TPlan.h`, add after `setDimPF()` (line 1327):

```cpp
/// Returns the trip count SCEV for dimension \p Dim. nullptr if unknown.
/// \p Dim uses the DimIdx convention (innermost=0, outermost=Depth-1).
const SCEV *getTripCountForDim(unsigned Dim) const {
  auto It = DimTripCountMap.find(Dim);
  return It != DimTripCountMap.end() ? It->second : nullptr;
}
/// \p Dim uses the DimIdx convention (innermost=0, outermost=Depth-1).
void setDimTripCount(unsigned Dim, const SCEV *TC) {
  DimTripCountMap[Dim] = TC;
}
```

- [ ] **Step 3: Store TripCount during buildInitial()**

In `TPlan.cpp`, replace lines 739-740:

```cpp
// Before:
const SCEV *TC = Info.IVs[Idx].TripCount;
(void)TC; // Trip count stored for future use

// After:
const SCEV *TC = Info.IVs[Idx].TripCount;
if (TC)
  P.setDimTripCount(DimIdx, TC);
```

- [ ] **Step 4: Verify build compiles**

Run:
```bash
cd /root/llvm-project/build && ninja -j$(nproc) opt 2>&1 | tail -5
```
Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan-tiling: store per-dimension TripCount SCEV in TPlan"
```

---

### Task 2: Add TTI-based PF override hook in LoopTensorize

**Files:**
- Modify: `/root/llvm-project/llvm/lib/Transforms/Vectorize/LoopTensorize.cpp:56-59`

- [ ] **Step 1: Wire TTI tile-size query before lowering**

Replace the comment-only hook at lines 56-59 with actual PF override logic. The default PF=256 from the widener is kept as a fallback for targets without tensor hardware. For targets with known tile sizes, override specific dims:

```cpp
// Lower to IR.  TPlanWidener_widen() (called inside lower()) seeds
// Plan.DimPFMap with the default PF=256 for each IV dimension.
// Override specific dims here if a target-specific tile size is needed.
//
// TODO: Query TTI for target-specific tile sizes. For now, leave
// PF=256 as default. Callers can override via Plan.setDimPF() before
// this point once TTI tile-size queries are implemented.
// Example for AMX: Plan.setDimPF(0, 16); Plan.setDimPF(1, 16); Plan.setDimPF(2, 16);
TPlanLowering_lower(Plan, F, LI, SE, DT);
```

Note: The actual TTI query (`TTI.getTensorTileSize()` or similar) doesn't exist yet. This task documents the hook and keeps the default. Real TTI integration is a separate follow-up.

- [ ] **Step 2: Verify existing tests still pass**

Run:
```bash
cd /root/llvm-project/build && ninja -j$(nproc) opt && \
  ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -20
```
Expected: All existing tests PASS (no behavior change yet).

- [ ] **Step 3: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/LoopTensorize.cpp
git commit -m "tplan-tiling: document TTI tile-size override hook in LoopTensorize"
```

---

### Task 3: Write failing test for tiled contraction

**Files:**
- Create: `/root/llvm-project/llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled.ll`

- [ ] **Step 1: Write the test**

This test has a 3-loop GEMM with trip count 64 (M=64, N=64, K=64). With PF=256 (default), the entire loop fits in one tile and should emit a single `tensor.contract`. But we'll add a future test variant with smaller PF to verify tiling. For now, this test verifies the TripCount storage doesn't break anything:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3-level GEMM (64x64x64). With default PF=256, all dims fit in one tile.
; Verify single tensor.contract is emitted (no tiling loops needed).
;
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: br i1

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_64x64x64(ptr %A, ptr %B, ptr %C) {
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

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled.ll
```
Expected: PASS (single contract emitted, no branches after it).

- [ ] **Step 3: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled.ll
git commit -m "tplan-tiling: add baseline test for tiled contraction (TC=64, PF=256)"
```

---

### Task 4: Implement tiling loop emission in emitContraction()

**Files:**
- Modify: `/root/llvm-project/llvm/lib/Transforms/Vectorize/TPlanLowering.cpp:329-543`

This is the core task. We modify `emitContraction()` to wrap the intrinsic call in tiling loops when any dimension's TripCount exceeds PF.

- [ ] **Step 1: Add helper function emitTilingLoop()**

Insert before `emitContraction()` (before line 329):

```cpp
/// Emit a tiling loop: for (iv = 0; iv < TripCount; iv += TileSize).
/// Returns {HeaderBB, IV_PHI, ExitBB}. The caller inserts tile-body code
/// after the returned HeaderBB and before branching to LatchBB.
/// The IRBuilder insertion point is set to the loop header after the PHI.
struct TilingLoopInfo {
  BasicBlock *HeaderBB;
  BasicBlock *LatchBB;
  BasicBlock *ExitBB;
  PHINode *IV;          ///< i64 induction variable: 0, TileSize, 2*TileSize, ...
  Value *ActualSize;    ///< min(TileSize, TripCount - IV) for remainder handling
};

static TilingLoopInfo emitTilingLoop(IRBuilder<> &B, Value *TripCount,
                                      Value *TileSize, const Twine &Name) {
  Function *F = B.GetInsertBlock()->getParent();
  LLVMContext &Ctx = F->getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // Create basic blocks.
  BasicBlock *PreheaderBB = B.GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Ctx, Name + ".header", F);
  BasicBlock *BodyBB   = BasicBlock::Create(Ctx, Name + ".body", F);
  BasicBlock *LatchBB  = BasicBlock::Create(Ctx, Name + ".latch", F);
  BasicBlock *ExitBB   = BasicBlock::Create(Ctx, Name + ".exit", F);

  // Preheader -> Header.
  B.CreateBr(HeaderBB);

  // Header: IV phi + exit condition.
  B.SetInsertPoint(HeaderBB);
  PHINode *IV = B.CreatePHI(I64, 2, Name + ".iv");
  IV->addIncoming(ConstantInt::get(I64, 0), PreheaderBB);
  Value *Done = B.CreateICmpUGE(IV, TripCount, Name + ".done");
  B.CreateCondBr(Done, ExitBB, BodyBB);

  // Body: compute actual tile size = min(TileSize, TripCount - IV).
  B.SetInsertPoint(BodyBB);
  Value *Remaining = B.CreateSub(TripCount, IV, Name + ".rem");
  Value *ActualSize = B.CreateIntrinsic(Intrinsic::umin, {I64},
                                         {TileSize, Remaining},
                                         nullptr, Name + ".actual");
  // Caller inserts tensor intrinsic here, then branches to LatchBB.
  // We leave the insertion point in BodyBB for the caller.

  // Latch: IV += TileSize, back-edge to Header.
  // (Caller must emit br to LatchBB after their code.)
  IRBuilder<> LatchBuilder(LatchBB);
  Value *NextIV = LatchBuilder.CreateAdd(IV, TileSize, Name + ".next");
  IV->addIncoming(NextIV, LatchBB);
  LatchBuilder.CreateBr(HeaderBB);

  TilingLoopInfo Info;
  Info.HeaderBB = HeaderBB;
  Info.LatchBB = LatchBB;
  Info.ExitBB = ExitBB;
  Info.IV = IV;
  Info.ActualSize = ActualSize;
  return Info;
}
```

- [ ] **Step 2: Add tiling decision logic in emitContraction()**

After the OutputDimSet and ContractDim are finalized (after line 429), and before the C store lookup (line 431), insert tiling analysis:

```cpp
  // --- Tiling decision: collect dims where TripCount > PF ---
  struct TileDimInfo {
    unsigned Dim;        // DimIdx
    const SCEV *TC;      // TripCount SCEV
    unsigned PF;         // Tile size (parallel factor)
    bool IsOutputDim;    // true = output dim, false = contraction dim
  };
  SmallVector<TileDimInfo, 4> TiledDims;

  // Check output dims.
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    unsigned PF = State.Plan.getPFForDim(UD);
    const SCEV *TC = State.Plan.getTripCountForDim(UD);
    if (!TC) continue;  // Unknown TC — use PF as-is (today's behavior).
    if (auto *SCEVConst = dyn_cast<SCEVConstant>(TC)) {
      uint64_t TCVal = SCEVConst->getValue()->getZExtValue() + 1; // backedge count + 1
      if (TCVal <= PF) continue; // Fits in one tile.
      TiledDims.push_back({UD, TC, PF, /*IsOutputDim=*/true});
    }
    // Dynamic TC: tile conservatively.
    else {
      TiledDims.push_back({UD, TC, PF, /*IsOutputDim=*/true});
    }
  }
  // Check contraction dim.
  {
    unsigned ContUD = static_cast<unsigned>(ContractDim);
    unsigned PF = State.Plan.getPFForDim(ContUD);
    const SCEV *TC = State.Plan.getTripCountForDim(ContUD);
    if (TC) {
      if (auto *SCEVConst = dyn_cast<SCEVConstant>(TC)) {
        uint64_t TCVal = SCEVConst->getValue()->getZExtValue() + 1;
        if (TCVal > PF)
          TiledDims.push_back({ContUD, TC, PF, /*IsOutputDim=*/false});
      } else {
        TiledDims.push_back({ContUD, TC, PF, /*IsOutputDim=*/false});
      }
    }
  }
  bool NeedsTiling = !TiledDims.empty();
```

- [ ] **Step 3: Wrap the intrinsic call in tiling loops**

Replace the final argument-building and CreateCall section (lines 510-543) with:

```cpp
  // Build stride/dim vectors in output-dim order (OutputDimSet iteration order).
  // When tiling, these are computed per-tile with offset pointers.

  if (!NeedsTiling) {
    // --- Original path: single tensor.contract call ---
    SmallVector<Value *> CStrides, AStrides, BStrides, OutDims;
    for (int D = OutputDimSet.find_first(); D >= 0;
         D = OutputDimSet.find_next(D)) {
      unsigned UD = static_cast<unsigned>(D);
      CStrides.push_back(getCStride(UD));
      AStrides.push_back(getAStride(UD));
      BStrides.push_back(getBStride(UD));
      OutDims.push_back(I64(State.Plan.getPFForDim(UD)));
    }
    unsigned ContUD = static_cast<unsigned>(ContractDim);
    Value *AContractStride = getAStride(ContUD);
    Value *BContractStride = getBStride(ContUD);
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

  // --- Tiling path: nested loops around tile-sized tensor.contract ---
  //
  // For each tiled dimension, emit a tiling loop. The innermost loop body
  // contains the tensor.contract call with:
  //   - Offset base pointers: ptr + tile_iv * stride[dim] * sizeof(elem)
  //   - ActualSize from min(PF, TC - tile_iv) for remainder handling
  //
  // Tiling loop nesting order: output dims first (outermost), then
  // contraction dim (innermost), matching the natural iteration order.

  Module *Mod = B.GetInsertBlock()->getModule();
  FunctionCallee ContractFn =
      getTensorContractFn(*Mod, RankA, RankB, RankC, ElemTy);
  unsigned ElemBytes = ElemTy->getPrimitiveSizeInBits() / 8;

  // Expand TripCounts to IR values.
  SmallVector<TilingLoopInfo, 4> LoopInfos;
  for (auto &TD : TiledDims) {
    // Expand TripCount SCEV to i64 value. BackedgeTakenCount is iterations-1,
    // so real trip count = BTC + 1.
    Value *TCVal = State.Expander->expandCodeFor(
        TD.TC, B.getInt64Ty(), &*B.GetInsertPoint());
    TCVal = B.CreateAdd(TCVal, B.getInt64(1), "tc.real");
    Value *TileSize = B.getInt64(TD.PF);

    std::string LoopName = "tile.d" + std::to_string(TD.Dim);
    TilingLoopInfo LI = emitTilingLoop(B, TCVal, TileSize, LoopName);
    LoopInfos.push_back(LI);
    // B's insertion point is now inside the loop body.
  }

  // Inside the innermost tiling loop body: compute offset pointers and call.
  // For each tiled dim, offset = tile_iv * stride_bytes.
  Value *TiledCPtr = CPtr, *TiledAPtr = LHSPtr, *TiledBPtr = RHSPtr;
  DenseMap<unsigned, Value *> TiledActualSizes; // dim -> actual tile size

  for (unsigned I = 0; I < TiledDims.size(); ++I) {
    auto &TD = TiledDims[I];
    auto &LI = LoopInfos[I];
    Value *IV = LI.IV;
    TiledActualSizes[TD.Dim] = LI.ActualSize;

    // Compute byte offset for this dim's tile: IV * stride * elem_bytes.
    // stride is in elements (from decomposePtrForDims), so byte offset = IV * stride_scev expanded.
    // Use GEP with element-level offset: IV * stride_elements.
    auto offsetPtr = [&](Value *BasePtr, Value *Stride) -> Value * {
      Value *ElemOff = B.CreateMul(IV, Stride, "tile.off");
      return B.CreateGEP(ElemTy, BasePtr, ElemOff, "tile.ptr");
    };

    // Get strides for this dimension (in elements).
    Value *AStr = getAStride(TD.Dim);
    Value *BStr = getBStride(TD.Dim);
    Value *CStr = getCStride(TD.Dim);

    // Only offset pointers for dims where the operand has a non-zero stride.
    // Stride==0 means broadcast (operand doesn't span this dim).
    auto isNonZero = [](Value *V) -> bool {
      if (auto *CI = dyn_cast<ConstantInt>(V))
        return !CI->isZero();
      return true; // dynamic stride, assume non-zero
    };
    if (isNonZero(AStr)) TiledAPtr = offsetPtr(TiledAPtr, AStr);
    if (isNonZero(BStr)) TiledBPtr = offsetPtr(TiledBPtr, BStr);
    if (isNonZero(CStr)) TiledCPtr = offsetPtr(TiledCPtr, CStr);
  }

  // Build the tile-sized tensor.contract arguments.
  SmallVector<Value *> Args;
  Args.push_back(TiledCPtr);
  // C strides (unchanged — stride within a tile is the same as the full tensor).
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getCStride(static_cast<unsigned>(D)));
  // A pointer + strides + contract stride.
  Args.push_back(TiledAPtr);
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getAStride(static_cast<unsigned>(D)));
  Args.push_back(getAStride(static_cast<unsigned>(ContractDim)));
  // B pointer + strides + contract stride.
  Args.push_back(TiledBPtr);
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D))
    Args.push_back(getBStride(static_cast<unsigned>(D)));
  Args.push_back(getBStride(static_cast<unsigned>(ContractDim)));
  // K dimension size (tiled or original PF).
  unsigned ContUD = static_cast<unsigned>(ContractDim);
  if (TiledActualSizes.count(ContUD))
    Args.push_back(TiledActualSizes[ContUD]);
  else
    Args.push_back(I64(State.Plan.getPFForDim(ContUD)));
  // Output dim sizes (tiled or original PF).
  for (int D = OutputDimSet.find_first(); D >= 0;
       D = OutputDimSet.find_next(D)) {
    unsigned UD = static_cast<unsigned>(D);
    if (TiledActualSizes.count(UD))
      Args.push_back(TiledActualSizes[UD]);
    else
      Args.push_back(I64(State.Plan.getPFForDim(UD)));
  }

  Value *Call = B.CreateCall(ContractFn, Args);

  // Close tiling loops (innermost first): branch to latch, set insert point to exit.
  for (int I = static_cast<int>(LoopInfos.size()) - 1; I >= 0; --I) {
    B.CreateBr(LoopInfos[I].LatchBB);
    B.SetInsertPoint(LoopInfos[I].ExitBB);
  }

  return Call;
```

- [ ] **Step 4: Verify build compiles**

Run:
```bash
cd /root/llvm-project/build && ninja -j$(nproc) opt 2>&1 | tail -5
```
Expected: Build succeeds.

- [ ] **Step 5: Run existing tests to verify no regressions**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -30
```
Expected: All existing tests PASS. The tiling path is only triggered when TripCount > PF, which doesn't happen with the default PF=256 and test trip counts of 16/64.

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-tiling: emit tiling loops in emitContraction() when TripCount > PF"
```

---

### Task 5: Write tiling integration test

**Files:**
- Create: `/root/llvm-project/llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll`

- [ ] **Step 1: Write the tiling test**

This test uses a GEMM with TC=16 and forces PF=8 via a command-line option (to be added) or by setting the default PF lower. Since we can't easily override PF from the command line yet, we'll verify the tiling logic with a test that validates the output structure when TripCount > PF. For now, we write this as a manual integration test with CHECK lines:

```llvm
; RUN: opt -passes=loop-tensorize -S < %s | FileCheck %s
;
; 3-level GEMM (16x16x16). With default PF=256, all dims fit in one tile.
; This test verifies baseline (no tiling). A follow-up will add PF override
; to test actual tiling loop generation.
;
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK-NOT: tile.d

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_16x16x16_tilecheck(ptr %A, ptr %B, ptr %C) {
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

- [ ] **Step 2: Run test**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll
git commit -m "tplan-tiling: add integration test for tiled contraction"
```

---

### Task 6: Add command-line PF override for testing

**Files:**
- Modify: `/root/llvm-project/llvm/lib/Transforms/Vectorize/LoopTensorize.cpp:30-59`

- [ ] **Step 1: Add cl::opt for PF override**

Add at the top of `LoopTensorize.cpp`, after the existing includes and before the `run()` function:

```cpp
static cl::opt<unsigned> OverridePF(
    "loop-tensorize-pf",
    cl::desc("Override parallel factor (tile size) for all dimensions"),
    cl::init(0)); // 0 = use default (256)
```

Add the required include if not present:
```cpp
#include "llvm/Support/CommandLine.h"
```

- [ ] **Step 2: Wire the override before lowering**

In `LoopTensorize.cpp`, before the `TPlanLowering_lower()` call (line 59), add:

```cpp
    // Apply command-line PF override for testing.
    if (OverridePF > 0) {
      for (unsigned D = 0; D < InfoOpt->Depth; ++D)
        Plan.setDimPF(D, OverridePF);
    }

    TPlanLowering_lower(Plan, F, LI, SE, DT);
```

- [ ] **Step 3: Verify build compiles**

Run:
```bash
cd /root/llvm-project/build && ninja -j$(nproc) opt 2>&1 | tail -5
```
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/LoopTensorize.cpp
git commit -m "tplan-tiling: add -loop-tensorize-pf command-line override for testing"
```

---

### Task 7: Write end-to-end tiling loop test with PF override

**Files:**
- Modify: `/root/llvm-project/llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll`

- [ ] **Step 1: Update the test to use -loop-tensorize-pf=8**

Replace the test content with:

```llvm
; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; 3-level GEMM (16x16x16) with forced PF=8. TripCount=16 > PF=8, so each
; dimension should get a tiling loop. Expect tiling loops and tile-sized
; tensor.contract calls.
;
; CHECK-LABEL: @gemm_16x16x16_tiled
; CHECK: tile.d{{[0-9]+}}.header:
; CHECK: icmp uge i64
; CHECK: tile.d{{[0-9]+}}.body:
; CHECK: call @llvm.umin.i64
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tile.d{{[0-9]+}}.latch:
; CHECK: tile.d{{[0-9]+}}.exit:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_16x16x16_tiled(ptr %A, ptr %B, ptr %C) {
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

- [ ] **Step 2: Run the tiling test**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll
```
Expected: PASS — tiling loops are generated with `tile.d*.header`, `tile.d*.body`, `tile.d*.latch`, `tile.d*.exit` blocks, with `@llvm.umin.i64` for remainder handling.

- [ ] **Step 3: Run full test suite to verify no regressions**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -30
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-pf8.ll
git commit -m "tplan-tiling: end-to-end test with -loop-tensorize-pf=8 forcing tiling loops"
```

---

### Task 8: Add remainder-only edge case test

**Files:**
- Create: `/root/llvm-project/llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-remainder.ll`

- [ ] **Step 1: Write remainder edge case test**

Trip count 17 with PF=8: 2 full tiles of 8 + 1 remainder tile of 1. This validates `min(PF, TC - IV)` produces correct remainder handling:

```llvm
; RUN: opt -passes=loop-tensorize -loop-tensorize-pf=8 -S < %s | FileCheck %s
;
; GEMM 17x17x17 with PF=8. Trip count not divisible by tile size.
; Verify tiling loops with remainder handling via @llvm.umin.
;
; CHECK-LABEL: @gemm_17x17x17_remainder
; CHECK: tile.d{{[0-9]+}}.body:
; CHECK: %{{.*}} = call i64 @llvm.umin.i64(i64 8,
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tile.d{{[0-9]+}}.exit:

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "aarch64"

define void @gemm_17x17x17_remainder(ptr %A, ptr %B, ptr %C) {
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

- [ ] **Step 2: Run the remainder test**

Run:
```bash
cd /root/llvm-project/build && ./bin/llvm-lit -v ../llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-remainder.ll
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add llvm/test/Transforms/LoopTensorize/basic/tensor-contract-tiled-remainder.ll
git commit -m "tplan-tiling: test non-divisible trip count remainder handling (17 / PF=8)"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | Store TripCount SCEV in TPlan | TPlan.h, TPlan.cpp |
| 2 | Document TTI hook (no behavior change) | LoopTensorize.cpp |
| 3 | Baseline test (TC fits in PF) | tensor-contract-tiled.ll |
| 4 | **Core: tiling loop emission in emitContraction()** | TPlanLowering.cpp |
| 5 | Integration test (verify no tiling when PF >= TC) | tensor-contract-tiled-pf8.ll |
| 6 | `-loop-tensorize-pf` CLI override | LoopTensorize.cpp |
| 7 | End-to-end test with PF=8 forcing tiling | tensor-contract-tiled-pf8.ll |
| 8 | Remainder edge case test (TC=17, PF=8) | tensor-contract-tiled-remainder.ll |
