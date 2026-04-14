# TTI Tensor Extension: ISA-Aware Tensorization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the TTI tensor interface so LoopTensorize can query per-ISA tile sizes (M, N, K), consume epilogue tiers, expose AArch64 tensor capabilities, and use per-operation throughput in the cost model.

**Architecture:** Extend `TensorContractTileInfo` with M/N fields and masking flags. Wire `EpilogueKSizes` consumption into `TPTilingRegion::execute()`. Implement `hasTensorOps()` + `getSupportedTensorOps()` for AArch64 (FMMLA, SME). Replace hardcoded cost model multipliers with per-op throughput from TTI via a new `getTensorOpThroughput()` method.

**Tech Stack:** C++ (LLVM), TargetTransformInfo, IRBuilder, LLVM lit tests.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h` | Add M/N to TensorContractTileInfo; add masking flag |
| Modify | `llvm/include/llvm/Analysis/TargetTransformInfo.h` | Add `getTensorOpThroughput()` declaration |
| Modify | `llvm/include/llvm/Analysis/TargetTransformInfoImpl.h` | Default `getTensorOpThroughput()` impl |
| Modify | `llvm/lib/Analysis/TargetTransformInfo.cpp` | Forward `getTensorOpThroughput()` |
| Modify | `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Use M/N from TTI in Policy PF; consume EpilogueKSizes |
| Modify | `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Add epilogue tier loops in TPTilingRegion::execute() |
| Modify | `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add EpiloguePFs to TPTilingRegion |
| Modify | `llvm/lib/Target/X86/X86TargetTransformInfo.cpp` | Return M/N in tile info; implement getTensorOpThroughput |
| Modify | `llvm/lib/Target/X86/X86TargetTransformInfo.h` | Declare getTensorOpThroughput |
| Modify | `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp` | Implement hasTensorOps, getSupportedTensorOps, getTensorOpThroughput, add M/N |
| Modify | `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.h` | Declare new overrides |
| Modify | `llvm/lib/Transforms/Vectorize/TensorCostModel.cpp` | Use getTensorOpThroughput instead of 10x hardcode |
| Test | `llvm/test/Transforms/LoopTensorize/x86/amx-tile-sizes.ll` | Verify X86 AMX M/N/K tile sizing |
| Test | `llvm/test/Transforms/LoopTensorize/aarch64/fmmla-tile-info.ll` | Verify AArch64 FMMLA tile info + epilogue |

---

## Background

Key types and functions you need to understand:

- **`TensorContractTileInfo`** (`TensorISAInfo.h:31`): returned by `TTI::getTensorContractTileInfo()`. Has `PrimaryK` and `EpilogueKSizes`. Currently no M/N fields.
- **`TensorOpDesc`** (`TensorISAInfo.h:17`): describes one hardware tensor op. Has M, N, K dims and IntrinsicID. Returned by `getSupportedTensorOps()`.
- **`TPlanPolicyAnalysis_analyze()`** (`TPlanLowering.cpp:349`): classifies each dim as Inline/StaticTiled/DynamicTiled. Uses `Plan.getPFForDim(D)` for tile size — does NOT query TTI for M/N.
- **`TPlanTransformer::transform()`** (`TPlanLowering.cpp:461`): creates TPTilingRegion for the innermost tiling dim only. Only one dim is tiled per transform call.
- **`TPTilingRegion::execute()`** (`TPlan.cpp:1225`): emits tiling loops. Dynamic path has `scalar.block` but NO epilogue tensor tiers.
- **`buildCostParams()`** (`TensorCostModel.cpp:17`): hardcodes `PeakTensorFLOPS = VectorFLOPS * 10.0f`.
- **`emitContraction()`** (`TPlanLowering.cpp:574`): called from Contraction recipe execute(). Queries `State.TTI->getTensorContractTileInfo()` to override PF for the K dim only.

---

## Task 1: Add M/N fields + SupportsMasking to TensorContractTileInfo

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h`

- [ ] **Step 1: Add fields to TensorContractTileInfo**

In `TensorISAInfo.h`, add `PrimaryM`, `PrimaryN`, and `SupportsMasking` to the struct:

```cpp
struct TensorContractTileInfo {
  /// Primary tile size along the contraction (K) dimension.
  /// Each main-loop iteration calls tensor.contract with exactly this K.
  unsigned PrimaryK = 0;

  /// Primary tile sizes for output dimensions M and N.
  /// 0 means "use the default PF from TPlan" (no ISA override).
  unsigned PrimaryM = 0;
  unsigned PrimaryN = 0;

  /// Ordered list of epilogue K sizes (largest first).
  /// Each tier handles remaining elements when rem >= EpilogueKSizes[i].
  /// Empty → fall straight to scalar.block after the main loop.
  SmallVector<unsigned, 4> EpilogueKSizes;

  /// When true, the target supports predicated/masked tensor ops for
  /// epilogue processing instead of scalar.block fallback.
  bool SupportsMasking = false;
};
```

Remove the old "reserved for future use" comment block (lines 41-45).

- [ ] **Step 2: Update X86 TTI to return M/N**

In `X86TargetTransformInfo.cpp`, update the three return statements in `getTensorContractTileInfo()`:

```cpp
  if (ST->hasAMXBF16() && ElemTy->isBFloatTy())
    return TensorContractTileInfo{/*PrimaryK=*/32, /*PrimaryM=*/16, /*PrimaryN=*/16,
                                  /*EpilogueKSizes=*/{}, /*SupportsMasking=*/false};

  if (ST->hasAMXINT8() && ElemTy->isIntegerTy(8))
    return TensorContractTileInfo{/*PrimaryK=*/64, /*PrimaryM=*/16, /*PrimaryN=*/16,
                                  /*EpilogueKSizes=*/{}, /*SupportsMasking=*/false};

  if (ST->hasAMXFP16() && ElemTy->isHalfTy())
    return TensorContractTileInfo{/*PrimaryK=*/32, /*PrimaryM=*/16, /*PrimaryN=*/16,
                                  /*EpilogueKSizes=*/{}, /*SupportsMasking=*/false};
```

- [ ] **Step 3: Update AArch64 TTI to return M/N**

In `AArch64TargetTransformInfo.cpp`, update the two return statements:

```cpp
  if (ST->hasMatMulFP32() && ElemTy->isFloatTy())
    return TensorContractTileInfo{/*PrimaryK=*/8, /*PrimaryM=*/8, /*PrimaryN=*/8,
                                  /*EpilogueKSizes=*/{4}, /*SupportsMasking=*/false};

  if (ST->hasSVE2() && ElemTy->isFloatTy())
    return TensorContractTileInfo{/*PrimaryK=*/4, /*PrimaryM=*/0, /*PrimaryN=*/0,
                                  /*EpilogueKSizes=*/{}, /*SupportsMasking=*/true};
```

- [ ] **Step 4: Wire M/N into emitContraction() PF override**

In `TPlanLowering.cpp`, in the `emitContraction()` function, find the existing TTI override block (around line 808):

```cpp
    unsigned TilePF = Spec->PF;
    if (Spec->Mode == DimEmitMode::DynamicTiled && State.TTI) {
      auto TileInfo =
          State.TTI->getTensorContractTileInfo(ElemTy, RankA, RankB, RankC);
      if (TileInfo && TileInfo->PrimaryK > 0)
        TilePF = TileInfo->PrimaryK;
    }
```

Replace with:

```cpp
    unsigned TilePF = Spec->PF;
    if (State.TTI) {
      auto TileInfo =
          State.TTI->getTensorContractTileInfo(ElemTy, RankA, RankB, RankC);
      if (TileInfo) {
        // Override K tile size for DynamicTiled contraction dims.
        if (Spec->Mode == DimEmitMode::DynamicTiled && TileInfo->PrimaryK > 0)
          TilePF = TileInfo->PrimaryK;
        // Override M/N tile sizes for output dims (StaticTiled).
        // OutputDimSet identifies which dims are M vs N; the TTI reports
        // PrimaryM for the outermost and PrimaryN for the next output dim.
        // This mapping is approximate — future work may add per-dim queries.
        if (Spec->Mode == DimEmitMode::StaticTiled) {
          bool IsOutermostOutput =
              (OutputDimSet.find_last() == static_cast<int>(D));
          unsigned ISA_PF = IsOutermostOutput ? TileInfo->PrimaryM
                                              : TileInfo->PrimaryN;
          if (ISA_PF > 0)
            TilePF = ISA_PF;
        }
      }
    }
```

Note: `D` and `OutputDimSet` are already in scope from the `addDimFromPolicy` lambda's capture. Verify this compiles.

- [ ] **Step 5: Build**

```bash
ninja -C build LLVMVectorize LLVMAArch64CodeGen LLVMX86CodeGen 2>&1 | tail -5
```

Expected: clean build (may have pre-existing warnings).

- [ ] **Step 6: Run regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: 44 PASS + 1 XFAIL. No behavior change — existing tests don't specify target ISAs that return non-zero M/N.

- [ ] **Step 7: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h \
        llvm/lib/Target/X86/X86TargetTransformInfo.cpp \
        llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tti: add M/N tile sizes + SupportsMasking to TensorContractTileInfo"
```

---

## Task 2: Consume EpilogueKSizes in TPTilingRegion::execute()

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h` (TPTilingRegion fields)
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp` (TPTilingRegion::execute dynamic path)
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` (pass EpilogueKSizes to TPTilingRegion)

- [ ] **Step 1: Add EpiloguePFs field to TPTilingRegion**

In `TPlan.h`, inside `class TPTilingRegion`, add a field after `OrigKIVPhi`:

```cpp
  /// Epilogue tile sizes for multi-tier epilogue (largest first).
  /// Between the main tensor.body loop and the scalar.block, each tier runs
  /// a guarded tensor.contract with this K size for the remainder.
  SmallVector<unsigned, 4> EpiloguePFs;
```

Update the constructor to accept it:

```cpp
  TPTilingRegion(unsigned Dim, unsigned PF, DimEmitMode Mode,
                 TPBasicBlock *Body, TPBasicBlock *ScalarEpilogue,
                 PHINode *OrigKIVPhi, SmallVector<unsigned, 4> EpiPFs = {})
      : TPBlockBase(TPTilingRegionSC), Dim(Dim), TilingPF(PF), Mode(Mode),
        Body(Body), ScalarEpilogue(ScalarEpilogue), OrigKIVPhi(OrigKIVPhi),
        EpiloguePFs(std::move(EpiPFs)) {}
```

Add accessor:

```cpp
  ArrayRef<unsigned> getEpiloguePFs() const { return EpiloguePFs; }
```

- [ ] **Step 2: Pass EpilogueKSizes from TPlanTransformer to TPTilingRegion**

In `TPlanLowering.cpp`, update `replaceWithTilingRegion()`. First, update the signature to accept epilogue sizes:

```cpp
  TPTilingRegion *replaceWithTilingRegion(TPRegionBlock *Innermost,
                                           const DimEmissionSpec &Spec,
                                           SmallVector<unsigned, 4> EpiPFs = {});
```

In the body, pass them to the constructor:

```cpp
  auto *TR = new TPTilingRegion(Spec.Dim, Spec.PF, Spec.Mode, Body, Epilogue,
                                 KIVPhi, std::move(EpiPFs));
```

In `transform()`, query TTI for epilogue sizes before calling `replaceWithTilingRegion()`:

```cpp
  // Query TTI for epilogue tier sizes.
  SmallVector<unsigned, 4> EpiPFs;
  if (TilingSpec->Mode == DimEmitMode::DynamicTiled) {
    // ElemTy is not known here — the Transformer operates on the TPlan tree
    // before IR emission. Use f32 as a conservative default for TTI queries.
    // emitContraction() will re-query with the actual ElemTy at emission time.
    auto *F32 = Type::getFloatTy(Builder.getContext());
    if (auto TileInfo = Plan.getTTI()
            ? Plan.getTTI()->getTensorContractTileInfo(F32, 2, 2, 2)
            : std::nullopt)
      EpiPFs = SmallVector<unsigned, 4>(TileInfo->EpilogueKSizes);
  }
  replaceWithTilingRegion(Innermost, *TilingSpec, std::move(EpiPFs));
```

> **Implementation note:** `Plan.getTTI()` does not exist. The TTI is on `State.TTI`. Since `transform()` takes `TPTransformState &State`, use `State.TTI` instead. Add it to the code above.

- [ ] **Step 3: Emit epilogue tier loops in TPTilingRegion::execute() dynamic path**

In `TPlan.cpp`, in the dynamic path of `TPTilingRegion::execute()`, after the tensor.body.exit PHI and BEFORE the scalar.block emission (after `ExitIV` is defined, around line 1368), insert the epilogue tier loops:

```cpp
    // ── Epilogue tensor tiers ────────────────────────────────────────────
    // Between main tensor.body and scalar.block, each tier handles remainder
    // elements when rem >= EpiloguePFs[i].
    Value *EpiStart = ExitIV;
    for (unsigned EpiPF : EpiloguePFs) {
      Value *EpiPFVal = ConstantInt::get(IVTy, EpiPF);
      Value *RemVal = B.CreateSub(TCNorm, EpiStart, "epi.rem");
      Value *HasEpi = B.CreateICmpUGE(RemVal, EpiPFVal, "epi.guard");

      BasicBlock *EpiPHBB  = B.GetInsertBlock();
      BasicBlock *EpiHdrBB = BasicBlock::Create(Ctx,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".header").str(), F);
      BasicBlock *EpiBodyBB = BasicBlock::Create(Ctx,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".body").str(), F);
      BasicBlock *EpiLchBB = BasicBlock::Create(Ctx,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".latch").str(), F);
      BasicBlock *EpiExitBB = BasicBlock::Create(Ctx,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".exit").str(), F);

      B.CreateCondBr(HasEpi, EpiHdrBB, EpiExitBB);

      // Epilogue header: IV + bounds check.
      B.SetInsertPoint(EpiHdrBB);
      Value *EpiLimit = B.CreateAdd(EpiStart,
          B.CreateMul(B.CreateUDiv(RemVal, EpiPFVal), EpiPFVal),
          (Twine("tensor.epi.") + Twine(EpiPF) + ".limit").str());
      PHINode *EIV = B.CreatePHI(IVTy, 2,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".iv").str());
      EIV->addIncoming(EpiStart, EpiPHBB);
      Value *EDone = B.CreateICmpUGE(EIV, EpiLimit,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".done").str());
      B.CreateCondBr(EDone, EpiExitBB, EpiBodyBB);

      // Epilogue body: register EIV in EmittedMap + run body recipes.
      B.SetInsertPoint(EpiBodyBB);
      State.EmittedMap[OrigKIVPhi] = EIV;
      Body->execute(State);
      B.CreateBr(EpiLchBB);

      // Epilogue latch: IV += EpiPF.
      IRBuilder<> EpiLB(EpiLchBB);
      Value *ENext = EpiLB.CreateAdd(EIV, EpiPFVal,
          (Twine("tensor.epi.") + Twine(EpiPF) + ".next").str());
      EIV->addIncoming(ENext, EpiLchBB);
      EpiLB.CreateBr(EpiHdrBB);

      // Epilogue exit: merge EpiStart.
      B.SetInsertPoint(EpiExitBB);
      PHINode *NewEpiStart = B.CreatePHI(IVTy, 2, "epi.start");
      NewEpiStart->addIncoming(EpiStart, EpiPHBB);
      NewEpiStart->addIncoming(EpiLimit, EpiHdrBB);
      EpiStart = NewEpiStart;
    }
```

Then update the scalar.block code to use `EpiStart` instead of `ExitIV`:

Replace `Value *ScRem = B.CreateSub(TCNorm, ExitIV, "scalar.rem");` with:

```cpp
    Value *ScRem  = B.CreateSub(TCNorm, EpiStart, "scalar.rem");
```

And update the ScIV incoming: `ScIV->addIncoming(EpiStart, ScPHBB);` — change `ScPHBB` assignment from `TBExit` to the current insert block (which is the last epilogue exit or TBExit if no tiers):

```cpp
    BasicBlock *ScPHBB  = B.GetInsertBlock();
```

This line already exists. Verify the existing code already uses `ScPHBB = B.GetInsertBlock()` (not `ScPHBB = TBExit`). If it uses `TBExit`, change it to `B.GetInsertBlock()`.

- [ ] **Step 4: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 5: Run regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: 44 PASS + 1 XFAIL. No behavior change — existing tests have empty EpiloguePFs.

- [ ] **Step 6: Write AArch64 FMMLA epilogue test**

Create `llvm/test/Transforms/LoopTensorize/aarch64/fmmla-tile-info.ll`:

```llvm
; RUN: opt -passes=loop-tensorize -S \
; RUN:   -mtriple=aarch64-- -mattr=+fp-armv8,+neon,+matmul-fp32 < %s | FileCheck %s
; REQUIRES: aarch64-registered-target
;
; GEMM with dynamic K on AArch64 FMMLA target.
; TTI returns PrimaryK=8, EpilogueKSizes={4}.
; Expected: tensor.body (K=8) + tensor.epi.4 (K=4) + scalar.block.
;
; CHECK-LABEL: @gemm_dynamic_k(
; CHECK: tensor.body.header:
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: tensor.epi.4.header:
; CHECK: tensor.epi.4.body:
; CHECK: call void @llvm.tensor.contract.2d.2d.2d.f32(
; CHECK: scalar.block:

define void @gemm_dynamic_k(ptr %A, ptr %B, ptr %C, i64 %K) {
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
  %ai  = mul i64 %i, 256
  %ak  = add i64 %ai, %k
  %aptr = getelementptr float, ptr %A, i64 %ak
  %bk   = mul i64 %k, 16
  %bj   = add i64 %bk, %j
  %bptr = getelementptr float, ptr %B, i64 %bj
  %av   = load float, ptr %aptr
  %bv   = load float, ptr %bptr
  %mul  = fmul float %av, %bv
  %sum  = fadd float %acc, %mul
  %k.next = add i64 %k, 1
  %k.done = icmp eq i64 %k.next, %K
  br i1 %k.done, label %k.latch, label %k.loop
k.latch:
  %ij   = add i64 %i, %j
  %cptr = getelementptr float, ptr %C, i64 %ij
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

- [ ] **Step 7: Run new test**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/aarch64/fmmla-tile-info.ll
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/test/Transforms/LoopTensorize/aarch64/fmmla-tile-info.ll
git commit -m "tplan: consume EpilogueKSizes in TPTilingRegion::execute() dynamic path"
```

---

## Task 3: Implement AArch64 hasTensorOps() + getSupportedTensorOps()

**Files:**
- Modify: `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.h`
- Modify: `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp`
- Test: `llvm/test/Transforms/LoopTensorize/aarch64/fmmla-tile-info.ll` (update CHECKs)

- [ ] **Step 1: Declare overrides in AArch64TTIImpl**

In `AArch64TargetTransformInfo.h`, near the existing `getTensorContractTileInfo` declaration (line 569), add:

```cpp
  bool hasTensorOps() const override;
  SmallVector<TensorOpDesc> getSupportedTensorOps() const override;
  unsigned getTensorTileSize(Type *ElemTy) const override;
```

- [ ] **Step 2: Implement in AArch64TargetTransformInfo.cpp**

After the existing `getTensorContractTileInfo()` function, add:

```cpp
bool AArch64TTIImpl::hasTensorOps() const {
  return ST->hasMatMulFP32() || ST->hasSVE2() || ST->hasSME();
}

SmallVector<TensorOpDesc> AArch64TTIImpl::getSupportedTensorOps() const {
  SmallVector<TensorOpDesc> Ops;
  if (ST->hasMatMulFP32()) {
    TensorOpDesc D;
    D.OpKind = TensorOpDesc::Kind::MatMul;
    D.M = 8;
    D.N = 8;
    D.K = 8;
    D.AccumType = Type::getFloatTy(ST->getTargetTriple().getContext());
    Ops.push_back(D);
  }
  if (ST->hasSME()) {
    TensorOpDesc D;
    D.OpKind = TensorOpDesc::Kind::OuterProduct;
    D.M = 0; // flexible (streaming SVE length)
    D.N = 0;
    D.K = 0;
    Ops.push_back(D);
  }
  return Ops;
}

unsigned AArch64TTIImpl::getTensorTileSize(Type *ElemTy) const {
  if (ST->hasMatMulFP32() && ElemTy && ElemTy->isFloatTy())
    return 8;
  return 0;
}
```

> **Implementation note:** `ST->getTargetTriple().getContext()` may not exist. Check how X86 handles Type creation. If LLVMContext isn't available from the subtarget, set `AccumType = nullptr` and leave it for future use.

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMAArch64CodeGen LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Run regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass (including the new aarch64 test from Task 2).

- [ ] **Step 5: Commit**

```bash
git add llvm/lib/Target/AArch64/AArch64TargetTransformInfo.h \
        llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp
git commit -m "aarch64/tti: implement hasTensorOps, getSupportedTensorOps, getTensorTileSize"
```

---

## Task 4: Add getTensorOpThroughput() to TTI; replace hardcoded cost model

**Files:**
- Modify: `llvm/include/llvm/Analysis/TargetTransformInfo.h`
- Modify: `llvm/include/llvm/Analysis/TargetTransformInfoImpl.h`
- Modify: `llvm/lib/Analysis/TargetTransformInfo.cpp`
- Modify: `llvm/lib/Target/X86/X86TargetTransformInfo.h`
- Modify: `llvm/lib/Target/X86/X86TargetTransformInfo.cpp`
- Modify: `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.h`
- Modify: `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp`
- Modify: `llvm/lib/Transforms/Vectorize/TensorCostModel.cpp`

- [ ] **Step 1: Declare getTensorOpThroughput in TTI**

In `TargetTransformInfo.h`, after `getTensorContractTileInfo()` (line 2087), add:

```cpp
  /// Returns the peak throughput (ops/cycle) for tensor operations on the
  /// given element type. Returns 0.0 if the target has no tensor support
  /// for that type. Used by the tensor cost model to replace hardcoded
  /// multipliers.
  LLVM_ABI float getTensorOpThroughput(Type *ElemTy) const;
```

- [ ] **Step 2: Add default implementation**

In `TargetTransformInfoImpl.h`, after `getTensorContractTileInfo()` default (around line 1214), add:

```cpp
  float getTensorOpThroughput(Type *) const { return 0.0f; }
```

- [ ] **Step 3: Forward in TargetTransformInfo.cpp**

Find the forwarding functions for the other tensor methods (search for `hasTensorOps` in `TargetTransformInfo.cpp`). Add a forwarding function following the same pattern:

```cpp
float TargetTransformInfo::getTensorOpThroughput(Type *ElemTy) const {
  return TTIImpl->getTensorOpThroughput(ElemTy);
}
```

> **Implementation note:** Check the TTI forwarding pattern — it may use `TTICallback` or direct virtual dispatch via `TTIImpl`. Follow the same pattern as `hasTensorOps()`.

- [ ] **Step 4: Implement for X86**

In `X86TargetTransformInfo.h`, add declaration:

```cpp
  float getTensorOpThroughput(Type *ElemTy) const override;
```

In `X86TargetTransformInfo.cpp`, after `getTensorContractTileInfo()`:

```cpp
float X86TTIImpl::getTensorOpThroughput(Type *ElemTy) const {
  // AMX BF16: TDPBF16PS produces 16×16 FP32 results (512 FMAs) per ~30 cycles.
  if (ST->hasAMXBF16() && ElemTy && ElemTy->isBFloatTy())
    return 512.0f / 30.0f; // ~17 ops/cycle
  // AMX INT8: TDPBSSD produces 16×16 INT32 results (1024 MACs) per ~30 cycles.
  if (ST->hasAMXINT8() && ElemTy && ElemTy->isIntegerTy(8))
    return 1024.0f / 30.0f; // ~34 ops/cycle
  return 0.0f;
}
```

- [ ] **Step 5: Implement for AArch64**

In `AArch64TargetTransformInfo.h`, add declaration:

```cpp
  float getTensorOpThroughput(Type *ElemTy) const override;
```

In `AArch64TargetTransformInfo.cpp`:

```cpp
float AArch64TTIImpl::getTensorOpThroughput(Type *ElemTy) const {
  // NEON FMMLA FP32: 2×4 FP32 results (8 FMAs) per cycle.
  if (ST->hasMatMulFP32() && ElemTy && ElemTy->isFloatTy())
    return 8.0f;
  // SVE2 MATMUL: throughput depends on SVE vector length.
  if (ST->hasSVE2() && ElemTy && ElemTy->isFloatTy())
    return 4.0f; // conservative: 128-bit SVE
  return 0.0f;
}
```

- [ ] **Step 6: Replace hardcoded 10x in buildCostParams()**

In `TensorCostModel.cpp`, replace:

```cpp
  P.PeakTensorFLOPS = TTI.hasTensorOps() ? P.PeakVectorFLOPS * 10.0f : 0.0f;
```

With:

```cpp
  float TensorThroughput = TTI.getTensorOpThroughput(ElemTy);
  if (TensorThroughput > 0.0f) {
    // Convert ops/cycle to FLOPS using estimated clock frequency.
    // 2 GHz is a conservative default; future work may query TTI for clock.
    P.PeakTensorFLOPS = TensorThroughput * 2e9f;
  } else if (TTI.hasTensorOps()) {
    // Fallback: target has tensor ops but no throughput data — use 10x vector.
    P.PeakTensorFLOPS = P.PeakVectorFLOPS * 10.0f;
  } else {
    P.PeakTensorFLOPS = 0.0f;
  }
```

- [ ] **Step 7: Build**

```bash
ninja -C build LLVMVectorize LLVMAArch64CodeGen LLVMX86CodeGen LLVMAnalysis 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 8: Run full regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add llvm/include/llvm/Analysis/TargetTransformInfo.h \
        llvm/include/llvm/Analysis/TargetTransformInfoImpl.h \
        llvm/lib/Analysis/TargetTransformInfo.cpp \
        llvm/lib/Target/X86/X86TargetTransformInfo.h \
        llvm/lib/Target/X86/X86TargetTransformInfo.cpp \
        llvm/lib/Target/AArch64/AArch64TargetTransformInfo.h \
        llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp \
        llvm/lib/Transforms/Vectorize/TensorCostModel.cpp
git commit -m "tti: add getTensorOpThroughput(); use per-ISA throughput in tensor cost model"
```

---

## Notes for Implementer

- **`emitContraction()` context for M/N override (Task 1 Step 4):** The lambda `addDimFromPolicy` captures `D` as its argument. The `OutputDimSet` is computed earlier in `emitContraction()` (line ~700). Make sure the `TileInfo` TTI query is inside the lambda or can reference `ElemTy`, `RankA`, `RankB`, `RankC` which are in scope. If not, hoist the TTI query above the lambda and store the M/N values.

- **`Plan.getTTI()` does not exist (Task 2 Step 2):** Use `State.TTI` which is already available in `TPlanTransformer::transform()`. The Transformer's `transform()` method takes `TPTransformState &State` which has `const TargetTransformInfo *TTI`.

- **AArch64 LLVMContext (Task 3 Step 2):** `Type::getFloatTy()` needs an `LLVMContext&`. The TTI implementation may not have direct access. Either set `AccumType = nullptr` or find the context from `ST` (subtarget). Check how X86 handles this — X86 doesn't set AccumType either.

- **TTI forwarding pattern (Task 4 Step 3):** The TargetTransformInfo uses a concept-model pattern with `TTIImpl`. Check how `hasTensorOps()` is forwarded — it may go through `TTIImpl->hasTensorOps()` directly, or through a `Concept` virtual dispatch. Follow the same pattern exactly.

- **Throughput values are approximate (Task 4 Steps 4-5):** The ops/cycle numbers are rough estimates from public ISA documentation. They're good enough for roofline modeling but should be refined with microbenchmarks.
