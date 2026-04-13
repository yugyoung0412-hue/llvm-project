# TPlanLowering Refactor: TPlan-Driven IR Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all pre-`execute()` LLVM IR surgery from `TPlanLowering_lower()`; IR is generated solely by walking TPlan node `execute()` methods.

**Architecture:** Add `IsSubsumed` flag to `TPRecipeBase` so tensor-absorbed recipes no-op during execute. Add `TPGuardBlock` (runtime profitability guard) and `TPTilingRegion` (tiling loop structure) as new `TPBlockBase` subclasses. A new `TPlanTransformer` pass rewrites the TPlan tree — wrapping it in `TPGuardBlock` and replacing the innermost `TPRegionBlock` with `TPTilingRegion` — before `execute()` is called. `TPlanLowering_lower()` then becomes: widen → match → analyze → transform → execute, with zero IR mutations before execute.

**Tech Stack:** C++ (LLVM), `IRBuilder<>`, SCEV, LLVM lit tests.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `llvm/include/llvm/Transforms/Vectorize/TPlan.h` | Add `IsSubsumed` to `TPRecipeBase`; declare `TPGuardBlock`, `TPTilingRegion` |
| Modify | `llvm/lib/Transforms/Vectorize/TPlan.cpp` | Implement `TPGuardBlock::execute()`, `TPTilingRegion::execute()` |
| Modify | `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Add `TPlanPolicyAnalysis`, `TPlanTransformer`; update `emitContraction()`; simplify `TPlanLowering_lower()` |
| Modify | `llvm/lib/Transforms/Vectorize/CMakeLists.txt` | No new files needed (all in existing files) |
| Test | `llvm/test/Transforms/LoopTensorize/basic/skeleton-guard.ll` | Existing — must still pass |
| Test | `llvm/test/Transforms/LoopTensorize/basic/tiling-dynamic-k.ll` | Existing — must still pass |

---

## Background

Key types you need to understand before starting:

- **`TPBlockBase`** (`TPlan.h:280`): base of all TPlan CFG nodes. Has virtual `execute(TPTransformState&)`.
- **`TPRegionBlock`** (`TPlan.h:447`): represents one loop level. Has `Inner` (nested region), `Entry` block, etc.
- **`TPBasicBlock`** (`TPlan.h:373`): owns an `iplist<TPRecipeBase>`. Recipes execute in order.
- **`TPRecipeBase`** (`TPlan.h:639`): base recipe. `execute()` emits IR; `IsSubsumed=true` means skip.
- **`TPWidenInductionRecipe`** (`TPlan.h:945`): holds `IVPhi` (the loop's PHINode) and `DimIndex`.
- **`TPTransformState`** (`TPlan.h:1468`): carries `IRBuilder<>`, `ValueMap` (recipe→Value\*), `EmittedMap` (original IR→cloned IR), `EmittedContractions`, `Policy`, `PrebuiltTilingPtr`.
- **`EmittedMap`**: when a recipe clones an instruction and inserts it, it stores `EmittedMap[origInstr] = clone`. `remapClone()` uses this to fix operand dominance.
- **`emitContraction()`** (`TPlanLowering.cpp:561`): called from `TPWidenRecipe::execute()` when `Kind==Contraction`. Uses `decomposePtrForDims()` to find base pointers and strides, then emits tiling loops + `tensor.contract` call.
- **`buildEmissionPolicy()`** (`TPlanLowering.cpp`): classifies each dim as `Inline/StaticTiled/DynamicTiled`. Result stored in `State.Policy`.
- **`createTensorizedLoopSkeleton()`** (`TPlanSkeleton.cpp`): clones the outermost loop as scalar fallback, inserts `tensor.guard` block between predecessor and preheader. Called *before* execute() — this is what we are removing.
- **`preBuildTilingBlocks()`** (`TPlanLowering.cpp:1906`): creates floating tiling loop BBs (anchored to a temporary `AnchorBB`) *before* execute(). Also being removed.

---

## Task 1: Add `IsSubsumed` flag to `TPRecipeBase`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h` (near `TPRecipeBase`, line ~639)
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp` (update print if needed)

- [ ] **Step 1: Add field + accessors to TPRecipeBase**

In `TPlan.h`, inside `class TPRecipeBase`, after the `SubclassID` field:

```cpp
  /// When true, execute() is a no-op. Set by TPlanTransformer on recipes
  /// that are absorbed into a tensor op (e.g. loads/fmul/fadd subsumed by
  /// tensor.contract). ScalarEpilogue copies always have IsSubsumed=false.
  bool IsSubsumed = false;

public:
  bool isSubsumed() const { return IsSubsumed; }
  void setSubsumed(bool V = true) { IsSubsumed = V; }
```

- [ ] **Step 2: Guard all recipe execute() methods**

In `TPlanLowering.cpp`, at the very top of each recipe's `execute()` that should respect the flag (TPWidenRecipe, TPWidenGEPRecipe, TPWidenLoadRecipe, TPWidenStoreRecipe, TPWidenCastRecipe, TPWidenIntOrFpInductionRecipe, TPReductionPHIRecipe), add:

```cpp
void TPWidenRecipe::execute(TPTransformState &State) const {
  if (isSubsumed()) return;
  // ... existing body unchanged ...
}
```

Do the same for: `TPWidenGEPRecipe::execute`, `TPWidenLoadRecipe::execute`, `TPWidenStoreRecipe::execute`, `TPWidenCastRecipe::execute`, `TPWidenIntOrFpInductionRecipe::execute`.

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Run regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: 44 PASS + 1 XFAIL. No behavior change — `IsSubsumed` is `false` everywhere by default.

- [ ] **Step 5: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan: add IsSubsumed flag to TPRecipeBase; guard recipe execute() methods"
```

---

## Task 2: Declare `TPGuardBlock` and `TPTilingRegion` in `TPlan.h`

**Files:**
- Modify: `llvm/include/llvm/Transforms/Vectorize/TPlan.h`

Add both class declarations after the `TPRegionBlock` class (line ~530).

- [ ] **Step 1: Add `TPGuardBlock` declaration**

```cpp
//===----------------------------------------------------------------------===//
// TPGuardBlock — runtime profitability guard (TC >=u PF ? tensor : scalar)
//===----------------------------------------------------------------------===//
/// Represents the runtime guard structure:
///
///   [Pred] → [tensor.guard: icmp uge TC, PF]
///                  | true              | false
///             [TensorPath]        [ScalarPath (loop clone)]
///                  \                  /
///               [MergeBB (original loop exit)]
///
/// execute() creates the guard BasicBlock, clones the outermost loop as
/// the scalar fallback, wires the CFG, then calls TensorPath->execute().
/// ScalarPath is a complete loop clone; no further execute() is needed.
class TPGuardBlock : public TPBlockBase {
  /// Outermost loop to guard (and clone as scalar fallback).
  Loop *OutermostLoop;
  /// Runtime trip-count Value* for the guarded dim. Dominated by loop pred.
  Value *RuntimeTC;
  /// Guard threshold (PF of the guarded dim).
  unsigned PF;
  /// The tensor path — the original TPRegionBlock tree.
  TPBlockBase *TensorPath;

public:
  TPGuardBlock(Loop *L, Value *TC, unsigned PF, TPBlockBase *Tensor)
      : TPBlockBase(TPGuardBlockSC), OutermostLoop(L), RuntimeTC(TC),
        PF(PF), TensorPath(Tensor) {}

  Loop        *getOutermostLoop() const { return OutermostLoop; }
  Value       *getRuntimeTC()     const { return RuntimeTC; }
  unsigned     getPF()            const { return PF; }
  TPBlockBase *getTensorPath()    const { return TensorPath; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPGuardBlockSC;
  }
};
```

- [ ] **Step 2: Add `TPTilingRegion` declaration**

```cpp
//===----------------------------------------------------------------------===//
// TPTilingRegion — tiling loop structure replacing innermost TPRegionBlock
//===----------------------------------------------------------------------===//
/// Replaces the innermost TPRegionBlock (K-loop) in the TPlan tree.
/// execute() emits:
///   - Static tiling: tile.header / tile.body / tile.latch / tile.exit loop
///   - Dynamic tiling: tensor.body fixed-count loop + scalar.block epilogue
///
/// Body contains original k-loop recipes with IsSubsumed flags set by
/// TPlanTransformer. Non-subsumed recipes (WIDEN-GEP, Contraction) execute
/// normally; subsumed recipes (WIDEN-LOAD, fmul, fadd) are no-ops.
/// ScalarEpilogue (dynamic-K only) has all recipes IsSubsumed=false for
/// K%PF scalar remainder processing.
class TPTilingRegion : public TPBlockBase {
  unsigned Dim;                     ///< TPlan dim index being tiled (K=0)
  unsigned PF;                      ///< Tile size (prefetch factor)
  DimEmitMode Mode;                 ///< StaticTiled or DynamicTiled
  TPBasicBlock *Body;               ///< Original k-loop recipes (IsSubsumed set)
  TPBasicBlock *ScalarEpilogue;     ///< K%PF remainder (DynamicTiled only; null otherwise)
  /// The original PHINode for the K induction variable.
  /// execute() registers EmittedMap[OrigKIVPhi] = TileIV so that WIDEN-GEP
  /// recipes naturally emit tile-corner GEPs via remapClone().
  PHINode *OrigKIVPhi;

public:
  TPTilingRegion(unsigned Dim, unsigned PF, DimEmitMode Mode,
                 TPBasicBlock *Body, TPBasicBlock *ScalarEpilogue,
                 PHINode *OrigKIVPhi)
      : TPBlockBase(TPTilingRegionSC), Dim(Dim), PF(PF), Mode(Mode),
        Body(Body), ScalarEpilogue(ScalarEpilogue), OrigKIVPhi(OrigKIVPhi) {}

  unsigned      getDim()             const { return Dim; }
  unsigned      getPF()              const { return PF; }
  DimEmitMode   getMode()            const { return Mode; }
  TPBasicBlock *getBody()            const { return Body; }
  TPBasicBlock *getScalarEpilogue()  const { return ScalarEpilogue; }
  PHINode      *getOrigKIVPhi()      const { return OrigKIVPhi; }

  void execute(TPTransformState &State) override;
  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;

  static bool classof(const TPBlockBase *B) {
    return B->getTPBlockID() == TPTilingRegionSC;
  }
};
```

- [ ] **Step 3: Add new SubclassID values to the block ID enum**

Find the existing block ID enum in `TPlan.h` (look for `TPBlockBase::getTPBlockID()` or similar enum near `TPBlockBase`). Add:

```cpp
  TPGuardBlockSC,
  TPTilingRegionSC,
```

- [ ] **Step 4: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: compile errors about missing `execute()` implementations — that's fine (they're forward declared). Fix by adding stub implementations in `TPlan.cpp`:

```cpp
void TPGuardBlock::execute(TPTransformState &State) {
  llvm_unreachable("TPGuardBlock::execute() not yet implemented");
}
void TPGuardBlock::print(raw_ostream &OS, unsigned, TPSlotTracker &) const {
  OS << "TPGuardBlock (TC >= " << PF << ")\n";
}
void TPTilingRegion::execute(TPTransformState &State) {
  llvm_unreachable("TPTilingRegion::execute() not yet implemented");
}
void TPTilingRegion::print(raw_ostream &OS, unsigned, TPSlotTracker &) const {
  OS << "TPTilingRegion dim=" << Dim << " PF=" << PF << "\n";
}
```

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add llvm/include/llvm/Transforms/Vectorize/TPlan.h \
        llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: declare TPGuardBlock and TPTilingRegion block types"
```

---

## Task 3: Implement `TPGuardBlock::execute()`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

This absorbs the logic of `createTensorizedLoopSkeleton()` (`TPlanSkeleton.cpp:34–147`). Read that function before implementing — the logic is identical, just moved into `execute()`.

- [ ] **Step 1: Add includes to TPlan.cpp if missing**

Near the top of `TPlan.cpp`, ensure these are present:

```cpp
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
```

- [ ] **Step 2: Implement `TPGuardBlock::execute()`**

Replace the `llvm_unreachable` stub in `TPlan.cpp`:

```cpp
void TPGuardBlock::execute(TPTransformState &State) {
  IRBuilder<> &B = State.Builder;
  LoopInfo &LI = *State.LI;
  DominatorTree &DT = *State.DT;

  BasicBlock *OrigPreheader = OutermostLoop->getLoopPreheader();
  assert(OrigPreheader && "TPGuardBlock: loop must have a unique preheader");
  BasicBlock *ExitBB = OutermostLoop->getExitBlock();
  assert(ExitBB && "TPGuardBlock: loop must have a single exit block");
  BasicBlock *OrigPred = OrigPreheader->getSinglePredecessor();
  assert(OrigPred && "TPGuardBlock: preheader must have a single predecessor");

  // Step 1: Clone outermost loop as scalar fallback.
  SmallVector<BasicBlock *, 16> ClonedBlocks;
  ValueToValueMapTy SkelVMap;
  Loop *ScalarLoop = cloneLoopWithPreheader(
      OrigPreheader, OrigPred, OutermostLoop, SkelVMap,
      ".scalar", &LI, &DT, ClonedBlocks);
  assert(ScalarLoop && "TPGuardBlock: cloneLoopWithPreheader() failed");
  remapInstructionsInBlocks(ClonedBlocks, SkelVMap);
  BasicBlock *ScalarPreheader = cast<BasicBlock>(SkelVMap[OrigPreheader]);

  // Step 2: Create GuardBB between OrigPred and OrigPreheader.
  LLVMContext &Ctx = OrigPreheader->getContext();
  Function *F = OrigPreheader->getParent();
  BasicBlock *GuardBB = BasicBlock::Create(Ctx, "tensor.guard", F, OrigPreheader);

  // Redirect OrigPred → GuardBB.
  Instruction *PredTerm = OrigPred->getTerminator();
  for (unsigned I = 0, E = PredTerm->getNumSuccessors(); I < E; ++I)
    if (PredTerm->getSuccessor(I) == OrigPreheader)
      PredTerm->setSuccessor(I, GuardBB);

  // Fix PHIs in OrigPreheader and ScalarPreheader.
  for (PHINode &Phi : OrigPreheader->phis()) {
    int Idx = Phi.getBasicBlockIndex(OrigPred);
    if (Idx >= 0) Phi.setIncomingBlock(static_cast<unsigned>(Idx), GuardBB);
  }
  for (PHINode &Phi : ScalarPreheader->phis()) {
    int Idx = Phi.getBasicBlockIndex(OrigPred);
    if (Idx >= 0) Phi.setIncomingBlock(static_cast<unsigned>(Idx), GuardBB);
  }

  // Emit guard: icmp uge RuntimeTC, PF → condbr tensor : scalar.
  {
    IRBuilder<> GB(GuardBB);
    Value *PFVal = ConstantInt::get(RuntimeTC->getType(), PF);
    Value *Cond = GB.CreateICmpUGE(RuntimeTC, PFVal, "tensor.profitable");
    GB.CreateCondBr(Cond, OrigPreheader, ScalarPreheader);
  }

  // Update DominatorTree.
  DT.addNewBlock(GuardBB, OrigPred);
  DT.changeImmediateDominator(OrigPreheader, GuardBB);
  DT.changeImmediateDominator(ScalarPreheader, GuardBB);

  LLVM_DEBUG(dbgs() << "TPGuardBlock: guard inserted before "
                    << OutermostLoop->getName() << "\n");

  // Step 3: Execute tensor path (TensorPath is the original TPRegionBlock tree).
  TensorPath->execute(State);
}
```

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: implement TPGuardBlock::execute() — absorbs createTensorizedLoopSkeleton"
```

---

## Task 4: Implement `TPTilingRegion::execute()` — static tiling path

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

This absorbs the static-tiling code path from `emitContraction()` (`TPlanLowering.cpp:1060–1157`). Read that code first. The key difference: loop structure is created here (not in `emitContraction`), and `EmittedMap[OrigKIVPhi] = TileIV` is registered so WIDEN-GEP recipes automatically compute tile-corner GEPs via `remapClone()`.

- [ ] **Step 1: Implement static tiling in `TPTilingRegion::execute()`**

Replace the `llvm_unreachable` stub in `TPlan.cpp`:

```cpp
void TPTilingRegion::execute(TPTransformState &State) {
  IRBuilder<> &B = State.Builder;
  LLVMContext &Ctx = B.getContext();
  Function *F = B.GetInsertBlock()->getParent();

  // Current insert block is the k.loop BB (set by parent TPRegionBlock).
  BasicBlock *KLoopBB = B.GetInsertBlock();

  // Capture successor for wiring after tiling (k.latch or equivalent).
  Instruction *OrigTerm = KLoopBB->getTerminator();
  // The k.loop branches back to itself (self-edge) and to k.latch.
  // The non-self successor is what we need to wire to the tiling exit.
  BasicBlock *OrigSuccessor = nullptr;
  for (BasicBlock *Succ : successors(KLoopBB))
    if (Succ != KLoopBB) { OrigSuccessor = Succ; break; }

  // Erase original k-loop terminator.
  OrigTerm->eraseFromParent();

  // Remove k-loop PHI self-edges (they will be driven by our tiling loop).
  for (PHINode &Phi : KLoopBB->phis()) {
    int SelfIdx = Phi.getBasicBlockIndex(KLoopBB);
    if (SelfIdx >= 0)
      Phi.removeIncomingValue(SelfIdx, /*DeletePHIIfEmpty=*/false);
  }

  if (Mode == DimEmitMode::StaticTiled) {
    // ── Static tiling path ────────────────────────────────────────────────
    // Trip count for the tiling dim (K). Expanded by TPlanTransformer before
    // execute(), stored in State (see Task 5).
    Value *TCVal = State.TilingTCVal;
    assert(TCVal && "TPTilingRegion: TilingTCVal not set for StaticTiled dim");

    Value *PFVal   = B.getInt64(PF);
    Value *Trips   = B.CreateUDiv(TCVal, PFVal, "tile.trips");
    Value *Limit   = B.CreateMul(Trips, PFVal, "tile.limit");

    // Create tile loop blocks.
    BasicBlock *TileHeader = BasicBlock::Create(Ctx, "tile.header", F);
    BasicBlock *TileBody   = BasicBlock::Create(Ctx, "tile.body",   F);
    BasicBlock *TileLatch  = BasicBlock::Create(Ctx, "tile.latch",  F);
    BasicBlock *TileExit   = BasicBlock::Create(Ctx, "tile.exit",   F);

    // KLoopBB → TileHeader.
    B.CreateBr(TileHeader);

    // TileHeader: IV PHI + bounds check.
    B.SetInsertPoint(TileHeader);
    PHINode *TileIV = B.CreatePHI(B.getInt64Ty(), 2, "tile.iv");
    TileIV->addIncoming(B.getInt64(0), KLoopBB);
    Value *Done = B.CreateICmpUGE(TileIV, Limit, "tile.done");
    B.CreateCondBr(Done, TileExit, TileBody);

    // Register TileIV in EmittedMap so WIDEN-GEP remapClone() substitutes
    // OrigKIVPhi → TileIV, producing tile-corner GEPs automatically.
    State.EmittedMap[OrigKIVPhi] = TileIV;

    // TileBody: run body recipes. WIDEN-GEP emits tile-corner GEPs;
    // Contraction emits tensor.contract; subsumed recipes are no-ops.
    B.SetInsertPoint(TileBody);
    Body->execute(State);
    B.CreateBr(TileLatch);

    // TileLatch: IV += PF, back to header.
    B.SetInsertPoint(TileLatch);
    Value *TileNext = B.CreateAdd(TileIV, PFVal, "tile.next");
    TileIV->addIncoming(TileNext, TileLatch);
    B.CreateBr(TileHeader);

    // TileExit → original successor.
    B.SetInsertPoint(TileExit);
    if (OrigSuccessor)
      B.CreateBr(OrigSuccessor);

    // Clear TileIV mapping so it doesn't pollute subsequent regions.
    State.EmittedMap.erase(OrigKIVPhi);

  } else {
    // DynamicTiled path: implemented in Task 5.
    llvm_unreachable("TPTilingRegion: DynamicTiled path not yet implemented");
  }
}
```

- [ ] **Step 2: Add `TilingTCVal` field to `TPTransformState` in `TPlan.h`**

Inside `struct TPTransformState`, after `PrebuiltTilingPtr`:

```cpp
  /// Set by TPlanTransformer before execute() for the tiling dim's trip-count.
  /// TPTilingRegion::execute() reads this to compute tile loop bounds.
  Value *TilingTCVal = nullptr;
```

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan: implement TPTilingRegion::execute() static tiling path"
```

---

## Task 5: Implement `TPTilingRegion::execute()` — dynamic tiling path + ScalarEpilogue

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlan.cpp`

The dynamic tiling path mirrors `emitContraction()`'s dynamic path (`TPlanLowering.cpp:1159–1360`). Read that carefully. Key structure: `tensor.body.header` (fixed-count loop, TC = K/PF\*PF) + `scalar.block` (epilogue for K%PF elements).

- [ ] **Step 1: Implement DynamicTiled path in `TPTilingRegion::execute()`**

Replace `llvm_unreachable("TPTilingRegion: DynamicTiled path not yet implemented")` with:

```cpp
  } else {
    // ── Dynamic tiling path ───────────────────────────────────────────────
    // TCVal: runtime K trip count. Expanded by TPlanTransformer.
    Value *TCVal = State.TilingTCVal;
    assert(TCVal && "TPTilingRegion: TilingTCVal not set for DynamicTiled dim");

    Value *PFVal  = B.getInt64(PF);
    Value *Trips  = B.CreateUDiv(TCVal, PFVal, "tensor.body.trips");
    Value *Limit  = B.CreateMul(Trips, PFVal, "tensor.body.limit");
    Value *Guard  = B.CreateICmpUGE(TCVal, PFVal, "tensor.body.guard");

    BasicBlock *TBHeader = BasicBlock::Create(Ctx, "tensor.body.header", F);
    BasicBlock *TBBody   = BasicBlock::Create(Ctx, "tensor.body.body",   F);
    BasicBlock *TBLatch  = BasicBlock::Create(Ctx, "tensor.body.latch",  F);
    BasicBlock *TBExit   = BasicBlock::Create(Ctx, "tensor.body.exit",   F);

    B.CreateCondBr(Guard, TBHeader, TBExit);

    // tensor.body.header: IV + exit check.
    B.SetInsertPoint(TBHeader);
    PHINode *TBIV = B.CreatePHI(B.getInt64Ty(), 2, "tensor.body.iv");
    TBIV->addIncoming(B.getInt64(0), KLoopBB);
    Value *TBDone = B.CreateICmpUGE(TBIV, Limit, "tensor.body.done");
    B.CreateCondBr(TBDone, TBExit, TBBody);

    // Register TileIV so WIDEN-GEP emits tensor.body tile-corner GEPs.
    State.EmittedMap[OrigKIVPhi] = TBIV;

    // tensor.body.body: recipes (GEP → tile corner, Contraction → tensor.contract).
    B.SetInsertPoint(TBBody);
    Body->execute(State);
    B.CreateBr(TBLatch);

    // tensor.body.latch: IV += PF.
    B.SetInsertPoint(TBLatch);
    Value *TBNext = B.CreateAdd(TBIV, PFVal, "tensor.body.next");
    TBIV->addIncoming(TBNext, TBLatch);
    B.CreateBr(TBHeader);

    // tensor.body.exit: ExitIV = IV at exit (0 if guard was false).
    B.SetInsertPoint(TBExit);
    PHINode *ExitIV = B.CreatePHI(B.getInt64Ty(), 2, "tensor.body.exit.iv");
    ExitIV->addIncoming(B.getInt64(0), KLoopBB);
    ExitIV->addIncoming(TBIV, TBHeader);

    // scalar.block: process K%PF remainder elements.
    Value *ScRem  = B.CreateSub(TCVal, ExitIV, "scalar.rem");
    Value *HasSc  = B.CreateICmpUGT(ScRem, B.getInt64(0), "scalar.guard");

    BasicBlock *ScPHBB   = B.GetInsertBlock();
    BasicBlock *ScBody   = BasicBlock::Create(Ctx, "scalar.block",      F);
    BasicBlock *ScExit   = BasicBlock::Create(Ctx, "scalar.block.exit", F);
    B.CreateCondBr(HasSc, ScBody, ScExit);

    // scalar.block body: IV over [ExitIV, TCVal).
    B.SetInsertPoint(ScBody);
    PHINode *ScIV = B.CreatePHI(B.getInt64Ty(), 2, "scalar.iv");
    ScIV->addIncoming(ExitIV, ScPHBB);

    // Register ScIV so ScalarEpilogue WIDEN-GEP emits scalar access GEPs.
    State.EmittedMap[OrigKIVPhi] = ScIV;
    if (ScalarEpilogue)
      ScalarEpilogue->execute(State);

    Value *ScNext = B.CreateAdd(ScIV, B.getInt64(1), "scalar.next");
    ScIV->addIncoming(ScNext, ScBody);
    Value *ScDone = B.CreateICmpUGE(ScNext, TCVal, "scalar.done");
    B.CreateCondBr(ScDone, ScExit, ScBody);

    B.SetInsertPoint(ScExit);
    if (OrigSuccessor)
      B.CreateBr(OrigSuccessor);

    // Clear mapping.
    State.EmittedMap.erase(OrigKIVPhi);
  }
```

- [ ] **Step 2: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlan.cpp
git commit -m "tplan: implement TPTilingRegion::execute() dynamic tiling + scalar.block"
```

---

## Task 6: Extract `TPlanPolicyAnalysis` + implement `TPlanTransformer`

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

`TPlanPolicyAnalysis` is a pure extraction of `buildEmissionPolicy()` with no behavioral change. `TPlanTransformer` is new code that rewrites the TPlan tree.

- [ ] **Step 1: Extract `TPlanPolicyAnalysis`**

Rename the existing `buildEmissionPolicy()` function to `TPlanPolicyAnalysis_analyze()` (or wrap it in a class). Keep the exact same logic. Update the call site in `TPlanLowering_lower()` to use the new name. Verify tests still pass.

```bash
ninja -C build LLVMVectorize && ./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -5
```

Expected: 44 PASS + 1 XFAIL.

- [ ] **Step 2: Add `TPlanTransformer` class in `TPlanLowering.cpp`**

Add before `TPlanLowering_lower()`. The transformer runs after `PatternMatcher` and produces the new TPlan structure:

```cpp
/// Rewrites the TPlan tree to replace IR-level surgery with explicit TPlan
/// nodes. After transform():
///  - If Policy has a DynamicTiled dim: a TPGuardBlock wraps the root TPRegionBlock.
///  - The innermost TPRegionBlock (K-loop) is replaced by TPTilingRegion.
///  - Recipes in TPTilingRegion::Body have IsSubsumed set.
///  - State.TilingTCVal is populated with the expanded trip-count Value*.
/// IR is NOT modified; only the TPlan tree changes.
class TPlanTransformer {
  TPlan &Plan;
  const EmissionPolicy &Policy;
  ScalarEvolution &SE;
  LoopInfo &LI;
  SCEVExpander &Expander;
  IRBuilder<> &Builder;
  const RecipeClassMap &CM;
  Loop *OutermostLoop;

public:
  TPlanTransformer(TPlan &P, const EmissionPolicy &Pol, ScalarEvolution &SE,
                   LoopInfo &LI, SCEVExpander &Exp, IRBuilder<> &B,
                   const RecipeClassMap &CM,
                   const DenseMap<unsigned, Loop *> &DimToLoop)
      : Plan(P), Policy(Pol), SE(SE), LI(LI), Expander(Exp), Builder(B),
        CM(CM) {
    // Find outermost loop (highest dim index).
    OutermostLoop = nullptr;
    unsigned MaxDim = 0;
    for (const auto &[D, L] : DimToLoop)
      if (!OutermostLoop || D > MaxDim) { MaxDim = D; OutermostLoop = L; }
  }

  /// Entry point: transform Plan according to Policy.
  /// Populates State.TilingTCVal before returning.
  void transform(TPTransformState &State);

private:
  /// Find the innermost TPRegionBlock in the Plan tree.
  TPRegionBlock *findInnermostRegion();

  /// Mark recipes in Body that are subsumed by tensor.contract.
  /// Subsumed: WIDEN-LOAD, WIDEN fmul/fadd (Kind!=Contraction), WIDEN-INDUCTION.
  /// Not subsumed: WIDEN-GEP (tile-corner GEPs needed), Contraction.
  void markSubsumedRecipes(TPBasicBlock *Body);

  /// Clone Body recipes for ScalarEpilogue with all IsSubsumed=false.
  TPBasicBlock *buildScalarEpilogue(TPBasicBlock *Body);

  /// Replace InnermostRegion with a TPTilingRegion in the parent's child slot.
  TPTilingRegion *replaceWithTilingRegion(TPRegionBlock *InnermostRegion,
                                           const DimEmissionSpec &Spec,
                                           Value *TCVal);

  /// Wrap Plan's root TPBlockBase in a TPGuardBlock.
  void insertGuardBlock(const DimEmissionSpec &Spec, Value *TCVal);
};
```

- [ ] **Step 3: Implement `TPlanTransformer::transform()`**

```cpp
void TPlanTransformer::transform(TPTransformState &State) {
  // Find the DynamicTiled or StaticTiled dim spec.
  const DimEmissionSpec *TilingSpec = nullptr;
  for (const auto &S : Policy.Specs)
    if (S.Mode == DimEmitMode::DynamicTiled || S.Mode == DimEmitMode::StaticTiled)
      TilingSpec = &S;
  if (!TilingSpec) return; // Inline only — no tiling needed.

  // Expand trip-count SCEV for the tiling dim into the loop's predecessor BB
  // (before the outermost loop preheader), so it dominates all tiling BBs.
  const SCEV *TCSCEV = Plan.getTCForDim(TilingSpec->Dim);
  assert(TCSCEV && "TPlanTransformer: no TC SCEV for tiling dim");

  BasicBlock *InsertBB = OutermostLoop
      ? OutermostLoop->getLoopPreheader()->getSinglePredecessor()
      : nullptr;
  assert(InsertBB && "TPlanTransformer: cannot find expansion insertion BB");

  Value *TCVal = Expander.expandCodeFor(
      TCSCEV, Builder.getInt64Ty(), InsertBB->getTerminator());

  // If dynamic: add 1 (SCEV gives backedge-taken count, not trip count).
  if (TilingSpec->Mode == DimEmitMode::DynamicTiled) {
    IRBuilder<> PredB(InsertBB->getTerminator());
    Value *BTC = Expander.expandCodeFor(TCSCEV, Builder.getInt64Ty(),
                                         InsertBB->getTerminator());
    TCVal = PredB.CreateAdd(BTC, PredB.getInt64(1), "tc.guard");
  }

  State.TilingTCVal = TCVal;

  // Replace innermost TPRegionBlock with TPTilingRegion.
  TPRegionBlock *Innermost = findInnermostRegion();
  assert(Innermost && "TPlanTransformer: no innermost TPRegionBlock");
  replaceWithTilingRegion(Innermost, *TilingSpec, TCVal);

  // If DynamicTiled: wrap root in TPGuardBlock.
  if (TilingSpec->Mode == DimEmitMode::DynamicTiled)
    insertGuardBlock(*TilingSpec, TCVal);
}
```

- [ ] **Step 4: Implement helper methods**

```cpp
TPRegionBlock *TPlanTransformer::findInnermostRegion() {
  // Walk the TPlan tree: innermost TPRegionBlock has Inner == nullptr.
  TPBlockBase *Cur = Plan.getEntry();
  TPRegionBlock *Last = nullptr;
  while (Cur) {
    if (auto *R = dyn_cast<TPRegionBlock>(Cur)) {
      Last = R;
      Cur = R->getInner(); // move to the nested region
    } else break;
  }
  return Last;
}

void TPlanTransformer::markSubsumedRecipes(TPBasicBlock *Body) {
  for (TPRecipeBase &R : *Body) {
    TensorOpKind Kind = CM.count(&R) ? CM.at(&R).Kind : TensorOpKind::Scalar;
    switch (R.getTPRecipeID()) {
    case TPRecipeBase::TPWidenLoadSC:
    case TPRecipeBase::TPWidenStoreSC:
      R.setSubsumed(true); break;
    case TPRecipeBase::TPWidenSC:
      // Subsumed unless it IS the Contraction recipe.
      if (Kind != TensorOpKind::Contraction)
        R.setSubsumed(true);
      break;
    case TPRecipeBase::TPWidenIntOrFpInductionSC:
      // K IV recipe: subsumed (TileIV takes its place via EmittedMap).
      R.setSubsumed(true); break;
    default:
      break; // WIDEN-GEP, Contraction, Reduction-PHI: not subsumed.
    }
  }
}

TPBasicBlock *TPlanTransformer::buildScalarEpilogue(TPBasicBlock *Body) {
  // Clone Body with all IsSubsumed=false for the scalar.block epilogue.
  // The epilogue recipes are shallow copies — they reference the same
  // original IR instructions but emit fresh clones via remapClone().
  auto *Epi = new TPBasicBlock("scalar.epilogue");
  for (TPRecipeBase &R : *Body) {
    TPRecipeBase *Clone = R.clone(); // assumes TPRecipeBase has clone()
    Clone->setSubsumed(false);
    Epi->appendRecipe(Clone);
  }
  return Epi;
}

TPTilingRegion *TPlanTransformer::replaceWithTilingRegion(
    TPRegionBlock *Innermost, const DimEmissionSpec &Spec, Value *TCVal) {
  // Extract the k-loop TPBasicBlock (the ir-bb<k.loop> block inside Innermost).
  TPBasicBlock *KBody = Innermost->getBodyBlock(); // returns ir-bb<k.loop> TPBasicBlock

  // Mark subsumed recipes on the tiling body.
  markSubsumedRecipes(KBody);

  // Build ScalarEpilogue for DynamicTiled dims.
  TPBasicBlock *Epilogue = (Spec.Mode == DimEmitMode::DynamicTiled)
      ? buildScalarEpilogue(KBody) : nullptr;

  // Find the K IV PHINode for EmittedMap registration in execute().
  PHINode *KIVPhi = nullptr;
  for (TPRecipeBase &R : *KBody)
    if (auto *IV = dyn_cast<TPWidenIntOrFpInductionRecipe>(&R))
      if (IV->getDimIndex() == Spec.Dim)
        { KIVPhi = IV->getIVPhi(); break; }
  assert(KIVPhi && "TPlanTransformer: no K IV recipe found");

  // Create the TPTilingRegion.
  auto *TR = new TPTilingRegion(Spec.Dim, Spec.PF, Spec.Mode,
                                 KBody, Epilogue, KIVPhi);

  // Replace Innermost in its parent's child slot.
  // (TPRegionBlock::replaceInner() or equivalent — see Task note below.)
  Innermost->getParentRegion()->setInner(TR);

  return TR;
}

void TPlanTransformer::insertGuardBlock(const DimEmissionSpec &Spec,
                                         Value *TCVal) {
  assert(OutermostLoop && "TPlanTransformer: no outermost loop for guard");
  TPBlockBase *Root = Plan.getEntry();
  auto *Guard = new TPGuardBlock(OutermostLoop, TCVal, Spec.PF, Root);
  Plan.setEntry(Guard);
}
```

> **Implementation note:** `TPRegionBlock::getBodyBlock()`, `getParentRegion()`, `setInner()`, and `TPlan::setEntry()` may not exist yet. Check `TPlan.h:447` and `TPlan.h` for the TPlan entry field. Add the needed accessors. `getBodyBlock()` should return the `ir-bb<k.loop>` `TPIRBasicBlock` — look at how `buildInitial()` in `TPlan.cpp` constructs the region to understand its layout.

- [ ] **Step 5: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -10
```

Fix any compilation errors (missing accessors, etc.). Do NOT run tests yet — the Transformer isn't wired into the lowering pass.

- [ ] **Step 6: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: add TPlanPolicyAnalysis and TPlanTransformer"
```

---

## Task 7: Update `emitContraction()` to use `EmittedMap` for tile pointers

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

When `TPTilingRegion::execute()` runs Body's WIDEN-GEP recipes, they store tile-corner GEPs in `EmittedMap`. `emitContraction()` must use these instead of re-computing pointers from scratch via `decomposePtrForDims()`.

- [ ] **Step 1: Locate pointer acquisition in `emitContraction()`**

Find lines ~597–617 in `TPlanLowering.cpp`:
```cpp
PtrDecomposition ADecomp = decomposePtrForDims(
    cast<LoadInst>(LHSVal)->getPointerOperand(), ...);
Value *LHSPtr = ADecomp.Base;
...
PtrDecomposition BDecomp = decomposePtrForDims(
    cast<LoadInst>(RHSVal)->getPointerOperand(), ...);
Value *RHSPtr = BDecomp.Base;
```

- [ ] **Step 2: Add EmittedMap lookup before decomposePtrForDims()**

```cpp
// Try to find tile-corner pointer from WIDEN-GEP recipe via EmittedMap.
// When TPTilingRegion::execute() ran Body->execute(), WIDEN-GEP recipes
// emitted tile-adjusted GEPs and stored them in EmittedMap.
auto lookupEmittedPtr = [&](Value *OrigGEP) -> Value * {
  auto It = State.EmittedMap.find(OrigGEP);
  return It != State.EmittedMap.end() ? It->second : nullptr;
};

Value *LHSGEPOrig = cast<LoadInst>(LHSVal)->getPointerOperand();
Value *LHSPtr = lookupEmittedPtr(LHSGEPOrig);
if (!LHSPtr) {
  PtrDecomposition ADecomp = decomposePtrForDims(LHSGEPOrig, ...);
  LHSPtr = ADecomp.Base;
}
if (!LHSPtr) return nullptr;

Value *RHSGEPOrig = cast<LoadInst>(RHSVal)->getPointerOperand();
Value *RHSPtr = lookupEmittedPtr(RHSGEPOrig);
if (!RHSPtr) {
  PtrDecomposition BDecomp = decomposePtrForDims(RHSGEPOrig, ...);
  RHSPtr = BDecomp.Base;
}
if (!RHSPtr) return nullptr;
```

Strides still come from `ADecomp`/`BDecomp` (unchanged) — only the base pointer is overridden.

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build. Tests still pass (EmittedMap lookup returns nullptr in non-tiling cases, so fallback to existing path).

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/ 2>&1 | tail -5
```

Expected: 44 PASS + 1 XFAIL.

- [ ] **Step 4: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
git commit -m "tplan-lower: use EmittedMap for tile-corner pointer in emitContraction()"
```

---

## Task 8: Wire `TPlanTransformer` into `TPlanLowering_lower()`; remove old IR surgery

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`

This is the wiring task. The old IR surgery (`createTensorizedLoopSkeleton()` call and `preBuildTilingBlocks()` call) is removed and replaced with `TPlanTransformer::transform()`.

- [ ] **Step 1: Add Transformer call + remove old IR surgery in `TPlanLowering_lower()`**

Find the block in `TPlanLowering_lower()` (around line 2096–2152) that contains:
```cpp
if (State.Policy.needsGuard()) {
  ...
  createTensorizedLoopSkeleton(...);    // ← REMOVE
  ...
}
if (State.Policy.needsGuard()) {
  preBuildTilingBlocks(...);            // ← REMOVE
}
```

Replace with:

```cpp
  // Transform TPlan structure (no IR mutation here).
  // TPlanTransformer inserts TPGuardBlock + TPTilingRegion nodes,
  // sets State.TilingTCVal, and marks IsSubsumed on absorbed recipes.
  TPlanTransformer Transformer(Plan, State.Policy, SE, LI, *State.Expander,
                                Builder, CM, State.DimToLoop);
  Transformer.transform(State);
```

- [ ] **Step 2: Remove `PrebuiltTilingPtr` from TPTransformState**

In `TPlan.h`, delete:
```cpp
  void *PrebuiltTilingPtr = nullptr;
```

Fix any remaining references (grep for `PrebuiltTilingPtr` and remove).

- [ ] **Step 3: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -10
```

Fix compilation errors.

- [ ] **Step 4: Run regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: 44 PASS + 1 XFAIL. If tests fail, compare actual vs expected IR with:
```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/skeleton-guard.ll 2>&1
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tiling-dynamic-k.ll 2>&1
```

- [ ] **Step 5: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/include/llvm/Transforms/Vectorize/TPlan.h
git commit -m "tplan-lower: wire TPlanTransformer; remove createTensorizedLoopSkeleton + preBuildTilingBlocks calls"
```

---

## Task 9: Remove dead code + final regression

**Files:**
- Modify: `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- Modify/Delete: `llvm/lib/Transforms/Vectorize/TPlanSkeleton.cpp`, `TPlanSkeleton.h`

- [ ] **Step 1: Remove `preBuildTilingBlocks()` function**

Delete the entire `preBuildTilingBlocks()` function body (~120 lines, starting around line 1906). Also remove its forward declaration and `PrebuiltTilingInfo` struct.

- [ ] **Step 2: Remove tiling loop creation code from `emitContraction()`**

`emitContraction()` no longer needs to create tiling loops — that's `TPTilingRegion::execute()`'s job. Remove the following paths from `emitContraction()`:
- Static tiling path (lines ~1060–1157): the `if (NeedsStaticTiling && !NeedsDynamicTiling)` block
- Dynamic tiling path (lines ~1159–1360): the `if (NeedsDynamicTiling)` block
- Pre-built path (lines ~898–1057): already replaced, remove

Keep only:
- No-tiling path (direct `tensor.contract` call, lines ~811–840)
- Stride analysis and pointer setup (lines ~597–720)

- [ ] **Step 3: Remove `TPlanSkeleton` if fully absorbed**

If `TPGuardBlock::execute()` now covers all cases previously handled by `createTensorizedLoopSkeleton()`, remove:
- `llvm/lib/Transforms/Vectorize/TPlanSkeleton.cpp`
- `llvm/include/llvm/Transforms/Vectorize/TPlanSkeleton.h`
- Update `CMakeLists.txt` to remove `TPlanSkeleton.cpp`
- Remove `#include "TPlanSkeleton.h"` from `TPlanLowering.cpp`

Check: `grep -r "TPlanSkeleton\|createTensorizedLoopSkeleton" llvm/` should return no hits.

- [ ] **Step 4: Remove `State.Policy` pre-built path check from `emitContraction()`**

`emitContraction()` no longer checks `State.PrebuiltTilingPtr` (removed in Task 8). Verify no references remain:

```bash
grep -n "PrebuiltTiling\|preBuildTiling\|TPlanSkeleton\|createTensorizedLoop" \
  llvm/lib/Transforms/Vectorize/TPlanLowering.cpp
```

Expected: no output.

- [ ] **Step 5: Build**

```bash
ninja -C build LLVMVectorize 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 6: Final regression suite**

```bash
./build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/
```

Expected: 44 PASS + 1 XFAIL. Zero regressions.

- [ ] **Step 7: Verify no pre-execute IR surgery in `TPlanLowering_lower()`**

```bash
grep -n "eraseFromParent\|setSuccessor\|getTerminator\|cloneLoop\|BasicBlock::Create" \
  llvm/lib/Transforms/Vectorize/TPlanLowering.cpp | grep -v "emitContraction\|TPTiling\|TPGuard"
```

Expected: no hits outside of recipe execute() methods and the two new node implementations.

- [ ] **Step 8: Commit**

```bash
git add llvm/lib/Transforms/Vectorize/TPlanLowering.cpp \
        llvm/lib/Transforms/Vectorize/TPlanSkeleton.cpp \
        llvm/include/llvm/Transforms/Vectorize/TPlanSkeleton.h \
        llvm/lib/Transforms/Vectorize/CMakeLists.txt
git commit -m "tplan-lower: remove preBuildTilingBlocks, tiling-loop code in emitContraction, TPlanSkeleton"
```

---

## Notes for Implementer

- **`TPRegionBlock::getBodyBlock()`**: look at `TPRegionBlock` fields in `TPlan.h:447` and how `buildInitial()` in `TPlan.cpp` populates the region to find what holds the `ir-bb<k.loop>` block. You may need to add accessor methods.
- **`TPRecipeBase::clone()`**: if this doesn't exist, implement `buildScalarEpilogue()` by re-using the same recipe objects with `setSubsumed(false)` instead of cloning (if recipes are immutable, cloning is required).
- **`TPlan::setEntry()` / `getEntry()`**: check current TPlan entry-point API. `Plan.getEntry()` already exists (`TPlan.cpp`); `setEntry()` may need to be added.
- **Task 6 accessors**: when you can't find `getParentRegion()` or `setInner()` on `TPRegionBlock`, add them — they're small one-liners.
- **Order matters in Tasks 7 vs 8**: Task 7 (EmittedMap in emitContraction) must land before Task 8 (wiring), so the tensor-path pointer lookup works when the new path runs.

---

## Spec Reference

`docs/superpowers/specs/2026-04-13-tplan-lowering-refactor-design.md`
