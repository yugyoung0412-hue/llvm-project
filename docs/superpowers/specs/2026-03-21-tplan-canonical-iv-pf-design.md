# TPlan: Canonical IV Recipe + PF Synthetic Live-in

**Date:** 2026-03-21
**Branch:** LoopTensorizebyClaude
**Files affected:** `llvm/include/llvm/Transforms/Vectorize/TPlan.h`, `llvm/lib/Transforms/Vectorize/TPlan.cpp`

---

## Motivation

The current TPlan print output lacks two features that VPlan provides:

1. **Synthetic live-in values** — VPlan surfaces `VF`, `UF`, `vector-trip-count` etc. as named
   synthetic values (`vp<N>`) separate from IR-backed live-ins (`ir<name>`). TPlan has no equivalent.
2. **Canonical induction variable** — VPlan inserts a `CANONICAL-INDUCTION` recipe as the primary
   loop counter (separate from the original IR induction). TPlan only has `WIDEN-INDUCTION` which
   mirrors the IR PHI directly.

This spec defines how to add both features to TPlan, following VPlan's conventions.

---

## Target Output (illustrative — actual slot numbers depend on traversal order)

```
TPlan '_ZL32ggml_compute_forward_mul_mat_f32PK11ggml_tensorS1_PS_' (depth=5) {
Live-in tp<%0> = PF
Live-in ir<0>
Live-in ir<%7>
...

<x1> loop[0] (trip=(-1 + %17)) {
  CANONICAL-INDUCTION tp<%1> = phi ir<0>, tp<%2>
  WIDEN-INDUCTION tp<%3> = phi ir<0>, tp<%4>
  ...body recipes...
  CANONICAL-INDUCTION-INC tp<%2> = add tp<%1>, tp<%0>
  CANONICAL-INDUCTION-CMP tp<%X> = icmp tp<%2>, ir<%17>
  <x1> loop[1] (trip=(-1 + %15)) {
    CANONICAL-INDUCTION tp<%A> = phi ir<0>, tp<%B>
    ...
    CANONICAL-INDUCTION-INC tp<%B> = add tp<%A>, tp<%0>
    CANONICAL-INDUCTION-CMP tp<%C> = icmp tp<%B>, ir<%15>
  }
}
}
```

Note: `tp<%2>` is the step operand of `CANONICAL-INDUCTION`. Its slot is assigned by **first use**
(when the phi's backedge operand is printed), not by definition order. This is intentional and
mirrors the existing lazy slot-assignment behavior of `TPSlotTracker` for all `TPDefVal`s.

---

## Design

### 1. PF Synthetic Live-in

**What:** `PF` (Parallel Factor) is TPlan's analogue of VPlan's `VF`. It represents the
parallelism width for tensorization — the tile/block size along the computation dimension.
It is a synthetic value with no underlying IR value.

**Why a new class:** `TPLiveIn` requires an IR `Value*` backing. `TPDefVal` requires a defining
recipe. Neither fits PF, which has no IR backing and no recipe. A new `TPSyntheticValue` class
derives directly from `TPValue` and stores a pre-assigned slot number instead.

**Where:** `TPlan` struct gains a `TPSyntheticValue PF` member (one instance, like VPlan's `VF`
field). `TPSyntheticValue::printAsOperand` emits `tp<%N>` using its pre-assigned slot.

**Slot assignment:** `TPSlotTracker::SlotMap` is widened from
`DenseMap<const TPDefVal *, unsigned>` to `DenseMap<const TPValue *, unsigned>`, unifying
synthetic and recipe-defined values under one map. The existing `getSlot(const TPDefVal *)` is
updated to accept `const TPValue *`. A new method is added:
```cpp
void preAssignSynthetic(const TPSyntheticValue *V);
```
This method inserts `V` (as a `const TPValue *` key) into the shared `SlotMap` at `NextSlot`
(assigning it slot 0 when called first) and increments `NextSlot`. This ensures PF always gets
`tp<%0>`, before any recipe-defined value can claim slot 0 via lazy assignment.

**Sequencing in `TPlan::print()`:**
```
1. Tracker.reset()               ← clear all previous slots
2. Tracker.preAssignSynthetic(&PF)  ← PF gets slot 0, NextSlot = 1
3. Print "Live-in tp<%0> = PF"
4. Print IR-backed live-ins (unchanged, still ir<>)
5. Print regions (recipe TPDefVals get slots 1, 2, 3, ... lazily)
```

**Print (in recipes):** PF is referenced as `tp<%0>` in the canonical IV increment recipes.

**IR-backed live-ins** remain unchanged — printed as `ir<>` in both header and recipe operands.

---

### 2. TPCanonicalIVRecipe — `CANONICAL-INDUCTION`

**What:** A new recipe class inserted as the **first recipe** in every `TPLoopRegion`. It
represents a synthetic sequential loop counter — not the original IR induction variable. This
mirrors `VPCanonicalIVPHIRecipe` in VPlan.

**Class:** `TPCanonicalIVRecipe : public TPRecipeBase`
- `RecipeKind::CanonicalIV` added to the enum
- Operand[0]: start value (`ir<0>` as a `TPLiveIn`)
- Operand[1]: step value (the `TPDefVal` of the companion `CANONICAL-INDUCTION-INC` recipe)
- Defines one `TPDefVal` (the phi result)
- Prints as: `CANONICAL-INDUCTION tp<%N> = phi ir<0>, tp<%step>`

**Forward reference and slot assignment:** `CANONICAL-INDUCTION` prints before its companion
`CANONICAL-INDUCTION-INC`. When `printAsOperand` is called on operand[1] (the step `TPDefVal`),
the slot tracker assigns it a slot at that moment (first use = definition of slot). The companion
`CANONICAL-INDUCTION-INC` then prints its own `DefVal` using the already-assigned slot. This is
intentional: slot numbers are assigned by first-use order, not definition order, which is how
`TPSlotTracker` already works for all `TPDefVal`s.

**Step operand resolution (no IR backing, no ValueMap lookup):**
Unlike `WIDEN-INDUCTION` (which patches its step via `ValueMap` using an IR `Value*` key),
`CANONICAL-INDUCTION`'s companion add recipe is synthetic — it has no IR instruction. The step
operand is resolved by storing a **direct pointer** to the companion add's `TPDefVal` in
`TPCanonicalIVRecipe` at construction time. Specifically:

```
1. Create TPCanonicalIVRecipe with operand[1] = placeholder (nullptr or a dummy)
2. Build all body recipes (normal WIDEN-INDUCTION, WIDEN, etc.)
3. Create TPCanonicalIVIncrRecipe → get its TPDefVal*
4. Call canonicalIV.setOperand(1, incrRecipe->getDefinedValue())
           + incrRecipe->getDefinedValue()->addUser(&canonicalIV)
5. Create TPCanonicalIVExitCmpRecipe
```

No post-pass lookup is needed. This is simpler and more direct than the WIDEN-INDUCTION mechanism.

---

### 3. Companion Recipes — `CANONICAL-INDUCTION-INC` and `CANONICAL-INDUCTION-CMP`

Both companion recipes are **synthetic** — they have no backing IR `Instruction*`. They cannot
use `TPWidenRecipe` (which requires an `Instruction*` for `getOpcodeName()`). Two dedicated
subclasses are introduced:

**`TPCanonicalIVIncrRecipe : public TPRecipeBase`**
- `RecipeKind::CanonicalIVIncr`
- Operand[0]: canonical IV phi result (the `TPDefVal` of `TPCanonicalIVRecipe`)
- Operand[1]: PF (`TPSyntheticValue*`)
- Defines one `TPDefVal` (the incremented value)
- Prints as: `CANONICAL-INDUCTION-INC tp<%M> = add tp<%N>, tp<%0>`

**`TPCanonicalIVExitCmpRecipe : public TPRecipeBase`**
- `RecipeKind::CanonicalIVExitCmp`
- Operand[0]: incremented canonical IV (`TPDefVal` of `TPCanonicalIVIncrRecipe`)
- Operand[1]: loop trip count bound (a `TPLiveIn` wrapping the IR bound value)
- Defines one `TPDefVal`
- Prints as: `CANONICAL-INDUCTION-CMP tp<%C> = icmp tp<%M>, ir<%bound>`

**Trip count bound source:** The bound IR `Value*` is read from the loop latch block's
conditional branch. Specifically: `L->getLoopLatch()` → `BranchInst` → condition `ICmpInst` →
`getRHS()`. This is the same IR value that the existing `WIDEN icmp` recipe already captures
(e.g., `ir<%17>` in `WIDEN tp<%9> = icmp tp<%1>, ir<%17>`). It is wrapped as a `TPLiveIn`
via the existing `getOrCreateLiveIn()` mechanism, so no new IR value lookup is required.

**Insertion order** in each `BuildRegion()` call:
```
1. Insert TPCanonicalIVRecipe (step = placeholder)
2. Build body recipes (WIDEN-INDUCTION, WIDEN, WIDEN-GEP, etc.)
3. Recurse into child region (if any)
4. Create TPCanonicalIVIncrRecipe → patch canonical IV step operand
5. Create TPCanonicalIVExitCmpRecipe
6. Append both companion recipes to region's recipe list
```

---

### 4. middle.block / scalar.ph / exit.block — Not Needed

These blocks exist in VPlan to handle the vector→scalar epilogue transition. TPlan does not
need them at the initial plan stage because:

- TPlan is an **analysis IR** — it models the loop nest structure, it does not drive codegen yet.
- Tensorization does not produce scalar epilogues; remainder tile handling (if needed) would be
  a separate future concern modeled as a "remainder region", not a scalar loop.
- Initial TPlan is complete without these.

---

## Class Hierarchy Summary

```
TPValue
├── TPLiveIn          — wraps IR Value*, prints ir<>
├── TPSyntheticValue  — no IR backing, pre-assigned slot, prints tp<%N>   [NEW]
└── TPDefVal          — defined by a recipe, lazy slot, prints tp<%N>

TPRecipeBase
├── TPWidenInductionRecipe       — WIDEN-INDUCTION
├── TPReductionPHIRecipe         — WIDEN-REDUCTION-PHI
├── TPWidenRecipe                — WIDEN (wraps IR Instruction*)
├── TPWidenGEPRecipe             — WIDEN-GEP (wraps IR Instruction*)
├── TPWidenLoadRecipe            — WIDEN load (wraps IR Instruction*)
├── TPWidenStoreRecipe           — WIDEN store (wraps IR Instruction*)
├── TPWidenCastRecipe            — WIDEN-CAST (wraps IR Instruction*)
├── TPCanonicalIVRecipe          — CANONICAL-INDUCTION             [NEW]
├── TPCanonicalIVIncrRecipe      — CANONICAL-INDUCTION-INC         [NEW]
└── TPCanonicalIVExitCmpRecipe   — CANONICAL-INDUCTION-CMP         [NEW]
```

---

## Verification

```bash
# Rebuild opt
ninja -C build opt

# Run on ggml test input — check structure
build/bin/opt -passes=loop-tensorize -debug-only=loop-tensorize \
  ggml_compute_forward_mul_mat.ll -o /dev/null 2>&1

# Expected: header has "Live-in tp<%0> = PF"
# Expected: each loop region starts with CANONICAL-INDUCTION
# Expected: each loop region ends with CANONICAL-INDUCTION-INC + CANONICAL-INDUCTION-CMP
# Expected: IR live-ins still appear as ir<> in all body recipes

# Run lit test (update expected output to match new format)
build/bin/llvm-lit -v llvm/test/Transforms/LoopTensorize/basic/tplan-print.ll
```
