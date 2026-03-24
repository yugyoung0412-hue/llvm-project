# TPlan PHI Recipes Design Spec

## Goal

Add VPlan-mirrored PHI recipe hierarchy to TPlan: introduce `TPHeaderPHIRecipe` abstract base, split `TPWidenInductionRecipe` into two concrete subclasses, rebase existing header PHI recipes, and add six new stub recipes covering the full VPlan PHI surface.

## Background

VPlan defines 13 PHI-related classes. TPlan currently has three: `TPCanonicalIVRecipe`, `TPWidenInductionRecipe`, and `TPReductionPHIRecipe`. These are not grouped under an abstract base and `TPWidenInductionRecipe` is monolithic. This spec brings TPlan's PHI hierarchy in line with VPlan's.

---

## Approach: Two Commits

**Commit 1 — Hierarchy restructure + rebase + split**
- Add `TPHeaderPHIRecipe` abstract base (mirrors `VPHeaderPHIRecipe`)
- Make `TPWidenInductionRecipe` abstract; hoist `IVPhi`, `DimIndex`, `getIVPhi()`, `getDimIndex()` into it
- Add `TPWidenIntOrFpInductionRecipe` and `TPWidenPointerInductionRecipe`
- Rebase `TPCanonicalIVRecipe` and `TPReductionPHIRecipe` onto `TPHeaderPHIRecipe`
- Update enum: replace `TPWidenInductionSC` with the two new SCs (inserted in place, before `TPReductionPHISC`)
- Update `TPSingleDefRecipe::classof`: replace `case TPWidenInductionSC` with the two new cases
- Update `TPlan.cpp buildInitial()` to branch on `Phi->getType()->isPointerTy()`
- Split `TPWidenInductionRecipe::print()` in `TPlan.cpp` into both concrete subclasses
- Update lit test CHECK lines

**Commit 2 — Six new recipe stubs**
- `TPWidenPHIRecipe` (flip existing `case TPWidenPHISC: return false` to `return true`)
- `TPFirstOrderRecurrencePHIRecipe`, `TPActiveLaneMaskPHIRecipe`, `TPEVLBasedIVPHIRecipe`
- `TPPredInstPHIRecipe`, `TPPhi` (generic PHI)
- Insert new SCs after `TPReductionPHISC` and before `TPCanonicalIVIncrSC`
- Update `TPLastHeaderPHISC = TPEVLBasedIVPHISC`, `TPLastPHISC = TPPhiSC`
- Update `TPSingleDefRecipe::classof`: add cases for all six new SCs

---

## Class Hierarchy

```
TPSingleDefRecipe
├── TPHeaderPHIRecipe  [abstract, NEW]             [header PHI range]
│   ├── TPCanonicalIVRecipe          (rebased)     [TPCanonicalIVSC]
│   ├── TPWidenInductionRecipe       (now abstract)
│   │   ├── TPWidenIntOrFpInductionRecipe  [NEW]   [TPWidenIntOrFpInductionSC]
│   │   └── TPWidenPointerInductionRecipe  [NEW]   [TPWidenPointerInductionSC]
│   ├── TPReductionPHIRecipe         (rebased)     [TPReductionPHISC]
│   ├── TPFirstOrderRecurrencePHIRecipe  [NEW]     [TPFirstOrderRecurrencePHISC]
│   ├── TPActiveLaneMaskPHIRecipe    [NEW]         [TPActiveLaneMaskPHISC]
│   └── TPEVLBasedIVPHIRecipe        [NEW]         [TPEVLBasedIVPHISC]
├── TPWidenPHIRecipe   [NEW]                       [TPWidenPHISC]
├── TPPredInstPHIRecipe  [NEW]                     [TPPredInstPHISC]
└── TPPhi              [NEW, generic]              [TPPhiSC]
```

`TPPhiAccessors` mixin applied to: `TPHeaderPHIRecipe`, `TPWidenPHIRecipe`, `TPPhi`.
`TPPredInstPHIRecipe` omits `TPPhiAccessors` — mirroring `VPPredInstPHIRecipe`, which holds a single value (the predicated instruction result), not an incoming-value list.

---

## Enum Layout

The ordering follows the existing pattern exactly. New SCs are inserted within the PHI block, **before** `TPCanonicalIVIncrSC`/`TPCanonicalIVExitCmpSC`, which must always remain outside the PHI range. `TPFirstPHISC = TPWidenPHISC` is an alias (unchanged).

**After Commit 1:**

```cpp
enum TPRecipeTy {
  // Non-PHI recipes (unchanged)
  TPWidenSC,
  TPWidenGEPSC,
  TPWidenLoadSC,
  TPWidenStoreSC,
  TPWidenCastSC,

  // PHI-like recipes
  TPWidenPHISC,       // placeholder; gains concrete class in Commit 2

  // Header PHI recipes
  TPCanonicalIVSC,
  TPWidenIntOrFpInductionSC,  // NEW: replaces TPWidenInductionSC
  TPWidenPointerInductionSC,  // NEW
  TPReductionPHISC,

  // Canonical IV companion recipes (outside PHI range — unchanged position)
  TPCanonicalIVIncrSC,
  TPCanonicalIVExitCmpSC,

  // Range markers (aliases)
  TPFirstPHISC       = TPWidenPHISC,
  TPFirstHeaderPHISC = TPCanonicalIVSC,
  TPLastHeaderPHISC  = TPReductionPHISC,  // unchanged after Commit 1
  TPLastPHISC        = TPReductionPHISC,  // unchanged after Commit 1
};
```

**After Commit 2** (adds entries before `TPCanonicalIVIncrSC`):

```cpp
  // Header PHI recipes (extended)
  TPCanonicalIVSC,
  TPWidenIntOrFpInductionSC,
  TPWidenPointerInductionSC,
  TPReductionPHISC,
  TPFirstOrderRecurrencePHISC,  // NEW
  TPActiveLaneMaskPHISC,        // NEW
  TPEVLBasedIVPHISC,            // NEW

  // Non-header PHIs (NEW)
  TPPredInstPHISC,
  TPPhiSC,

  // Canonical IV companions (moved to higher numeric values — still outside PHI range)
  TPCanonicalIVIncrSC,
  TPCanonicalIVExitCmpSC,

  // Range markers
  TPLastHeaderPHISC  = TPEVLBasedIVPHISC,  // updated
  TPLastPHISC        = TPPhiSC,            // updated
```

Moving `TPCanonicalIVIncrSC`/`TPCanonicalIVExitCmpSC` to higher numeric values is safe — these are in-memory C++ enum values used only for runtime type dispatch, not serialized.

`TPWidenInductionSC` is removed in Commit 1.

---

## Key Class Definitions

### `TPHeaderPHIRecipe` (abstract, NEW)

```cpp
class TPHeaderPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
protected:
  // Operands vary by subclass: e.g. {StartVal} or {StartVal, StepVal}
  TPHeaderPHIRecipe(TPRecipeTy ID, ArrayRef<TPValue *> Operands)
      : TPSingleDefRecipe(ID, Operands) {}
  const TPRecipeBase *getAsRecipe() const override { return this; }
public:
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() >= TPFirstHeaderPHISC &&
           R->getTPRecipeID() <= TPLastHeaderPHISC;
  }
  TPValue *getStartValue() const { return getOperand(0); }
  void setStartValue(TPValue *V) { setOperand(0, V); }
};
```

No `Instruction *I` / UV parameter — existing header PHI classes do not set a `UnderlyingValue`.

### `TPCanonicalIVRecipe` (rebased)

Change base class from `TPSingleDefRecipe` to `TPHeaderPHIRecipe`. Preserve the existing two-operand constructor signature:

```cpp
TPCanonicalIVRecipe(TPValue *StartVal, TPValue *StepVal)
    : TPHeaderPHIRecipe(TPCanonicalIVSC, {StartVal, StepVal}) {}
```

`classof` unchanged: `return R->getTPRecipeID() == TPCanonicalIVSC`.

### `TPReductionPHIRecipe` (rebased)

Change base class from `TPSingleDefRecipe` to `TPHeaderPHIRecipe`. Preserve the existing constructor and `RedPhi`/`getPhi()` fields unchanged. The inherited `getStartValue()` returns `getOperand(0)` = the init value, which is semantically correct. `getPhi()` is kept as-is.

```cpp
// Constructor: pass operands to TPHeaderPHIRecipe; keep RedPhi field.
TPReductionPHIRecipe(PHINode *Phi, TPValue *StartVal, ...)
    : TPHeaderPHIRecipe(TPReductionPHISC, {StartVal, ...}), RedPhi(Phi) {}
```

`classof` unchanged: `return R->getTPRecipeID() == TPReductionPHISC`.

### `TPWidenInductionRecipe` (now abstract)

Holds `IVPhi`, `DimIndex`, `getIVPhi()`, `getDimIndex()` — hoisted from the current monolithic class. Does **not** set a `UnderlyingValue` (UV remains nullptr, matching current behavior).

```cpp
class TPWidenInductionRecipe : public TPHeaderPHIRecipe {
protected:
  PHINode *IVPhi;
  unsigned DimIndex;
  TPWidenInductionRecipe(TPRecipeTy ID, PHINode *Phi,
                         TPValue *Start, TPValue *Step, unsigned Dim)
      : TPHeaderPHIRecipe(ID, {Start, Step}),
        IVPhi(Phi), DimIndex(Dim) {}
public:
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenIntOrFpInductionSC ||
           R->getTPRecipeID() == TPWidenPointerInductionSC;
  }
  PHINode *getIVPhi()    const { return IVPhi; }
  unsigned getDimIndex() const { return DimIndex; }
};
```

### Concrete induction classes

Both delegate to `TPWidenInductionRecipe`. `execute()` body: `State.setValue(this, IVPhi)`. Use the correct `print()` signature matching the existing codebase: `void print(raw_ostream &OS, unsigned Indent, TPSlotTracker &Tracker) const override`.

```cpp
class TPWidenIntOrFpInductionRecipe : public TPWidenInductionRecipe {
public:
  TPWidenIntOrFpInductionRecipe(PHINode *Phi, TPValue *Start,
                                 TPValue *Step, unsigned Dim)
      : TPWidenInductionRecipe(TPWidenIntOrFpInductionSC, Phi,
                               Start, Step, Dim) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenIntOrFpInductionSC;
  }
  void execute(TPTransformState &State) const override;
  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
};

class TPWidenPointerInductionRecipe : public TPWidenInductionRecipe {
public:
  TPWidenPointerInductionRecipe(PHINode *Phi, TPValue *Start,
                                 TPValue *Step, unsigned Dim)
      : TPWidenInductionRecipe(TPWidenPointerInductionSC, Phi,
                               Start, Step, Dim) {}
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenPointerInductionSC;
  }
  void execute(TPTransformState &State) const override;
  void print(raw_ostream &OS, unsigned Indent,
             TPSlotTracker &Tracker) const override;
};
```

### Six new stubs (Commit 2)

All have no-op `execute()` bodies and minimal `print(raw_ostream &, unsigned, TPSlotTracker &)` overrides. They exist to complete the enum and enable `isa<>`/`classof` dispatch.

---

## `TPSingleDefRecipe::classof` Updates

**Commit 1**: Remove `case TPWidenInductionSC`. Add:
```cpp
case TPWidenIntOrFpInductionSC:
case TPWidenPointerInductionSC:
  return true;
```

**Commit 2**: Change the **existing** `case TPWidenPHISC: return false` (placeholder) to fall through to `return true`. Do NOT add a second `case TPWidenPHISC`. Then add:
```cpp
case TPFirstOrderRecurrencePHISC:
case TPActiveLaneMaskPHISC:
case TPEVLBasedIVPHISC:
case TPPredInstPHISC:
case TPPhiSC:
  return true;
```

---

## Consumer Updates

| File | Change |
|------|--------|
| `TPlan.h` `TPSingleDefRecipe::classof` | Per commit schedule above |
| `TPlan.cpp` `buildInitial()` | Branch on `Phi->getType()->isPointerTy()` to select `TPWidenPointerInductionRecipe` vs `TPWidenIntOrFpInductionRecipe`; pass `StepVal` and `DimIndex` |
| `TPlan.cpp` `TPWidenInductionRecipe::print()` | Split into `TPWidenIntOrFpInductionRecipe::print()` and `TPWidenPointerInductionRecipe::print()` (same body for now) |
| `TPlan.cpp` `printAsOperand` (~line 242) | Cast `dyn_cast<TPWidenInductionRecipe>` remains valid (abstract base `classof` covers both new SCs); **no change** to `printAsOperand` itself |
| `TPlanLowering.cpp` `TPWidenInductionRecipe::execute()` | Split into both concrete subclasses (same body: `State.setValue(this, IVPhi)`) |
| `TPlanWidener.cpp` | `dyn_cast<TPWidenInductionRecipe>`, `getDimIndex()`, `getIVPhi()` remain valid via abstract base; no SC-level change needed |
| `TPRecipeMatcher.cpp` | No change — does not dispatch on induction SC |
| Lit test (`tplan-build-print.ll`) | Update `CHECK` lines: `TPWidenInductionRecipe` → `TPWidenIntOrFpInductionRecipe` or `TPWidenPointerInductionRecipe` depending on IV type |

---

## Testing

- Existing lit test `tplan-build-print.ll` updated in Commit 1 to match renamed recipe in output.
- No new lit tests required for stubs (Commit 2) — stubs have no output-visible behavior yet.
- Build must pass with no new errors or warnings (beyond the pre-existing domination failure in lit tests).
