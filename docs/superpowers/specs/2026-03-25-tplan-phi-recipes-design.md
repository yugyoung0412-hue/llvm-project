# TPlan PHI Recipes Design Spec

## Goal

Add VPlan-mirrored PHI recipe hierarchy to TPlan: introduce `TPHeaderPHIRecipe` abstract base, split `TPWidenInductionRecipe` into two concrete subclasses, rebase existing header PHI recipes, and add six new stub recipes covering the full VPlan PHI surface.

## Background

VPlan defines 13 PHI-related classes. TPlan currently has three: `TPCanonicalIVRecipe`, `TPWidenInductionRecipe`, and `TPReductionPHIRecipe`. These are not grouped under an abstract base and `TPWidenInductionRecipe` is monolithic. This spec brings TPlan's PHI hierarchy in line with VPlan's.

---

## Approach: Two Commits

**Commit 1 — Hierarchy restructure + rebase + split**
- Add `TPHeaderPHIRecipe` abstract base (mirrors `VPHeaderPHIRecipe`)
- Make `TPWidenInductionRecipe` abstract (mirrors `VPWidenInductionRecipe`)
- Add `TPWidenIntOrFpInductionRecipe` and `TPWidenPointerInductionRecipe`
- Rebase `TPCanonicalIVRecipe` and `TPReductionPHIRecipe` onto `TPHeaderPHIRecipe`
- Update `buildInitial()` to select the correct concrete induction class
- Update lit test CHECK lines

**Commit 2 — Six new recipe stubs**
- `TPWidenPHIRecipe` (fills existing placeholder SC)
- `TPFirstOrderRecurrencePHIRecipe`
- `TPActiveLaneMaskPHIRecipe`
- `TPEVLBasedIVPHIRecipe`
- `TPPredInstPHIRecipe`
- `TPPhi` (generic PHI)
- Update `TPRecipeTy` enum range markers

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

---

## Enum Layout

```cpp
enum TPRecipeTy {
  // PHI recipes
  TPFirstPHISC,

  // Header PHIs
  TPFirstHeaderPHISC,
  TPCanonicalIVSC = TPFirstHeaderPHISC,
  TPWidenIntOrFpInductionSC,   // NEW (replaces TPWidenInductionSC)
  TPWidenPointerInductionSC,   // NEW
  TPReductionPHISC,
  TPFirstOrderRecurrencePHISC, // NEW
  TPActiveLaneMaskPHISC,       // NEW
  TPEVLBasedIVPHISC,           // NEW
  TPLastHeaderPHISC = TPEVLBasedIVPHISC,

  // Non-header PHIs
  TPWidenPHISC,                // NEW
  TPPredInstPHISC,             // NEW
  TPPhiSC,                     // NEW
  TPLastPHISC = TPPhiSC,

  // Non-PHI recipes
  TPCanonicalIVIncrSC,
  TPCanonicalIVExitCmpSC,
  TPWidenSC,
  TPWidenGEPSC,
  TPWidenLoadSC,
  TPWidenStoreSC,
  TPWidenCastSC,
};
```

`TPWidenInductionSC` is removed; the abstract `TPWidenInductionRecipe` uses no SC.

---

## Key Class Definitions

### `TPHeaderPHIRecipe` (abstract)

```cpp
class TPHeaderPHIRecipe : public TPSingleDefRecipe, public TPPhiAccessors {
protected:
  TPHeaderPHIRecipe(TPRecipeTy ID, Instruction *I, TPValue *StartVal)
      : TPSingleDefRecipe(ID, {StartVal}, I) {}
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

### `TPWidenInductionRecipe` (abstract)

```cpp
class TPWidenInductionRecipe : public TPHeaderPHIRecipe {
protected:
  PHINode *IVPhi;
  TPWidenInductionRecipe(TPRecipeTy ID, PHINode *Phi, TPValue *Start)
      : TPHeaderPHIRecipe(ID, Phi, Start), IVPhi(Phi) {}
public:
  static bool classof(const TPRecipeBase *R) {
    return R->getTPRecipeID() == TPWidenIntOrFpInductionSC ||
           R->getTPRecipeID() == TPWidenPointerInductionSC;
  }
  PHINode *getPHINode() const { return IVPhi; }
};
```

### Concrete induction classes

Both carry the same `IVPhi` field from `TPWidenInductionRecipe`. `execute()` on both does `State.setValue(this, IVPhi)`. `print()` emits the recipe name and IV phi operand.

### Six new stubs (Commit 2)

All have no-op `execute()` bodies and minimal `print()` implementations for now. They exist to complete the enum and enable `isa<>`/`classof` dispatch.

---

## Consumer Updates

| File | Change |
|------|--------|
| `LoopTensorize.cpp` `buildInitial()` | Branch on `Phi->getType()->isPointerTy()` to select `TPWidenPointerInductionRecipe` vs `TPWidenIntOrFpInductionRecipe` |
| `TPlanLowering.cpp` | Split `TPWidenInductionRecipe::execute()` body into both concrete subclasses (same body: `State.setValue(this, IVPhi)`) |
| `TPlan.cpp` | Split `TPWidenInductionRecipe::print()` into both concrete subclasses |
| `TPRecipeMatcher.cpp` | No change — does not dispatch on induction SC directly |
| `TPlanWidener.cpp` | No change — operates on DimSets, not recipe kinds |
| Lit test (`tplan-build-print.ll`) | Update `CHECK` lines: `TPWidenInductionRecipe` → `TPWidenIntOrFpInductionRecipe` or `TPWidenPointerInductionRecipe` depending on IV type |

---

## Testing

- Existing lit test `tplan-build-print.ll` updated in Commit 1 to match renamed recipe in output.
- No new lit tests required for stubs (Commit 2) — stubs have no output-visible behavior yet.
- Build must pass with no new errors or warnings (beyond the pre-existing domination failure in lit tests).
