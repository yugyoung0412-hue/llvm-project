# Tensor Lowering Extension Design

**Date:** 2026-04-06
**Scope:** TPlanLowering — ElementWise opcode/type generalization, BroadcastBinary intrinsic, PlainReduction intrinsic
**Status:** Approved

---

## 1. Background

`TPWidenRecipe::execute()` in `TPlanLowering.cpp` lowers classified recipes to LLVM IR.
Three cases currently fall back to scalar clone instead of emitting tensor intrinsics:

| Case | Current state | Gap |
|------|--------------|-----|
| ElementWise | `llvm.tensor.elementwise.<op>.<rank>d.<type>` — only FAdd/FSub/FMul, only f32/f64 | Missing opcodes, missing types |
| BroadcastBinary | Scalar fallback (TODO comment) | No intrinsic defined |
| PlainReduction | Scalar fallback | No intrinsic defined |

Dynamic shape target: **rank is compile-time fixed, dimension sizes are runtime `i64`** (AI inference batch/sequence length variation). This is already compatible with the existing `i64` argument convention — no structural change needed.

---

## 2. ElementWise Extension

### 2.1 Type Suffix Generalization

Replace the hardcoded `isFloatTy() ? "f32" : "f64"` with a `getTypeSuffix(Type *)` helper:

```cpp
static StringRef getTypeSuffix(Type *Ty) {
  if (Ty->isHalfTy())    return "f16";
  if (Ty->isFloatTy())   return "f32";
  if (Ty->isDoubleTy())  return "f64";
  if (Ty->isIntegerTy(1))  return "i1";
  if (Ty->isIntegerTy(8))  return "i8";
  if (Ty->isIntegerTy(16)) return "i16";
  if (Ty->isIntegerTy(32)) return "i32";
  if (Ty->isIntegerTy(64)) return "i64";
  return "";  // unsupported → scalar fallback
}
```

Applied in both `getTensorElementwiseFn()` and all new `getT*Fn()` helpers.

### 2.2 Opcode Coverage

Replace the `switch(BO->getOpcode())` block with full coverage:

| Category | LLVM Opcode | Intrinsic op name |
|----------|-------------|-------------------|
| FP arithmetic | FAdd, FSub, FMul, FDiv, FRem | `fadd`, `fsub`, `fmul`, `fdiv`, `frem` |
| FP comparison | FCmp (per predicate) | `fcmp_oeq`, `fcmp_olt`, `fcmp_ole`, `fcmp_ogt`, `fcmp_oge`, `fcmp_one`, `fcmp_ord`, `fcmp_ueq`, `fcmp_ult`, `fcmp_ule`, `fcmp_ugt`, `fcmp_uge`, `fcmp_une`, `fcmp_uno` |
| Integer arithmetic | Add, Sub, Mul, SDiv, UDiv, SRem, URem | `add`, `sub`, `mul`, `sdiv`, `udiv`, `srem`, `urem` |
| Bitwise / logical | And, Or, Xor, Shl, LShr, AShr | `and`, `or`, `xor`, `shl`, `lshr`, `ashr` |
| Integer comparison | ICmp (per predicate) | `icmp_eq`, `icmp_ne`, `icmp_slt`, `icmp_sle`, `icmp_sgt`, `icmp_sge`, `icmp_ult`, `icmp_ule`, `icmp_ugt`, `icmp_uge` |

For FCmp/ICmp: result element type is `i1`, but the **type suffix reflects the operand type** (e.g., `fcmp_olt` on `f32` tensors → `llvm.tensor.elementwise.fcmp_olt.2d.f32`).

### 2.3 Intrinsic Name & Signature

Unchanged structure — only opcode name and type suffix are generalized:

```
llvm.tensor.elementwise.<op>.<rank>d.<type>

void(
  ptr C, i64×Rank C_strides,
  ptr A, i64×Rank A_strides,
  ptr B, i64×Rank B_strides,
  i64×Rank dims              ← runtime values for dynamic shapes
)
```

### 2.4 Scalar Fallback Conditions

`tryVectorize()` returns false (scalar fallback) when:
- `getTypeSuffix()` returns `""` (unsupported element type)
- Rank < 1 or Rank > 3
- Operands are not `TPWidenLoadRecipe`
- No consuming `TPWidenStoreRecipe` found for C pointer

---

## 3. BroadcastBinary Intrinsic

### 3.1 Classification Recap

Classified when one operand's DimSet is a strict subset of the other:
```
DimSet(A) ⊂ DimSet(B)  or  DimSet(B) ⊂ DimSet(A)
```
Example: A is a 1D vector `{0}`, B is a 2D matrix `{0,1}` — A is broadcast across rows of B.

### 3.2 New Helper: `getTensorBroadcastFn()`

```
llvm.tensor.broadcast.<op>.<rank_out>d.<type>
```

- `<rank_out>`: rank of the larger DimSet (output rank)
- Signature identical to `getTensorElementwiseFn()` with `rank_out` ranks

```
void(
  ptr C, i64×rank_out C_strides,
  ptr A, i64×rank_out A_strides,   ← broadcast dims use stride=0
  ptr B, i64×rank_out B_strides,   ← broadcast dims use stride=0
  i64×rank_out dims
)
```

**stride=0 convention** encodes broadcast semantics: a dimension with stride=0 repeats the same element along that axis. This follows NumPy/cuBLAS conventions and keeps the signature uniform with elementwise.

### 3.3 execute() Flow

```
tryVectorize() in BroadcastBinary case:
  Determine rank_out = max(DimSet(A).count(), DimSet(B).count())
  For each operand:
    For each dim in [0, rank_out):
      if dim ∈ operand's DimSet → use getMemStride(dim)
      else                      → stride = 0  (broadcast)
  C strides: from consuming TPWidenStoreRecipe (same as ElementWise)
  Call getTensorBroadcastFn(*Mod, OpName, rank_out, ElemTy)
  On failure → scalar fallback
```

### 3.4 Supported Opcodes

Same full opcode table as ElementWise (Section 2.2). BroadcastBinary is a shape-level distinction, not an opcode-level one.

---

## 4. PlainReduction Intrinsic

### 4.1 Classification Recap

Reduction update (`fadd`/`add`/etc. with a `TPReductionPHI` operand) where no fuseable mul-like producer exists:
```c
sum += A[i][j];         // full reduction → scalar output
row_sum[i] += A[i][j]; // partial reduction → lower-rank output
```

### 4.2 New Helper: `getTensorReduceFn()`

```
llvm.tensor.reduce.<op>.<rank_in>d.<type>
```

- `<rank_in>`: rank of the input tensor
- Accumulator (`Acc`) may be scalar (rank 0) or lower-rank tensor

```
void(
  ptr Acc, i64×rank_in Acc_strides,   ← reduction dims use stride=0
  ptr A,   i64×rank_in A_strides,
  i64×rank_in dims                    ← runtime values for dynamic shapes
)
```

**stride=0 for reduction dims** on Acc: dimensions that are being reduced have stride=0 in Acc (the accumulator does not advance along those axes). This is consistent with the BroadcastBinary stride=0 convention.

### 4.3 Supported Opcodes

| Opcode | Intrinsic op name |
|--------|-------------------|
| FAdd | `fadd` |
| FSub | `fsub` |
| FMul | `fmul` |
| FMax / FMin | `fmax`, `fmin` |
| Add | `add` |
| Sub | `sub` |
| Mul | `mul` |
| And, Or, Xor | `and`, `or`, `xor` |

### 4.4 execute() Flow

```
tryReduce() in PlainReduction case:
  Input = getReductionInput(this)  → non-PHI operand
  InputDR = dyn_cast<TPSingleDefRecipe>(Input)
  rank_in = InputDR->DimSet.count()
  A_strides = getTPValueStrides(*InputDR, Plan, SE)
  ReductionDims = Plan.getReductionDims()
  Acc pointer: extracted from TPReductionPHIRecipe (alloca or IR PHI pointer operand)
  Acc_strides:
    for each dim in [0, rank_in):
      if dim ∈ ReductionDims → stride = 0
      else                   → getMemStride(dim) from store recipe
  Call getTensorReduceFn(*Mod, OpName, rank_in, ElemTy)
  On failure → scalar fallback
```

---

## 5. Shared Infrastructure Changes

### 5.1 `getTypeSuffix()` helper

New file-static function in `TPlanLowering.cpp`, used by all three `getT*Fn()` helpers. Returns `""` for unsupported types — callers treat this as a signal to fall back to scalar.

### 5.2 `getOpcodeStr()` helper

New file-static function that maps `Instruction::BinaryOps` and `CmpInst::Predicate` to the string used in intrinsic names. Centralizes the opcode → string mapping for all three intrinsic families.

```cpp
static std::string getOpcodeStr(const Instruction *I);
// Returns "" for unsupported opcodes → scalar fallback
```

### 5.3 Rank limit

Rank 1–3 supported for all three intrinsic families. Rank 0 (scalar) and rank > 3 fall through to scalar clone. This limit can be raised later without interface changes.

---

## 6. Files Affected

| File | Change |
|------|--------|
| `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp` | Add `getTypeSuffix()`, `getOpcodeStr()`, `getTensorBroadcastFn()`, `getTensorReduceFn()`; extend `getTensorElementwiseFn()`; rewrite ElementWise/BroadcastBinary/PlainReduction cases in `TPWidenRecipe::execute()` |
| `llvm/include/llvm/Transforms/Vectorize/TPlanTypes.h` | No change |
| `llvm/test/Transforms/LoopTensorize/basic/` | New lit tests: `tensor-eltwise-int.ll`, `tensor-eltwise-fcmp.ll`, `tensor-broadcast-2d.ll`, `tensor-reduce-fadd.ll`, `tensor-reduce-partial.ll` |

---

## 7. Non-Goals

- Rank > 3 support (deferred)
- Dynamic rank (deferred — rank is compile-time fixed per Section 1)
- Runtime implementation of the new intrinsics (separate concern; lowering to target code is out of scope here)
- Changes to `TPRecipePatternMatcher` classification logic (classification is already correct)
