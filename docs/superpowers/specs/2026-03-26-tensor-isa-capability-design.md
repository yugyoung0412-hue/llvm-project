# TensorISACapability Design Spec

**Date:** 2026-03-26
**Goal:** Extend TTI to expose ISA instruction capabilities so that
`TPlanWidener_widen()` can set per-dimension PF values based on the actual
hardware (SIMD width, matrix tile shape, dot-product depth, outer-product
support, vector×matrix) instead of the current hardcoded PF=256.

---

## 1. New Struct: `TensorISACapability` (in `TensorISAInfo.h`)

`TensorOpDesc` describes *one operation* (kind + tile + type).
`TensorISACapability` is the *resolved capability profile* for a given element
type — the single struct the widener reads to decide all PFs.

```cpp
/// ISA capability profile for one element type.
/// Returned by TargetTransformInfo::getTensorISAInfo(ElemTy).
struct TensorISACapability {

  //--- Elementwise / vectorization ----------------------------------------
  /// Native SIMD lane count for this element type.
  /// = getRegisterBitWidth(RGK_FixedWidthVector) / ElemTy->getScalarSizeInBits()
  /// Drives PF for plain-vector dimensions (non-reduction outer loops).
  unsigned SIMDWidth = 1;

  //--- Matrix-multiply tile (AMX, SME MMLA, WMMA, etc.) -------------------
  /// Hardware tile dimensions M×K×N.  All zero when no dedicated matmul hw.
  unsigned MatM = 0;
  unsigned MatK = 0;
  unsigned MatN = 0;

  //--- Dot-product reduction depth (VNNI, SDOT, UDOT, FDOT) ---------------
  /// How many consecutive input elements reduce into one accumulator lane.
  /// Examples: 4 for AVX-512 VNNI int8, 2 for BF16 DPBF16PS, 4 for AArch64 SDOT.
  /// 1 means no dot-product instruction (fall back to SIMD).
  unsigned DotDepth = 1;

  //--- Outer-product tile (SVE FMOPA, AMX tile outer-product mode) ---------
  /// Hardware outer-product tile rows×cols.  Both zero when not available.
  unsigned OPRows = 0;
  unsigned OPCols = 0;

  //--- Vector × matrix (NEON SDOT as vec×mat, SME GEMV) -------------------
  /// True when the ISA can multiply a vector against a matrix panel in one
  /// instruction, without needing a full M×N×K matmul tile.
  bool HasVecMatMul = false;
};
```

---

## 2. New TTI Method Declaration (`TargetTransformInfo.h`)

Add alongside `getTensorTileSize`:

```cpp
/// Returns the ISA capability profile for tensor/vector operations
/// on elements of type \p ElemTy.
/// The default implementation derives SIMDWidth from getRegisterBitWidth;
/// targets override to expose hardware matmul, dot-product, and outer-product.
LLVM_ABI TensorISACapability getTensorISAInfo(Type *ElemTy) const;
```

**Default impl** in `TargetTransformInfoImpl.h`:

```cpp
virtual TensorISACapability getTensorISAInfo(Type *ElemTy) const {
  TensorISACapability Cap;
  unsigned RegBits =
      getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();
  unsigned ElemBits = ElemTy->getPrimitiveSizeInBits();
  Cap.SIMDWidth = (ElemBits > 0 && RegBits > 0) ? RegBits / ElemBits : 1;
  // All matrix/dot/outerproduct fields remain zero/false — scalar ISA assumed.
  return Cap;
}
```

---

## 3. Target Overrides (Illustrative)

### x86 (`X86TargetTransformInfo.cpp`)

| Element type | CPU feature      | SIMDWidth | MatM/K/N   | DotDepth | OPRows/Cols | HasVecMatMul |
|---|---|---|---|---|---|---|
| `float`      | AVX-512          | 16        | 0/0/0      | 1        | 0/0         | false        |
| `int8`       | AMX-INT8         | 16        | 16/64/16   | 4        | 0/0         | false        |
| `bf16`       | AMX-BF16         | 16        | 16/32/16   | 2        | 0/0         | false        |
| `int8`       | VNNI (no AMX)    | 16        | 0/0/0      | 4        | 0/0         | false        |

### AArch64 (`AArch64TargetTransformInfo.cpp`)

| Element type | CPU feature      | SIMDWidth | MatM/K/N   | DotDepth | OPRows/Cols | HasVecMatMul |
|---|---|---|---|---|---|---|
| `float`      | NEON             | 4         | 0/0/0      | 1        | 0/0         | false        |
| `int8`       | NEON+SDOT        | 4         | 0/0/0      | 4        | 0/0         | true         |
| `float`      | SVE FMOPA        | VL/32     | 0/0/0      | 1        | VL/32/VL/32 | false        |
| `float`      | SME MMLA         | VL/32     | 4/1/4      | 1        | 0/0         | false        |

---

## 4. PF-Mapping Logic in `LoopTensorize.cpp`

Insert between `TPlan::buildInitial()` and `TPlanLowering_lower()`:

```cpp
// --- ISA-aware PF selection ---
TensorISACapability Cap = TTI.getTensorISAInfo(ElemTy);

auto applyGEMMPF = [&](unsigned iDim, unsigned jDim, unsigned kDim) {
  if (Cap.MatM > 0) {
    // Hardware matrix-multiply: lock to tile shape.
    Plan.setDimPF(iDim, Cap.MatM);
    Plan.setDimPF(jDim, Cap.MatN);
    Plan.setDimPF(kDim, Cap.MatK);
  } else if (Cap.DotDepth > 1) {
    // Dot-product: k must be a multiple of DotDepth; vectorize j.
    Plan.setDimPF(iDim, 1);
    Plan.setDimPF(jDim, Cap.SIMDWidth);
    Plan.setDimPF(kDim, Cap.DotDepth);
  } else if (Cap.OPRows > 0) {
    // Outer-product: tile rows×cols.
    Plan.setDimPF(iDim, Cap.OPRows);
    Plan.setDimPF(jDim, Cap.OPCols);
    Plan.setDimPF(kDim, 1);
  } else {
    // Plain SIMD: vectorize the innermost non-reduction dim (j).
    Plan.setDimPF(iDim, 1);
    Plan.setDimPF(jDim, Cap.SIMDWidth);
    Plan.setDimPF(kDim, 1);
  }
};

switch (Hint.Kind) {
case PatternKind::GEMM:
  // Dimensions: 0=i(row), 1=j(col), 2=k(reduction).
  applyGEMMPF(/*i=*/0, /*j=*/1, /*k=*/2);
  break;

case PatternKind::Conv2D:
  // Treat as: M=ow/oh, N=oc, K=ic×kh×kw.
  // Map N→SIMDWidth (output channels vectorized), K→DotDepth or MatK.
  if (Cap.MatM > 0) {
    Plan.setDimPF(/*ow=*/3, Cap.MatM);
    Plan.setDimPF(/*oc=*/1, Cap.MatN);
    Plan.setDimPF(/*ic=*/4, Cap.MatK);
  } else {
    Plan.setDimPF(/*ow=*/3, 1);
    Plan.setDimPF(/*oc=*/1, Cap.SIMDWidth);
    Plan.setDimPF(/*ic=*/4, Cap.DotDepth);
  }
  break;

case PatternKind::Elementwise:
  // Vectorize innermost only.
  for (unsigned D = 0; D + 1 < InfoOpt->IVs.size(); ++D)
    Plan.setDimPF(D, 1);
  if (!InfoOpt->IVs.empty())
    Plan.setDimPF(InfoOpt->IVs.size() - 1, Cap.SIMDWidth);
  break;

default:
  // Generic: vectorize dim0 only.
  Plan.setDimPF(0, Cap.SIMDWidth);
  break;
}
```

---

## 5. File Change Summary

| File | Change |
|---|---|
| `llvm/include/llvm/Transforms/Vectorize/TensorISAInfo.h` | Add `TensorISACapability` struct |
| `llvm/include/llvm/Analysis/TargetTransformInfo.h` | Declare `getTensorISAInfo(Type*)` |
| `llvm/include/llvm/Analysis/TargetTransformInfoImpl.h` | Default impl (SIMDWidth from register width) |
| `llvm/lib/Analysis/TargetTransformInfo.cpp` | Forward to impl |
| `llvm/lib/Transforms/Vectorize/LoopTensorize.cpp` | PF-mapping block above `TPlanLowering_lower()` |
| `llvm/lib/Target/X86/X86TargetTransformInfo.cpp` | x86 override (AMX/VNNI) |
| `llvm/lib/Target/AArch64/AArch64TargetTransformInfo.cpp` | AArch64 override (SME/SDOT) |

---

## 6. Why a Separate Struct (Not Extending `TensorOpDesc`)

`TensorOpDesc` is one entry in an *operation table* (per-instruction capability,
used for code-gen selection). `TensorISACapability` is the *resolved profile* for
a given element type — a single struct the widener reads to decide all PFs.

Keeping them separate means:
- `getSupportedTensorOps()` stays as a list of concrete instructions for
  code-gen.
- `getTensorISAInfo()` gives the *best capability summary* for one element type
  for tiling/PF decisions.

Internally, the default target impl can derive `TensorISACapability` from
`getSupportedTensorOps()` by picking the best matching `TensorOpDesc` for that
type.
