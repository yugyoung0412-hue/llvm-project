# LoopTensorize Pass — Design Document

**Date:** 2026-02-22
**Status:** Draft
**Author:** (your name)

---

## 1. Motivation

LLVM's existing `LoopVectorize` pass operates at the SIMD level — it maps scalar loops to vector intrinsics but has no awareness of higher-level tensor operations (GEMM, CONV2D, etc.). LLM inference frameworks like `ggml` express these operations as plain scalar C++ loops, leaving significant performance on the table that hardware tensor units (x86 AMX, ARM SME) could otherwise provide.

The `LoopTensorize` pass replaces `LoopVectorize` with a unified, search-based optimization engine that:

- Recognizes high-level tensor patterns (GEMM, CONV2D, elementwise, reduction) in scalar C++ IR
- Emits hardware tensor intrinsics (AMX, SME) when the target supports them
- Falls back to LLVM vector intrinsics (`llvm.matrix.multiply`, SIMD) otherwise
- Covers all behaviors of `LoopVectorize` and `SLPVectorizer` as special cases of the same search

---

## 2. Goals & Non-Goals

### Goals

- Replace `LoopVectorize` + `SLPVectorizer` at `-O2`/`-O3`
- Detect GEMM, CONV2D, elementwise, and reduction patterns from scalar C++ IR
- Emit target-native tensor intrinsics (x86 AMX, ARM SME) guided by `TargetTransformInfo`
- Use a beam search over a finite, target-parameterized transformation space
- Score candidates with an analytical roofline cost model — no hardcoded constants
- Produce correct code for non-divisible trip counts via cleanup loops

### Non-Goals

- MLIR input (this pass operates on LLVM IR from C++ frontends)
- Runtime profiling or ML-based cost models (analytical only for now)
- GPU backends (CPU-focused initially)
- Replacing Polly (no full polyhedral scheduling infrastructure)

---

## 3. Overall Architecture

### 3.1 Pass Class

```cpp
class LoopTensorizePass : public PassInfoMixin<LoopTensorizePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
```

A **FunctionPass** (new pass manager) — not a LoopPass — because cross-loop analysis (fusion, multi-nest patterns) requires function-level visibility.

### 3.2 Pipeline Position

Replaces `LoopVectorize` in the standard optimization pipeline:

```
LoopSimplify → LCSSA → LoopTensorize   (replaces LoopVectorize + SLPVectorizer)
```

**Prerequisites:**
- `LoopSimplifyPass` — canonical loop form
- `LCSSAPass` — loop-closed SSA form
- `ScalarEvolutionAnalysis`, `LoopAnalysis`, `DependenceAnalysis`, `TargetTransformInfoAnalysis`, `DominatorTreeAnalysis`

### 3.3 Required Analyses

| Analysis | Purpose |
|---|---|
| `LoopInfo` | Loop forest structure |
| `ScalarEvolution` | Induction variable and trip count characterization |
| `DependenceInfo` | Memory dependence between loop iterations |
| `TargetTransformInfo` | Hardware specs for cost model and tensor ISA queries |
| `DominatorTree` | Loop legality checking |

### 3.4 High-Level Data Flow

```
Function
  └─ [startup] TensorISAInfo  ←  TTI.getSupportedTensorOps()
       └─ LoopNestCollector        →  list of LoopNests
            └─ LoopNestAnalyzer    →  LoopNestInfo per nest
                 └─ PatternClassifier  →  PatternHint (GEMM / CONV2D / etc.)
                      └─ BeamSearch(TransformSpace, CostModel)
                           →  BestPlan (sequence of transforms)
                                └─ CodeGen
                                     ├─ Tensor path: emit TensorOpDesc.Intrinsic
                                     └─ Vector fallback: delegate to InnerLoopVectorizer / SLP
```

---

## 4. Target-Aware Tensor ISA Interface

### 4.1 Motivation

Pattern recognition must be target-driven. On x86+AMX the pass looks for 16×16 BF16 tiled GEMM; on ARM+SME it looks for outer-product accumulate patterns; on AVX2-only targets it falls through to `llvm.matrix.multiply`. The transformation search space itself is target-parameterized — no budget is wasted on tile sizes or intrinsics the hardware cannot execute.

### 4.2 `TensorOpDesc`

A new struct describing each natively supported tensor operation:

```cpp
struct TensorOpDesc {
  enum class Kind { MatMul, Conv2D, OuterProduct, Elementwise };
  Kind           OpKind;
  unsigned       M, N, K;       // tile/operand dimensions (0 = flexible)
  Type          *InputTypeA;
  Type          *InputTypeB;
  Type          *AccumType;
  Intrinsic::ID  Intrinsic;     // e.g., x86.amx.tdpbf16ps, llvm.matrix.multiply
};
```

### 4.3 New TTI Hooks

```cpp
// In TargetTransformInfo:
bool                        hasTensorOps() const;
SmallVector<TensorOpDesc>   getSupportedTensorOps() const;
unsigned                    getTensorTileSize(Type *ElemTy) const;
```

### 4.4 Backend Implementations

| Target | TTI impl | Source of truth |
|---|---|---|
| x86 + AMX | `X86TTIImpl` | `HasAMX`, `HasAMXBF16` SubtargetFeatures (from `X86.td`) |
| ARM + SME | `ARMTTIImpl` | `HasSME`, `HasSMEF32` SubtargetFeatures (from `ARM.td`) |
| x86 AVX2 only | `X86TTIImpl` | Falls back → `llvm.matrix.multiply` |
| RISC-V + V | `RISCVTTIImpl` | `HasV` — vector fallback |

SubtargetFeatures are compiled from `.td` files by TableGen → backend TTI implementations read them at pass startup via `TTI.getSupportedTensorOps()`. The middle-end never reads `.td` files directly.

---

## 5. LoopNestCollector + LoopNestAnalyzer

### 5.1 LoopNestCollector

Walks `LoopInfo`'s loop forest, identifies outermost loops, and filters out non-analyzable nests:

```cpp
struct LoopNest {
  SmallVector<Loop *> Loops;   // outermost → innermost
  bool IsPerfectNest;          // no code between loop headers
  bool IsAffine;               // all memory accesses affine in loop IVs
};
```

**Filtered out:**
- Loops with irreducible control flow
- Loops without analyzable SCEV trip counts
- Loops marked `llvm.loop.vectorize.enable = false`

### 5.2 LoopNestAnalyzer

Produces `LoopNestInfo` per nest — the structured representation the beam search operates on.

**Step 1 — Induction variable analysis (SCEV):**

```cpp
struct InductionDesc {
  PHINode *IndVar;
  SCEV    *TripCount;
  SCEV    *Step;
};
```

**Step 2 — Memory access characterization:**

```cpp
struct MemAccess {
  Value               *BasePtr;
  SmallVector<SCEV *>  IndexExprs;  // linear combination of loop IVs per dimension
  AccessKind           Kind;        // Read / Write / ReadWrite
  Type                *ElemType;
};
```

SCEV decomposes GEP index expressions into the access matrix — the key input to pattern matching.

**Step 3 — Dependence analysis:**

Uses `DependenceInfo` to build a dependence graph. Records loop-carried dependences and which loop dimensions are dependence-free (safe to vectorize or tile).

**Step 4 — `LoopNestInfo`:**

```cpp
struct LoopNestInfo {
  SmallVector<InductionDesc> IVs;
  SmallVector<MemAccess>     Accesses;
  DependenceGraph            DepGraph;
  bool                       IsPerfectNest;
  bool                       IsAffine;
  unsigned                   Depth;
};
```

### 5.3 Pattern Classifier

A lightweight pre-pass that labels the nest before beam search, steering the search toward relevant transforms:

| Pattern | Detection signal |
|---|---|
| **GEMM** | Depth ≥ 3, 3 distinct base pointers, access matrix matches `A[i][k]`, `B[k][j]`, `C[i][j]` |
| **CONV2D** | Depth ≥ 4, sliding window access `in[n][oh+kh][ow+kw][c]` |
| **Elementwise** | All accesses index the same loop IVs, no reduction |
| **Reduction** | One write with loop-carried dep on itself, all reads affine |
| **Generic** | None of the above |

The `PatternHint` is passed to the beam search to prioritize which `TensorOpDesc` entries to try first.

---

## 6. TransformationSpace

### 6.1 Transform Primitives

| Transform | Legality | Effect on LoopNestInfo | Type |
|---|---|---|---|
| `TensorRecognize` | Nest matches a `TensorOpDesc` | Replaced by tensor intrinsic | Terminal |
| `LoopTile(dim, size)` | No cross-tile loop-carried deps; trip count known | Splits loop into outer+inner strip | Intermediate |
| `LoopPermute(order)` | Dependence direction vectors allow reorder | Reorders IVs and access matrix | Intermediate |
| `LoopUnroll(factor)` | Trip count analyzable | Replicates body N times | Intermediate |
| `LoopFuse` | Adjacent nests, same trip count, no fusion-preventing deps | Merges two nests | Intermediate |
| `Vectorize` | No loop-carried deps on innermost; element type legal | SIMD vectorization | Terminal |
| `SLPVectorize` | Isomorphic instruction chains exist | SLP across iterations | Terminal |

### 6.2 Search State

```cpp
struct SearchState {
  LoopNestInfo            Current;
  SmallVector<Transform>  Applied;
  float                   Cost;
  bool                    IsTerminal;
};
```

### 6.3 Composition DAG

```
[Root]
  ├─ TensorRecognize ──────────────────────────────────────► [Terminal]
  ├─ LoopTile ──┬─ TensorRecognize ──────────────────────► [Terminal]
  │             ├─ LoopPermute ──┬─ TensorRecognize ──────► [Terminal]
  │             │                ├─ Vectorize ────────────► [Terminal]
  │             │                └─ LoopTile (L2 tiling) ─► ...
  │             ├─ LoopUnroll ── SLPVectorize ────────────► [Terminal]
  │             └─ Vectorize ────────────────────────────► [Terminal]
  ├─ LoopPermute ──┬─ LoopTile ──► ...
  │                └─ Vectorize ─► [Terminal]
  ├─ LoopFuse ─────┬─ LoopTile ──► ...
  │                └─ Vectorize ─► [Terminal]
  └─ Vectorize ────────────────────────────────────────────► [Terminal]
```

**Rules:**
- `TensorRecognize` is tried first at every node — immediately terminal if matched
- Terminal transforms end a branch (no further expansion)
- `LoopTile` can stack (L1 tile then L2 tile for cache hierarchy)
- `LoopFuse` only applies at root level

### 6.4 Target-Parameterized Parameters

```cpp
// Tile sizes from target hardware
TileSize    = TTI.getTensorTileSize(ElemTy);          // e.g., 16 for AMX BF16
VectorWidth = TTI.getRegisterBitWidth(RGK_FixedWidth) // e.g., 512 for AVX-512
              / ElemTy->getBitWidth();
```

No hardcoded constants — all parameters derived from TTI at pass startup.

---

## 7. BeamSearch + CostModel

### 7.1 BeamSearch

```
Input:  LoopNestInfo root, TensorISAInfo, CostModel, beam_width k=8
Output: BestPlan (lowest-cost terminal SearchState)

1. beam = [SearchState(root, applied=[], cost=∞)]
2. repeat:
     next_beam = []
     for each state in beam:
       if state.IsTerminal: carry forward
       for each legal Transform T:
         new_state = apply(T, state)
         new_state.cost = CostModel.score(new_state)
         next_beam.append(new_state)
     beam = top_k(next_beam, k)
3. until all states terminal
4. return beam[0]
```

**Early exit:** `TensorRecognize` match immediately floats to beam top — it's near-always optimal when available.

**Beam width:** Default `k=8`. At `-Os`: `k=2` to minimize compile time.

### 7.2 CostModel (Roofline)

```
Score (cycles) = FLOPs / min(PeakFLOPS, MemBandwidth × ArithmeticIntensity)
```

Lower = better.

**FLOPs:**
```cpp
uint64_t FLOPs = product(IV.TripCount for IV in NestInfo.IVs)
               * countArithOpsInBody(Loop);
```

**Memory traffic (cache-aware):**

Tiling structure from `Applied` transforms is used to estimate cache residency. A tiled GEMM fitting in L2 → DRAM bytes ≈ matrix sizes, not matrix sizes³. Uses `TTI.getCacheSize(Level)`.

**Peak FLOPS (transform-dependent):**

```cpp
if (usesTensorIntrinsic(state))
  PeakFLOPS = TTI.getThroughput(TensorOpDesc.Intrinsic);  // AMX/SME peak
else if (usesVectorIntrinsic(state))
  PeakFLOPS = TTI.getArithmeticInstrCost(...) * VectorWidth;
else
  PeakFLOPS = TTI.getArithmeticInstrCost(...);            // scalar
```

**Score adjustments:**

| Condition | Multiplier |
|---|---|
| `TensorRecognize` matched | ×0.1 (strong bonus) |
| Tiled access fits in L1 | ×0.5 on memory term |
| SIMD width matches register | ×0.7 |
| Loop-carried dep forces scalar | ×3.0 penalty |
| Unaligned access | ×1.5 penalty |

**TTI queries used:**

```cpp
TTI.getCacheSize(Level)
TTI.getMemoryOpCost(...)
TTI.getArithmeticInstrCost(Opcode, Ty)
TTI.getRegisterBitWidth(RGK_FixedWidthVector)
TTI.getTensorTileSize(ElemTy)
```

All hardware constants come from TTI — zero hardcoding.

---

## 8. CodeGen

### 8.1 Phase 1 — Apply Intermediate Transforms

Applied in `BestPlan.Applied` sequence order, delegating to existing LLVM utilities:

| Transform | LLVM utility |
|---|---|
| `LoopTile` | `insertTiledLoop()` in `LoopUtils.cpp` |
| `LoopPermute` | `LoopInterchange` pass logic |
| `LoopUnroll` | `UnrollLoop()` from `LoopUnrollAndJam` |
| `LoopFuse` | `LoopFuse` pass logic |

### 8.2 Phase 2 — Terminal Lowering

#### Path A: Tensor Intrinsic (TensorRecognize)

Example: GEMM → AMX BF16 after 3-level 16×16×16 tiling:

```
for i0, for j0, for k0:
  %tile_a = call @llvm.x86.tileloadd64(&A[i0*16][k0*16], stride)
  %tile_b = call @llvm.x86.tileloadd64(&B[k0*16][j0*16], stride)
  %tile_c = call @llvm.x86.tdpbf16ps(%tile_c, %tile_a, %tile_b)
call @llvm.x86.tilestored64(&C[i0*16][j0*16], %tile_c, stride)
```

For targets without AMX/SME: `TensorOpDesc.Intrinsic` = `llvm.matrix.multiply`. Same emission path, backend lowers to available SIMD.

**Remainder handling:** Scalar cleanup loop for `TripCount % TileSize` tail iterations — identical to `InnerLoopVectorizer`'s scalar epilogue pattern.

#### Path B: Vector Intrinsics (Vectorize)

```cpp
InnerLoopVectorizer ILV(L, SE, LI, DT, TLI, TTI, VF, IC);
ILV.vectorize();
```

Loop has already been optimally shaped by Phase 1, so `InnerLoopVectorizer` sees a better loop than it would have originally.

#### Path C: SLP Vectorization (SLPVectorize)

```cpp
SLPVectorizerPass SLP;
SLP.runOnFunction(F, TTI, TLI, AA, LI, SE, DT, AC);
```

`LoopUnroll` in Phase 1 exposes independent iterations; SLP cross-vectorizes them.

---

## 9. Testing Strategy

### 9.1 Unit Tests (GTest)

`llvm/unittests/Transforms/LoopTensorize/`

| Component | Tests |
|---|---|
| `LoopNestAnalyzer` | Correct `LoopNestInfo` from crafted IR fixtures |
| `TensorPatternMatcher` | Correct classification: GEMM / CONV2D / elementwise / reduction |
| `CostModel` | Monotonically improving scores: scalar → vectorized → tiled → tensor |
| `BeamSearch` | Simple 3-loop GEMM → `TensorRecognize` is winning plan |
| `TensorISAInfo` | Correct `TensorOpDesc` for x86+AMX, ARM+SME; empty set for plain AVX2 |

### 9.2 Lit Tests (FileCheck)

`llvm/test/Transforms/LoopTensorize/`

```
basic/
  gemm-recognition.ll
  conv2d-recognition.ll
  elementwise.ll
  no-tensorize-negative.ll
x86/
  amx-bf16-gemm.ll
  avx512-fallback.ll
arm/
  sme-gemm.ll
  neon-fallback.ll
remainder/
  non-divisible-tripcount.ll
beam-search/
  tile-permute-vectorize.ll
```

**Example:**
```llvm
; RUN: opt -passes=loop-tensorize -mattr=+amx-bf16 -S < %s | FileCheck %s
; CHECK: call void @llvm.x86.tilezero
; CHECK: call void @llvm.x86.tileloadd64
; CHECK: call void @llvm.x86.tdpbf16ps
; CHECK: call void @llvm.x86.tilestored64
; CHECK-NOT: br i1
```

**Negative tests:** Non-affine loops, unanalyzable trip counts, aliased pointers, `-O0` (no-op).

### 9.3 Performance Benchmarks

`llvm/benchmarks/LoopTensorize/` (Google Benchmark)

**Compile-time:** Beam search overhead vs `LoopVectorize` baseline — target < 5% on typical TUs.

**Runtime:**

| Kernel | Baseline | Expected speedup |
|---|---|---|
| GEMM FP32 512×512 | LoopVectorize AVX2 | 2–4× with AMX |
| GEMM BF16 1024×1024 | scalar ggml | 8–16× with AMX BF16 |
| CONV2D 3×3 | LoopVectorize | 2–3× with tiling + SME |
| Elementwise (ReLU) | LoopVectorize | ~1× (parity) |

### 9.4 Correctness Validation

Every benchmark kernel is verified against a reference scalar implementation:

```cpp
ASSERT_NEAR(output[i][j], ref[i][j], 1e-3f);  // BF16 tolerance
```

**Edge cases:** Non-power-of-2 dimensions, batch size 1, transposed operands, aliased input/output (must fall back safely).

---

## 10. File Layout

```
llvm/lib/Transforms/Vectorize/
  LoopTensorize.cpp            // main pass, LoopNestCollector, BeamSearch driver
  LoopNestAnalyzer.cpp         // LoopNestAnalyzer, PatternClassifier
  TensorTransformSpace.cpp     // TransformationSpace, SearchState, transforms
  TensorCostModel.cpp          // roofline cost model
  TensorCodeGen.cpp            // CodeGen, tensor intrinsic emission

llvm/include/llvm/Transforms/Vectorize/
  LoopTensorize.h
  LoopNestAnalyzer.h
  TensorISAInfo.h              // TensorOpDesc definition

llvm/include/llvm/Analysis/
  TargetTransformInfo.h        // new TTI hooks: hasTensorOps(), getSupportedTensorOps()

llvm/lib/Target/X86/
  X86TargetTransformInfo.cpp   // X86TTIImpl: AMX TensorOpDesc
llvm/lib/Target/AArch64/
  AArch64TargetTransformInfo.cpp  // ARMTTIImpl: SME TensorOpDesc

llvm/unittests/Transforms/LoopTensorize/
llvm/test/Transforms/LoopTensorize/
llvm/benchmarks/LoopTensorize/
```

---

## 11. Open Questions

1. **Interaction with LTO:** Should LoopTensorize run at LTO time for cross-TU loop fusion opportunities?
2. **Dynamic shapes:** ggml sometimes has runtime-variable matrix dimensions — how should the pass handle unknown trip counts gracefully?
3. **Instruction scheduling:** After AMX tile emission, is additional scheduling needed, or does the backend handle it?
4. **Integration with `-ffast-math`:** BF16 accumulation introduces rounding differences — should tensorization require `-ffast-math` or be opt-in?
