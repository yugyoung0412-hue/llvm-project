# TPlanAnalysis: `decomposePtrForDims` 설계 문서

**날짜:** 2026-04-07  
**브랜치:** `LoopTensorizebyClaude`  
**관련 파일:**
- `llvm/include/llvm/Transforms/Vectorize/TPlanAnalysis.h`
- `llvm/lib/Transforms/Vectorize/TPlanAnalysis.cpp`
- `llvm/lib/Transforms/Vectorize/TPlanLowering.cpp`
- `llvm/test/Transforms/LoopTensorize/basic/ptr-decompose-*.ll`

---

## 1. 배경: tensor.contract intrinsic이 필요한 정보

`LoopTensorize` pass가 matmul 루프를 `llvm.tensor.contract` intrinsic으로 lowering하려면 각 텐서마다 두 가지 정보가 필요하다.

```
llvm.tensor.contract.<Ra>d.<Rb>d.<Rc>d.<type>(
  ptr C,  [C_stride_dim0, C_stride_dim1, ...],
  ptr A,  [A_stride_dim0, A_stride_dim1, ...],  A_contract_stride,
  ptr B,  [B_stride_dim0, B_stride_dim1, ...],  B_contract_stride,
  K,
  [output_dim0, output_dim1, ...]
)
```

- **Base pointer**: 텐서의 현재 슬라이스 시작 주소
- **Per-dim byte stride**: 각 loop dimension을 1 증가시킬 때 포인터가 몇 바이트 이동하는지

이 두 정보를 LLVM IR의 GEP chain에서 자동으로 추출하는 것이 이번 작업의 핵심이다.

---

## 2. 문제: 기존 stride 추출 방식의 한계

### 2-1. Canonical IV로 인한 SCEV 단절

기존 `populateSCEVStridesFromIndex()`는 단순 while 루프로 SCEV를 탐색했다.

```
// 기존 방식
S = SE.getSCEV(GEPIdx)
while S is SCEVAddRecExpr:
    stride[S.loop] = S.stepRecurrence
    S = S.start                   ← 다음 레벨로
```

**문제**: TPlan은 모든 induction variable을 canonical화한다 (`{0, +, 1}_Loop`). 그 결과 GEP index의 SCEV start가 항상 0이 되어 while 루프가 첫 iteration 후 즉시 종료된다.

```
원래:  A[i][k]의 GEP index = i*nb1 + k*nb0
SCEV: {0, +, nb1}_outerLoop + {0, +, nb0}_innerLoop

canonical IV 적용 후 innerLoop의 start = 0
→ while 첫 iteration에서 S.start = 0
→ outerLoop stride는 추출되나 innerLoop stride 누락
```

### 2-2. GEP chain 역방향 탐색 부재

실제 메모리 접근은 여러 GEP가 체인을 이룬다.

```
load ptr:
  GEP(                              ← innermost (k)
    GEP(                            ← i
      GEP(                          ← batch2 (srem)
        GEP(A_base, batch3*nb3),    ← batch3 (srem)
        batch2%ne2 * nb2
      ),
      i * nb1
    ),
    k * nb0
  )
```

기존 코드는 load instruction의 **직접 pointer operand 하나**만 보고 SCEV를 계산했다. 체인 전체를 역방향으로 따라가지 않아 nb1, nb0 stride를 모두 놓쳤다.

### 2-3. Non-affine GEP (srem) 처리 불가

ggml의 batch broadcasting 패턴:

```c
// ggml: A는 B/C보다 batch 수가 적을 때 srem으로 wrapping
A_batch_idx = batch % A_ne_batch;   // srem
A_slice = A_base + A_batch_idx * nb_batch;
```

`srem`의 SCEV는 `SCEVCouldNotCompute`다. 기존 방식으로는:
- stride 추출 실패 → `IsAffine = false`
- `classifyPattern()`에서 Generic 반환 → `emitContraction()` 미호출
- → tensor intrinsic 미생성

### 2-4. C 포인터 탐색의 fadd→phi→store 체인 처리 부재

ggml에서 reduction 결과가 C에 저장되는 경로:

```
%114 = fadd float %acc, %fmul    ← ReductionUpdate
%96  = phi float [0.0, %entry], [%114, %loop]   ← 외부 phi
store float %96, float* %99      ← 실제 C store
```

기존 C 포인터 탐색은 `%114`의 직접 IR user만 확인했다. `%96`(phi) 경유를 처리하지 않아 C 포인터를 찾지 못했다 (`Contraction cannot find C pointer`).

---

## 3. 해결: `decomposePtrForDims()`

### 핵심 아이디어

> GEP chain을 역방향으로 걸으면서 각 dimension의 affine stride를 추출하되,
> non-affine GEP(srem 등)를 만나면 그 시점의 포인터를 **base**로 확정하고 종료한다.

이렇게 하면 srem 아래의 affine 부분(i, k stride)은 intrinsic에 포함하고, srem이 포함된 batch 차원은 outer scalar loop에 위임할 수 있다.

### 3-1. `PtrDecomposition` 구조체

```cpp
struct PtrDecomposition {
  Value *Base = nullptr;                    // non-affine까지 흡수된 base ptr
  DenseMap<unsigned, const SCEV *> Strides; // dim index → byte stride SCEV
  SmallBitVector NonAffineDims;             // srem 등으로 인해 제외된 dim
  SmallBitVector AffineDims;                // stride 추출 성공한 dim
};
```

### 3-2. `decomposePtrForDims()` 수도코드

```
decomposePtrForDims(Ptr, DimSet, DimToLoop, OutermostLoop, SE):
  Cur = Ptr
  MaxSteps = 64

  while Cur != null and MaxSteps > 0:

    // ── Case 1: bitcast / addrspacecast ──────────────────────────────
    // 포인터 값 자체는 동일하므로 투명하게 skip
    if Cur is BitCastInst or AddrSpaceCastInst:
      Cur = Cur.operand(0)
      continue

    // ── Case 2: loop-invariant PHI ───────────────────────────────────
    // OutermostLoop에 대해 loop-invariant한 incoming value를 따라간다
    // (배치 루프의 pointer-induction PHI 패턴 처리)
    if Cur is PHINode:
      Inv = 유일한 loop-invariant incoming (OutermostLoop 기준)
      if Inv found:
        Cur = Inv
        continue
      else:
        break   // ambiguous 또는 모두 loop-variant → base 확정

    // ── Case 3: single-index GEP ─────────────────────────────────────
    if Cur is GEP with exactly 1 index:
      IdxSCEV = SE.getSCEV(GEP.index)

      // non-affine (srem, udiv 등) → 이 GEP result를 base로 확정
      if IdxSCEV == SCEVCouldNotCompute:
        Base = Cur
        return (Base, Strides, AffineDims, NonAffineDims)

      // SCEV 내 모든 AddRec 항목을 worklist로 수집
      // (SCEVAdd 중첩, nested AddRec 모두 처리)
      worklist = [IdxSCEV]
      while worklist not empty:
        S = worklist.pop()

        if S is SCEVAddExpr:
          worklist.push_all(S.operands)   // 합산 항들 분해

        if S is SCEVAddRecExpr:
          Dim = LoopToDim[S.loop]
          if Dim ∈ DimSet and Dim not yet recorded:
            Strides[Dim] = S.stepRecurrence   // stride 저장
            AffineDims.set(Dim)
          worklist.push(S.start)   // 중첩 AddRec을 위해 start도 탐색

      Cur = GEP.pointerOperand   // 한 레벨 위 GEP로 이동
      continue

    // ── Case 4: 기타 ─────────────────────────────────────────────────
    break

  Base = Cur
  NonAffineDims = DimSet - AffineDims
  return (Base, Strides, AffineDims, NonAffineDims)
```

### 3-3. 4가지 체인 케이스 처리

| 체인 패턴 | 처리 방식 | 결과 |
|-----------|-----------|------|
| `bitcast(GEP(...))` | bitcast skip → GEP 분석 | stride 정상 추출 |
| `addrspacecast(GEP(...))` | addrspacecast skip → GEP 분석 | stride 정상 추출 |
| `GEP(phi[base, next], ...)` | loop-invariant incoming `base` 추적 | stride 정상 추출 |
| `GEP(GEP(base, srem*nb), i*nb1)` | srem GEP에서 멈춤, 상위 GEP result = Base | i-stride만 추출, batch-stride 제외 |

---

## 4. `emitContraction()` 변경

### 4-1. A, B 포인터 decompose

```
// 기존
LHSPtr = State.getValue(LHSLoad.ptrOperandRecipe)
// → TPlan 레시피가 이미 lowered된 값을 참조
//   srem이 포함된 경우 GEP 체인 전체 stride 누락

// 변경
ADecomp = decomposePtrForDims(LHSLoad.instruction.ptrOperand, ...)
BDecomp = decomposePtrForDims(RHSLoad.instruction.ptrOperand, ...)
LHSPtr = ADecomp.Base
RHSPtr = BDecomp.Base
```

### 4-2. Rank 재계산 및 OutputDimSet 조정

```
// affine dim 수만 intrinsic rank로 사용
RankA = ADecomp.AffineDims.count()
RankB = BDecomp.AffineDims.count()

// srem batch dim은 outer scalar loop이 처리 → intrinsic에서 제외
OutputDimSet = (A.DimSet | B.DimSet) - {ContractDim} - NonAffineDims
```

### 4-3. C 포인터 탐색 개선 (fadd→phi→store 처리)

```
findCStore(ReductionUpdate):
  // Level 0: 직접 StoreInst user
  for U in ReductionUpdate.users:
    if U is StoreInst: return U

  // Level 1: PHI 한 단계 통과
  for U in ReductionUpdate.users:
    if U is PHINode:
      for PU in U.users:
        if PU is StoreInst: return PU

  return null

// C 포인터도 동일하게 decompose
CStore = findCStore(ReductionUpdate)
CDecomp = decomposePtrForDims(CStore.ptrOperand, OutputDimSet, ...)
CPtr = CDecomp.Base
```

### 4-4. Stride 벡터 구성

```
for Dim in OutputDimSet:
  AStrides[Dim] = ADecomp.Strides[Dim]   or 0 (broadcast)
  BStrides[Dim] = BDecomp.Strides[Dim]   or 0 (broadcast)
  CStrides[Dim] = CDecomp.Strides[Dim]
                  or CStoreRecipe.getMemStride(Dim)
                  or Plan.getDenseStrideForDim(Dim)
```

---

## 5. ggml 예시: 4D matmul with srem batch broadcasting

### 입력 IR 구조

```
// A: 2 batch dim (srem), 2 inner dim (affine)
A_batch3_slice = GEP(A_base, srem(b3, A_ne3) * nb3_A)
A_batch2_slice = GEP(A_batch3_slice, srem(b2, A_ne2) * nb2_A)
A_i_slice      = GEP(A_batch2_slice, i * nb1_A)
A_ik_ptr       = GEP(A_i_slice, k * nb0_A)   ← load ptr

// B, C: 모두 affine (srem 없음)
```

### decomposePtrForDims(A_ik_ptr) 결과

```
GEP(k*nb0_A)     → Strides[dim_k] = nb0_A,  Cur = A_i_slice
GEP(i*nb1_A)     → Strides[dim_i] = nb1_A,  Cur = A_batch2_slice
GEP(srem*nb2_A)  → SCEVCouldNotCompute → Base = A_batch2_slice, 종료

결과:
  Base         = A_batch2_slice   (srem이 선택한 현재 batch 슬라이스)
  Strides      = { dim_i: nb1_A, dim_k: nb0_A }
  AffineDims   = { dim_i, dim_k }
  NonAffineDims = { dim_batch3, dim_batch2 }
```

### 생성되는 intrinsic

```
// srem batch dim은 outer scalar loop이 순회하며 slice ptr을 갱신
// intrinsic은 이미 slice된 포인터를 받아 2D (i×k, j×k) 연산만 수행

for b3 in range(B_ne3):
  for b2 in range(B_ne2):
    A_slice = A_base + srem(b3, A_ne3)*nb3 + srem(b2, A_ne2)*nb2  ← scalar
    B_slice = B_base + b3*nb3_B + b2*nb2_B
    C_slice = C_base + b3*nb3_C + b2*nb2_C
    for j in range(N):
      llvm.tensor.contract.2d.2d.2d.f32(
        C_slice + j*nb1_C,  [nb0_C],         ← 2D: i만
        A_slice,            [nb1_A], nb0_A,   ← 2D: i, k
        B_slice + j*nb1_B,  [0    ], nb0_B,   ← 2D: k만 (j는 호출 측이 처리)
        K,
        [M]
      )
```

---

## 6. 검증

### Lit 테스트

| 테스트 | 검증 내용 | 결과 |
|--------|-----------|------|
| `ptr-decompose-bitcast.ll` | bitcast 투명 skip → `Contraction (contractDim=0)` | PASS |
| `ptr-decompose-addrspacecast.ll` | addrspacecast 투명 skip | PASS |
| `ptr-decompose-srem.ll` | srem base 설정 + `cannot find C pointer` 없음 | PASS |
| `ptr-decompose-phi.ll` | loop-invariant PHI 추적 | XFAIL (LoopNestAnalyzer 미지원) |

### ggml 스모크 테스트

```bash
opt -passes=loop-tensorize --debug-only=tplan-lower \
    ggml_compute_forward_mul_mat.ll -o /dev/null 2>&1 \
    | grep -E "Contraction|cannot find"

# 출력:
#   [ir-bb<>] Contraction (contractDim=0)   ✅
#   [tensor.latch0] Contraction (contractDim=0)   ✅
# (cannot find C pointer 없음)   ✅
```

---

## 7. 설계 결정

### Non-affine dim을 intrinsic에서 제외하는 이유

srem stride를 intrinsic에 포함하려면 런타임 값(`srem(b, ne_batch)`)이 필요하다. 이는 루프마다 다른 값이므로 정적으로 stride 벡터에 넣을 수 없다. 대신 **outer scalar loop이 이미 올바른 slice pointer를 계산**하므로, intrinsic은 그 slice pointer를 base로 받아 내부 affine 차원만 처리하면 된다. 이것이 "batch peeling" 접근법이다.

### C 포인터를 PHI chain으로 탐색하는 이유

reduction 결과가 곧바로 store되지 않고 outer loop의 PHI(초기값 0.0)를 경유하는 것은 LLVM의 일반적인 패턴이다. `fadd` 자체는 loop-backedge를 통해 피드백되는 구조이기 때문에 직접 store가 붙지 않는다. PHI 한 단계를 통과하면 실제 C store를 안정적으로 찾을 수 있다.
