# TPlanLowering Refactor: TPlan-Driven IR Generation

**Date:** 2026-04-13
**Branch:** LoopTensorizebyClaude
**Status:** Design approved

---

## Problem

현재 `TPlanLowering_lower()`는 두 가지 메커니즘이 혼재한다:

1. **IR 직접 수술** — `execute()` 이전에 LLVM IR BasicBlock을 직접 조작:
   - `createTensorizedLoopSkeleton()` → GuardBB + scalar clone 삽입
   - `preBuildTilingBlocks()` → floating tiling loop 스켈레톤 생성
2. **TPlan Recipe 순회** — `execute()` 시 Recipe를 소비해 IR 생성

이 혼합 구조의 결과:
- IR 수술이 TPlan 구조와 독립적으로 일어나므로 General하지 않음
- `emitContraction()`이 루프 생성 + 내용 채우기를 동시에 담당 (복잡도 집중)
- `PrebuiltTilingInfo` opaque pointer로 타입 안정성 부재
- 새 패턴 추가 시 IR 수술 코드와 Recipe 코드를 동시에 수정해야 함

---

## Goal

**Lowering 함수는 오직 TPlan 구조를 순회하며 각 노드의 `execute()`만 호출한다.**  
`execute()` 이전에 LLVM IR을 수정하는 코드는 없다.

---

## Design

### 전체 파이프라인 (새 설계)

```
buildInitial()
  → 원본 루프 구조를 TPlan으로 (기존과 동일)

TPlanWidener_widen()
  → DimSet BFS 전파 (기존과 동일)

TPRecipePatternMatcher_match()
  → TensorOpKind 분류 (기존과 동일)

TPlanPolicyAnalysis::analyze()          ← 신규 (분석 전용, TPlan/IR 변형 없음)
  → EmissionPolicy 반환
     (dim별 Inline / StaticTiled / DynamicTiled 분류)

TPlanTransformer::transform(Policy)     ← 신규 (TPlan 구조 변형, IR 미수정)
  → TPGuardBlock 삽입 (DynamicTiled dim 존재 시)
  → TPRegionBlock(innermost) → TPTilingRegion 교체
  → IsSubsumed 플래그 세팅

constructionOrder() 순회 → B->execute(State)   ← IR 생성은 오직 여기서만
```

---

## 새 TPlan 노드 타입

### TPGuardBlock : public TPBlockBase

Runtime profitability guard를 표현한다.

```
필드:
  Value *RuntimeTC           — 런타임 trip-count (SCEV 확장 결과)
  unsigned PF                — prefetch factor (guard threshold)
  TPBlockBase *TensorPath    — guard true  → tensor 경로 (기존 TPRegionBlock)
  TPBlockBase *ScalarPath    — guard false → scalar clone 경로

execute():
  1. GuardBB 생성: icmp uge RuntimeTC, PF + condbr
  2. TensorPath->execute(State)
  3. ScalarPath->execute(State)
```

### TPTilingRegion : public TPBlockBase

정적/동적 타일링 루프 구조를 표현한다. 기존 innermost `TPRegionBlock`을 교체한다.

```
필드:
  unsigned Dim               — 타일링 대상 dim index
  unsigned PF                — tile 크기
  DimEmitMode Mode           — StaticTiled | DynamicTiled
  TPBasicBlock *Body         — 기존 k.loop recipes (IsSubsumed 플래그 포함)
  TPBasicBlock *ScalarEpilogue — K%PF 나머지 (동적 K일 때)

execute():
  1. tiling loop BB 구조 생성 (header / body / latch / exit)
  2. tile IV PHI 생성 → State.ValueMap[기존 k IV recipe] = TileIV  등록
  3. Body->execute(State):
     — IsSubsumed=false recipes만 IR emit
     — WIDEN-GEP는 ValueMap에서 tile IV를 자동으로 조회 → tile offset GEP
     — Contraction recipe → tensor.contract call emit
  4. DynamicTiled이면 ScalarEpilogue->execute(State) (K%PF scalar 처리)
```

### IsSubsumed 플래그 (TPRecipeBase 확장)

```cpp
class TPRecipeBase {
  ...
  bool IsSubsumed = false;   // true이면 execute()가 no-op
};
```

`TPlanTransformer`가 tensor.contract에 흡수되는 recipes에 세팅:
- `WIDEN-LOAD` (A, B) → `IsSubsumed=true`
- `WIDEN fmul`, `WIDEN fadd` → `IsSubsumed=true`
- `WIDEN-GEP` (tile pointer 계산용) → `IsSubsumed=false`
- `Contraction` → `IsSubsumed=false`

ScalarEpilogue 및 ScalarPath의 recipes는 전부 `IsSubsumed=false` (scalar IR 그대로 생성).

---

## TPlanPolicyAnalysis

기존 `buildEmissionPolicy()`를 별도 클래스로 분리.

```cpp
class TPlanPolicyAnalysis {
public:
  EmissionPolicy analyze(const TPlan &Plan,
                         const DenseMap<unsigned, Loop *> &DimToLoop,
                         const RecipeClassMap &CM);
  // 분석만 수행 — TPlan과 IR은 변형하지 않음
};
```

---

## TPlanTransformer

Policy를 받아 TPlan 구조를 변형. IR은 수정하지 않는다.

```cpp
class TPlanTransformer {
public:
  void transform(TPlan &Plan, const EmissionPolicy &Policy,
                 ScalarEvolution &SE, LoopInfo &LI);

private:
  // DynamicTiled dim → TPGuardBlock 삽입
  void insertGuard(TPlan &Plan, const DimEmissionSpec &Spec, SE, LI);

  // innermost TPRegionBlock → TPTilingRegion 교체
  TPTilingRegion *replaceWithTilingRegion(TPRegionBlock *KRegion,
                                           const DimEmissionSpec &Spec);

  // tensor.contract에 흡수되는 recipes에 IsSubsumed=true 세팅
  void markSubsumedRecipes(TPBasicBlock *Body, const RecipeClassMap &CM);

  // ScalarEpilogue 구성 (DynamicTiled 시 K%PF 처리)
  TPBasicBlock *buildScalarEpilogue(TPBasicBlock *Body);
};
```

---

## TPlanLowering_lower() 변화

### 현재

```cpp
void TPlanLowering_lower(TPlan &Plan, ...) {
  // Stage 1-3: 기존과 동일
  TPlanWidener_widen(Plan);
  TPRecipePatternMatcher_match(Plan, CM, SE, LI);
  State.Policy = buildEmissionPolicy(Plan, ...);

  // IR 직접 수술 ← 제거 대상
  createTensorizedLoopSkeleton(...);   // IR 수술
  preBuildTilingBlocks(...);           // IR 수술

  // execute
  for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
    B->execute(State);
}
```

### 새 설계

```cpp
void TPlanLowering_lower(TPlan &Plan, ...) {
  // Stage 1-3: 동일
  TPlanWidener_widen(Plan);
  TPRecipePatternMatcher_match(Plan, CM, SE, LI);

  // 분석 (TPlan/IR 미변형)
  TPlanPolicyAnalysis Analysis;
  EmissionPolicy Policy = Analysis.analyze(Plan, DimToLoop, CM);

  // TPlan 구조 변형 (IR 미수정)
  TPlanTransformer Transformer;
  Transformer.transform(Plan, Policy, SE, LI);

  // IR 생성 — 오직 여기서만
  for (TPBlockBase *B : constructionOrder(Plan.getEntry()))
    B->execute(State);
}
```

---

## 기존 코드 처리

| 현재 코드 | 새 설계에서 |
|---|---|
| `buildEmissionPolicy()` | `TPlanPolicyAnalysis::analyze()`로 이동 |
| `createTensorizedLoopSkeleton()` | `TPGuardBlock::execute()`로 이동 |
| `preBuildTilingBlocks()` | 제거 (TPTilingRegion::execute()가 대체) |
| `emitContraction()` 내 루프 생성 코드 | 제거 (TPTilingRegion::execute()가 대체) |
| `emitContraction()` 내 tensor.contract emit | `TPContractionRecipe::execute()`로 이동 |
| `PrebuiltTilingInfo` + opaque pointer | 제거 (TPTilingRegion이 구조 보유) |
| `TPTransformState::PrebuiltTilingPtr` | 제거 |

---

## 불변 조건

1. `TPlanTransformer::transform()` 완료 후 LLVM IR BasicBlock은 원본과 동일해야 함
2. `TPTilingRegion::Body`의 recipes는 Widening/Matching 단계의 결과를 그대로 보유 (IsSubsumed 플래그만 추가)
3. `ScalarPath` 및 `ScalarEpilogue`의 모든 recipes는 `IsSubsumed=false`
4. `execute()` 호출 전에 IR을 수정하는 코드는 `TPlanLowering_lower()` 안에 없어야 함

---

## 테스트 전략

- 기존 45개 LoopTensorize lit 테스트 전부 통과 (회귀 없음)
- `TPlanTransformer` 단독 단위 테스트: Policy 입력 → TPlan 구조 출력 검증
- TPlan dump (`-debug-only=tplan`) 확인: Transformer 전후 구조 비교
- `skeleton-guard.ll`, `tiling-dynamic-k.ll` 출력 IR 동일 유지

---

## 파일 변경 범위

| 파일 | 변경 내용 |
|---|---|
| `TPlan.h` | `TPGuardBlock`, `TPTilingRegion` 추가; `TPRecipeBase::IsSubsumed` 추가 |
| `TPlan.cpp` | `TPGuardBlock::execute()`, `TPTilingRegion::execute()` 구현 |
| `TPlanLowering.cpp` | `TPlanPolicyAnalysis`, `TPlanTransformer` 추가; `preBuildTilingBlocks()` 제거; `emitContraction()` 단순화 |
| `TPlanSkeleton.h/.cpp` | `TPGuardBlock::execute()`로 흡수 후 제거 가능 |
| `TPlanLowering.cpp` | `TPlanLowering_lower()` 단순화 |
