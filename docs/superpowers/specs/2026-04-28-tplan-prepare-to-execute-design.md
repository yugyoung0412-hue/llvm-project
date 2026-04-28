# TPlan::prepareToExecute 재설계

**날짜:** 2026-04-28
**목적:** `TPlan::prepareToExecute`가 `VPlan::prepareToExecute`를 올바르게 모방하도록 수정

---

## 배경

`TPlan::prepareToExecute`는 VPlan의 동일 함수를 텐서화(tensorization)에 맞게 모방하기 위해 작성되었다. 그러나 현재 구현에는 다음 문제가 있다.

### 현재 구현의 문제

| 항목 | VPlan (올바른 방식) | TPlan (현재 잘못된 방식) |
|---|---|---|
| TensorTripCount | 외부에서 계산된 `floor(TC/VF)*VF`를 파라미터로 수신 | `CanonicalIVStartValue`를 TTC 대용으로 사용 (의미 혼용) |
| BackedgeTakenCount | 원본 스칼라 `TC - 1` | `TTC - 1` (잘못된 기반값) |
| BTC 가드 | `getNumUsers() > 0`일 때만 생성 | 무조건 생성 |
| CanonicalIV 시작값 | `IV->setOperand(0, VPV)`로 교체 | 누락 |

### 인덕션 변수 불일치

`BranchOnCount`는 `index.next == TTC`일 때 루프를 종료한다. 인덕션 변수가 `TFxUF`씩 증가한다면 `TTC`는 반드시 **총 스텝 수** (`floor(TC/TF)*TF`) 여야 한다.

```
N=10, TF=4 기준:
  올바른 TTC = floor(10/4)*4 = 8
  인덕션: 0 → 4 → 8 (== TTC) → 종료  ✓

  현재 TTC = TC/TF = 2  (반복 횟수 개념)
  인덕션: 0 → 4 → 8 → 12 → ...  절대 TTC(=2)에 도달 못함 → 무한 루프 ✗
```

---

## 목표

VPlan 방식에 맞게 `prepareToExecute`를 재설계한다.

- TTC = 총 스텝 수 (`floor(TC/TF)*TF`), 외부에서 IR Value로 계산해 전달
- BTC = 원본 스칼라 `TC - 1`
- `CanonicalIVStartValue`는 에필로그 루프의 IV 시작값 교체에만 사용

---

## 설계 (접근법 A)

### 변경 범위

```
executePlan (TPlanner.cpp)
  └─ TensorTripCountV 계산 후 prepareToExecute에 전달

TPlan::prepareToExecute 시그니처 (TPlan.h)
  └─ MapVector<Loop*, Value*> TensorTripCountV 파라미터 추가

TPlan::prepareToExecute 바디 (TPlan.cpp)
  └─ TensorTripCount: TensorTripCountV에서 직접 대입
  └─ BackedgeTakenCount: TC - 1로 변경 + getNumUsers() 가드 추가
  └─ CanonicalIV start operand 업데이트 (TODO: 에필로그 루프 지원 시점에 구현)
```

---

### 변경 1: executePlan에서 TensorTripCountV 계산

`createTensorLoopSkeleton` 호출 직후, `prepareToExecute` 호출 직전에 삽입한다.

```cpp
MapVector<Loop*, Value*> TensorTripCountV;
IRBuilder<> Builder(State.TBS.TPH->getTerminator());

for (auto &[L, TC_SCEV] : BestTPlan.getTripCount()) {
    // SCEV → IR Value
    Value *TC_Val = tputils::getOrCreateTPValueForSCEVExpr(
        BestTPlan, TC_SCEV, *SE)->getLiveInIRValue();

    unsigned TFVal = State.TF[L].getKnownMinValue();
    Value *TF_IR = ConstantInt::get(TC_Val->getType(), TFVal);

    // floor(TC / TF) * TF = TC - (TC % TF)
    Value *Rem = Builder.CreateURem(TC_Val, TF_IR, "tc.rem");
    Value *TTC = Builder.CreateSub(TC_Val, Rem, "tensor.trip.count");
    TensorTripCountV[L] = TTC;
}

BestTPlan.prepareToExecute(TensorTripCountV, CanonicalIVStartValue, State);
```

---

### 변경 2: prepareToExecute 시그니처

```cpp
// TPlan.h
void prepareToExecute(MapVector<Loop *, Value *> TensorTripCountV,
                      MapVector<Loop *, Value *> CanonicalIVStartValue,
                      TPTransformState &State);
```

---

### 변경 3: prepareToExecute 바디

```cpp
void TPlan::prepareToExecute(
    MapVector<Loop *, Value *> TensorTripCountV,
    MapVector<Loop *, Value *> CanonicalIVStartValue,
    TPTransformState &State) {

  IRBuilder<> Builder(State.TBS.TPH->getTerminator());

  // [1] TFxUF = TF * UF  (기존 로직 유지)
  for (auto &[L, TFxUFTPV] : getTFxUF()) {
    // ... (현재 구현과 동일)
  }

  // [2] TensorTripCount: 외부에서 받은 IR Value를 직접 대입
  //     VPlan의 VectorTripCount.setUnderlyingValue(VectorTripCountV)에 해당
  for (auto &[L, TTC_VPVal] : TensorTripCount) {
    auto It = TensorTripCountV.find(L);
    if (It != TensorTripCountV.end() && It->second)
      TTC_VPVal->setUnderlyingValue(It->second);
  }

  // [3] BackedgeTakenCount = TC - 1  (VPlan 방식)
  //     이전: TTC - 1  →  변경: 원본 스칼라 TC - 1
  for (auto &[L, BTCPV] : BackedgeTakenCount) {
    if (!BTCPV->getNumUsers()) continue;  // VPlan과 동일한 가드
    auto TCIt = TripCount.find(L);
    if (TCIt != TripCount.end() && TCIt->second) {
      Value *TCVal = tputils::getOrCreateTPValueForSCEVExpr(
          *this, TCIt->second, *State.SE)->getLiveInIRValue();
      Value *One = ConstantInt::get(TCVal->getType(), 1);
      BTCPV->setUnderlyingValue(
          Builder.CreateSub(TCVal, One, "trip.count.minus.1"));
    }
  }

  // [4] CanonicalIV start operand 업데이트 (에필로그 루프 진입 시)
  //     VPlan: IV->setOperand(0, getOrAddLiveIn(CanonicalIVStartValue))
  //     TODO: TPlan에서 루프별 canonical IV를 찾는 방법 확인 후 구현
}
```

---

## 미결 사항

- **CanonicalIV 업데이트 (섹션 4)**: `getCanonicalIV(L)` 또는 이에 해당하는 루프별 accessor가 TPlan에 존재하는지 확인 필요. 에필로그 루프 지원 전까지는 TODO로 남긴다.
- **타입 일관성**: `TC_Val`은 SCEV 확장 결과 (i64 가능), `TF_IR`은 같은 타입으로 맞춰야 한다. `ConstantInt::get(TC_Val->getType(), TFVal)` 사용으로 보장.

---

## VPlan 대응 요약

| VPlan | TPlan |
|---|---|
| `TripCountV` | `TripCount[L]` SCEV 확장 |
| `VectorTripCountV` | `TensorTripCountV[L]` (새로 추가) |
| `VectorTripCount.setUnderlyingValue(V)` | `TensorTripCount[L]->setUnderlyingValue(V)` |
| `VFxUF.setUnderlyingValue(createStepForVF(...))` | `TFxUF[L]->setUnderlyingValue(TF*UF)` |
| `BackedgeTakenCount`: `TC - 1` | 동일 (`TC - 1`) |
| `IV->setOperand(0, VPV)` | TODO |
