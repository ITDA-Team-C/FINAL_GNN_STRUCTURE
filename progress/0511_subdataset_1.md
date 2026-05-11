# 2026-05-11 참고 데이터셋 (Amazon · YelpChi) 한계 정리

**대상**: 본 연구의 cross-dataset 검증용 보조 데이터셋
**범위**: 샘플링, split, feature, relation 의미상 차이로 인한 비교 한계 — 보고서 §6 (한계와 보완 계획)에 포함될 내용

---

## 1. 개요

본 연구의 **메인 데이터셋**은 YelpZip이며, Amazon과 YelpChi는 **모델 일반화 검증용 참고 데이터셋**으로 사용된다. 두 데이터셋은 CARE-GNN / PC-GNN repo의 표준 `.mat` 포맷을 그대로 사용하였다 (이미 features + 3 relations adjacency가 처리된 형태).

```text
amazon/data/Amazon.mat     ← features (N=11944, F=25), label, net_upu/usu/uvu
yelchi/data/YelpChi.mat    ← features (N=45954, F=32), label, net_rur/rtr/rsr
```

---

## 2. 샘플링을 수행하지 않은 이유 — 정당화

### 2-1. 노드 수가 이미 규정 권장 범위 내

대회 규정의 권장 노드 범위는 **10,000 ~ 50,000**개 (1만~5만 노드 서브그래프).

| 데이터셋 | 노드 수 | 규정 범위 내? | 샘플링 필요? |
|---|---|---|---|
| YelpZip (메인) | 608,458 | ❌ 너무 큼 | ✅ 50k로 샘플링 |
| **Amazon** | **11,944** | ✅ **범위 내** | ❌ 불필요 |
| **YelpChi** | **45,954** | ✅ **범위 내 (5만 직전)** | ❌ 불필요 |

→ 본 연구는 Amazon/YelpChi를 **전체 노드** 그대로 사용하여 cross-dataset 비교의 일관성을 유지하였다.

### 2-2. 데이터셋이 이미 학계 표준 그래프로 제공됨

`.mat` 파일은 CARE-GNN 논문(Dou et al., CIKM 2020) 및 PC-GNN 논문(Liu et al., WWW 2021)이 사용한 동일한 그래프 형식으로, **features와 adjacency matrix가 이미 구축되어 있다.** 본 연구가 직접 raw text → 그래프 변환을 수행할 필요가 없다.

만약 별도로 샘플링한다면:
- 학계 표준 벤치마크 결과와의 직접 비교가 불가능해짐
- 그래프 구조 재구성에 따른 임의 변형이 cross-dataset 비교의 의미를 약화시킴

따라서 본 연구는 **표준 형식 그대로 사용**하는 것이 cross-dataset 검증의 학문적 타당성을 보장한다고 판단하였다.

### 2-3. 대회 규정과의 관계

대회 규정상 "무작위 샘플링 금지 + 1만~5만 권장"은 **메인 데이터셋(YelpZip)** 에 적용되는 사항이다. Amazon/YelpChi는 본 연구가 자체 추가한 **검증용 보조 데이터셋**이므로 별도 규정 적용 대상이 아니며, 학계 표준 형식을 따른다.

---

## 3. 한계 및 비교 시 주의사항

### 3-1. Train/Valid/Test Split 차이

| 측면 | 본 연구 | 학계 표준 (CARE-GNN, PC-GNN) |
|---|---|---|
| Train ratio | **0.64** | 0.40 |
| Valid ratio | 0.16 | 0.20 |
| Test ratio | 0.20 | 0.40 |
| Stratified | ✅ | ✅ |
| Random seed | 42 | 다양 (paper별 다름) |

**선택 근거**: 본 연구는 YelpZip 메인 실험과 **동일한 64/16/20 stratified split**을 적용하여 데이터셋 간 비교 일관성을 확보하였다.

**한계**:
- 본 연구의 학습 데이터 비율(64%)이 학계 표준(40%)보다 크다 → **학습에 더 유리한 환경**이므로 본 연구의 PR-AUC/Macro-F1 결과는 **CARE-GNN/PC-GNN 원논문 수치보다 살짝 높게 나올 수 있다.**
- 따라서 원논문 수치와의 **직접 수치 비교는 부적절**하며, "본 연구가 학계 SOTA 능가" 같은 주장은 피해야 한다.
- 보고서에서는 "동일 split 내 다른 모델과의 상대 비교"로만 활용한다.

### 3-2. 노드 단위(Node Type) 차이

| 데이터셋 | Node | 의미 |
|---|---|---|
| **YelpZip** (메인) | Review | 리뷰 단위 — 대회 규정 |
| YelpChi (참고) | Review | 동일 |
| **Amazon** (참고) | **User** | 사용자 단위 — **다름** |

**한계**:
- Amazon은 user-level node라 YelpZip의 review-level node와 의미가 다르다.
- Cross-dataset 비교 시 "동일 GNN이 다른 node type에도 작동"이라는 **약한 검증** 정도로만 해석.
- 동일 의미의 직접 비교는 YelpChi(review-level)와 YelpZip(review-level) 사이에서만 가능.

### 3-3. Feature 차원 차이

| 데이터셋 | Feature dim | 출처 |
|---|---|---|
| YelpZip | **140D** | TF-IDF→SVD 128 + numeric 12 (본 연구 직접 생성) |
| Amazon | 25D | CARE-GNN 논문 제공 (handcrafted) |
| YelpChi | 32D | CARE-GNN 논문 제공 (handcrafted) |

**한계**:
- Feature dim과 추출 방식이 데이터셋마다 다르다.
- 본 연구 모델이 어떤 feature 환경에서도 작동하는지를 보이는 데에는 유효하나, **feature 품질 자체의 차이**는 통제되지 않는다.
- 만약 Amazon/YelpChi에서 성능이 낮게 나오더라도 모델 자체보다 feature 품질의 영향일 수 있다.

### 3-4. Relation 의미 차이

| 데이터셋 | Relations | 의미 |
|---|---|---|
| **YelpZip** (메인) | 6 (basic 3 + custom 3) | R-U-R, R-T-R, R-S-R, R-Burst-R, R-SemSim-R, R-Behavior-R |
| YelpChi | 3 | R-U-R, R-T-R, R-S-R (basic만) |
| Amazon | 3 | U-P-U, U-S-U, U-V-U (basic만, user 기준) |

**한계**:
- YelpZip에는 본 연구가 직접 설계한 **커스텀 relation 3개**(R-Burst-R / R-SemSim-R / R-Behavior-R)가 있으나, Amazon/YelpChi에는 없다.
- 따라서 Amazon/YelpChi에서는 **본 연구의 핵심 차별점(커스텀 relation 설계)이 적용되지 않는다.**
- 두 데이터셋에서는 단지 "본 모델 아키텍처(Skip + CARE + Gating + Aux)가 3-relation 환경에서도 작동하는지" 검증할 뿐이다.

### 3-5. Fraud Ratio 차이

| 데이터셋 | fraud_ratio | 비고 |
|---|---|---|
| YelpZip 원본 | 13.22% | |
| YelpZip 샘플 | 11.16% | sampling 후 |
| YelpChi | 14.53% | 학계 표준 |
| Amazon | 9.50% | 학계 표준 |

**한계**:
- 데이터셋별 imbalance 정도가 다르다 → PR-AUC random baseline도 다름.
- 따라서 **PR-AUC 절대값 비교는 부적절**. random baseline 대비 향상률(예: "random 대비 2.7배")로 정규화해 비교해야 공정.

---

## 4. 본 연구가 Amazon/YelpChi에서 얻을 수 있는 것 — 명확히 정리

### ✅ 검증 가능한 것

1. **본 모델 아키텍처의 일반화** — Skip + CARE + Gating + Aux 조합이 다른 fraud dataset에서도 baseline GNN(GCN/GAT/SAGE)을 능가하는지 확인
2. **모듈별 기여도의 robustness** — CAGE-CareRF w/o CARE, w/o Aux ablation 결과가 데이터셋 간 일관된지 확인
3. **CARE-GNN 관점의 보편적 가치** — feature similarity 기반 neighbor filtering이 다양한 데이터에서 유효한지 확인

### ❌ 검증할 수 없는 것

1. **YelpZip 메인 결과의 학계 SOTA 우위** — Amazon/YelpChi와 split 다르고 의미 다른 결과
2. **커스텀 relation의 cross-dataset 가치** — Amazon/YelpChi에 R-Burst-R 등이 없음
3. **fraud_edge_lift 분석의 일반화** — 메인 YelpZip의 6 relation 품질 분석 결과는 Amazon/YelpChi에 직접 이전 불가

---

## 5. 보고서 §6 (한계와 보완 계획) 권장 문구

다음 문구를 보고서 §6에 포함하는 것을 권장한다:

```text
본 연구는 메인 데이터셋(YelpZip)에 더해 Amazon과 YelpChi 두 fraud
detection 표준 벤치마크에서 동일 모델을 학습하여 cross-dataset 검증을
수행하였다. 두 보조 데이터셋은 노드 수가 각각 11,944개와 45,954개로
대회 규정 권장 범위(1만~5만) 내이므로 별도 샘플링을 수행하지 않고
CARE-GNN/PC-GNN repo가 제공하는 .mat 표준 그래프 형식을 그대로
사용하였다. 학습 split은 메인 실험과 동일한 64/16/20 stratified
(seed=42)을 적용하여 데이터셋 간 비교의 일관성을 확보하였다.

다만 다음 사항에 따라 cross-dataset 결과의 해석에는 주의가 필요하다:
(1) 본 연구의 train ratio(64%)는 CARE-GNN/PC-GNN 원논문의 표준(40%)
보다 크므로 학계 발표 수치와의 직접 비교는 부적절하다.
(2) Amazon은 user-level node이며 YelpZip/YelpChi의 review-level과
의미가 다르므로 동일 의미 비교는 YelpZip ↔ YelpChi 사이에서만 가능하다.
(3) 본 연구의 핵심 커스텀 relation(R-Burst-R 등) 3개는 YelpZip에만
적용 가능하므로, Amazon/YelpChi에서는 모델 아키텍처(Skip+CARE+Gating+Aux)의
보편성 검증에 한정된다.
(4) 데이터셋별 fraud_ratio 차이로 인해 PR-AUC 절대값 직접 비교는
부적절하며, random baseline 대비 향상률로 정규화해 해석한다.

따라서 본 연구는 Amazon/YelpChi 결과를 "본 모델 아키텍처가 다양한
fraud detection 환경에서도 작동함을 검증하는 보조 실험"으로 위치시키며,
메인 결과(YelpZip)의 학계 SOTA 우위를 주장하는 근거로는 사용하지 않는다.
```

---

## 6. 향후 보완 계획

본 한계를 더 강력하게 다루려면 다음 후속 연구가 가능하다:

| # | 보완 항목 | 효과 | 난이도 |
|---|---|---|---|
| 1 | Amazon/YelpChi에 학계 표준 split(40/40/20)으로 재학습 | 학계 수치와 직접 비교 가능 | 낮음 (yaml 1줄) |
| 2 | 데이터셋별 5-seed multi-run | variance 정량화 | 중간 (시간 5배) |
| 3 | YelpChi의 review 단위 raw 데이터에서 본 연구 6 relation 재구축 | 커스텀 relation 일반화 검증 | 매우 높음 |
| 4 | YelpZip의 features를 Amazon/YelpChi 형식으로 압축한 추가 실험 | feature 차원 영향 통제 | 높음 |

위 항목은 모두 예선 범위 밖이며 본선/후속 연구에서 다룰 수 있다.

---

## 7. 요약

- Amazon (11,944) 과 YelpChi (45,954) 는 이미 규정 권장 사이즈 내이므로 **샘플링 불필요**
- `.mat` 표준 형식 그대로 사용하여 학계 비교의 학문적 타당성 유지
- 단 split, node type, feature, relation 의미, fraud_ratio 등 **5가지 차이점** 존재
- 보고서에서는 "본 모델 아키텍처의 일반화 검증"으로 위치시키고, 학계 SOTA 우위 주장은 회피
- §6 한계 섹션에 명시적 문구 박제 권장
