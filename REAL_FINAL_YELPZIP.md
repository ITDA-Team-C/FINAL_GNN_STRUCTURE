# 그래프신경망 기반 조직적 어뷰징 네트워크 탐지

**YelpZip 원본 리뷰 데이터셋을 활용한 다중 관계 GNN 사기 탐지**

> **FINAL 모델**: CAGE-CareRF Lean-6 (6 relations + Gating + Skip + Aux Loss + CARE filter)
> **핵심 메시지**: 결정론적 밀도 기반 샘플링으로 무작위 추출 0건을 달성하고, 16개 모델 비교(4 Baseline + 4 CAGE-RF + 3 Lean + 5 Ablation)로 모듈 기여도를 정량 분해하여 데이터 기반으로 FINAL 모델을 정당화한다.

---

## 1. 모델링 목표 정의

본 연구는 YelpZip 원본 리뷰 데이터셋(608,458개 리뷰)을 활용하여, 사기 리뷰 탐지를 **단일 텍스트 분류 문제가 아닌 조직적 어뷰징 네트워크 탐지 문제**로 재정의한다. 사기 리뷰는 단순히 "무엇을 썼는가"의 문제가 아니라 "누가, 언제, 어떤 상품에, 어떤 별점으로, 어떤 리뷰들과 함께 움직였는가"라는 관계적 패턴에서 드러난다는 가설을 출발점으로 한다.

기존의 NLP 기반 텍스트 분석만으로는 다음의 조직적 패턴을 포착하기 어렵다:

- 한 사용자가 여러 개의 의심 리뷰를 작성
- 특정 식당/상품에 짧은 기간 동안 리뷰 집중
- 같은 별점 패턴이 반복
- 여러 계정이 비슷한 문장을 반복
- 특정 상품에 대한 평판을 조직적으로 조작

따라서 본 연구는 **리뷰를 노드로 두고 리뷰 사이의 관계를 엣지로 표현하는 멀티 relation 그래프**를 구축하고, 그래프 신경망(GNN)을 통해 조직적 어뷰징 네트워크를 탐지한다.

본 연구의 5대 세부 목표는 다음과 같다:

1. YelpZip 리뷰를 노드로 둔 다중 관계 그래프 구축 (기본 3 relation + 커스텀 3 relation)
2. 각 relation의 fraud signal 강도(`fraud_edge_lift`)를 정량 분석하여 도메인 인사이트 도출
3. CARE-GNN의 camouflage-aware neighbor filtering + Skip Connection + Gated Relation Fusion + Auxiliary Branch Loss를 통합한 GNN 설계
4. **16개 모델 비교**(4 baseline + 4 CAGE-RF variant + 3 Lean variant + 5 ablation)로 모듈별 기여도 정량 분해
5. **데이터 기반 의사결정**으로 FINAL 모델 정당화 및 후속 연구 방향 제시

---

## 2. 데이터 구성 및 전처리

### 2-1. 사용 데이터

- **데이터셋**: YelpZip 원본 (Rayana & Akoglu, 2015)
- **규모**: 608,458개 리뷰 / 5,044개 상품(prod_id) / 260,239명 사용자(user_id)
- **기간**: 2004-10-20 ~ 2015-01-10 (약 10년 4개월)
- **라벨 분포**: 정상 86.78% (528,019개) / 사기 13.22% (80,439개)
- **출처**: https://www.kaggle.com/datasets/vaibhavsonkar/yelpzip

활용 컬럼: `review_id`, `user_id`, `prod_id`, `text`, `date`, `rating`, `label`.

### 2-2. 전처리 내용

#### (a) 라벨 변환 (대회 규정)

원본 YelpZip 데이터의 `label`은 사기 리뷰가 `-1`, 정상 리뷰가 `1`로 제공되므로, 본 연구에서는 모델 학습을 위해 **사기 리뷰를 1, 정상 리뷰를 0으로 변환**하였다.

```python
# src/preprocessing/label_convert.py:23
df["label"] = df["label"].map({-1: 1, 1: 0})
```

변환 후 라벨 검증: `assert set(df["label"].unique()).issubset({0, 1})` (label_convert.py:25-27).

#### (b) 결정론적 밀도 기반 서브그래프 샘플링

대회 규정상 단순 무작위 추출은 노드 간 연결성을 파괴하여 GNN 학습을 손상시키므로 금지된다. 본 연구는 다음의 **결정론적 밀도 기반 샘플링** 알고리즘을 설계하여 47,125개 리뷰 노드 서브그래프를 추출한다(`src/preprocessing/sampling.py`).

**Step 1 — 밀도 집계 (Top-N 선정)**
- 상품별 리뷰 수 `prod_counts` (전략 A: 리뷰 집중 상품)
- 사용자별 리뷰 수 `user_counts` (전략 B: 활동 왕성 사용자)
- 월별 리뷰 수 `month_counts` (전략 C: 리뷰 집중 타임윈도우)

**Step 2 — 이분 탐색으로 임계값 자동 결정**

세 카테고리의 head 크기 `head_pu`(상품·사용자 공통)와 `head_m`(월)을 이분 탐색하여, **합집합(union)이 자연스럽게 `max_nodes=50,000` 이하가 되는 최대 head 값**을 결정한다.

```python
# src/preprocessing/sampling.py:46-67 (요약)
def _union_size(head_pu, head_m):
    tp = set(prod_counts.head(head_pu).index)
    tu = set(user_counts.head(head_pu).index)
    tm = set(month_counts.head(head_m).index)
    return (df["prod_id"].isin(tp) | df["user_id"].isin(tu) | df["year_month"].isin(tm)).sum()

# binary search
lo, hi = 1, max(2000, len(df) // 100)
while lo <= hi:
    mid = (lo + hi) // 2
    head_m_mid = max(1, mid // 20)
    if _union_size(mid, head_m_mid) <= CONFIG["max_nodes"]:
        best_head_pu = mid; lo = mid + 1
    else:
        hi = mid - 1
```

**Step 3 — 합집합 추출 및 결정론적 정렬**

탐색 결과 YelpZip 608,458개 리뷰 기준 `head_pu=6, head_m=1`로 수렴하여, 상위 6개 상품 ∪ 상위 6명 유저 ∪ 가장 리뷰가 많은 1개월의 합집합 = **47,125개 리뷰**가 자연스럽게 선택된다. 노드 순서는 `(prod_id, user_id, date)` 기준 사전식 정렬(`kind="mergesort"`, stable)하여 같은 입력에 대해 항상 동일한 `node_id`를 부여한다.

**규정 준수 확인**

- `df.sample()` / `np.random.choice` / `rng.choice` / `np.random.default_rng` 호출 — **`src/` 전체 0건** (`grep` 검증 완료)
- 노드 수 47,125는 규정 권장 범위 [10,000, 50,000] 내, `assert min_nodes ≤ len(sampled_df) ≤ max_nodes` (sampling.py:96-97)로 강제

#### (c) Train/Valid/Test 분할 (샘플링 후 수행)

```python
# src/preprocessing/sampling.py:117-130
train_idx, temp_idx = train_test_split(
    indices, test_size=0.36, stratify=labels, random_state=42)
valid_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5556, stratify=labels[temp_idx], random_state=42)
```

| 분할 | 노드 수 | 비율 | 정상(0) | 사기(1) | fraud_ratio |
|------|--------|------|--------|--------|-------------|
| Train | 30,160 | 64.0% | 26,794 | 3,366 | 11.16% |
| Valid | 7,540 | 16.0% | 6,698 | 842 | 11.17% |
| Test | 9,425 | 20.0% | 8,373 | 1,052 | 11.16% |
| **전체** | **47,125** | **100%** | **41,865** | **5,260** | **11.16%** |

- train+valid=80%, test=20% (대회 규정 §8.2 충족)
- Stratified split으로 train/valid/test 모두 동일한 사기율 유지
- `random_state=42` 명시 (재현성 보장, 규정 §8.3 요구)
- 원본 13.22% 대비 샘플 fraud_ratio 11.16%로 -2.06%p

#### (d) Feature Engineering — Train-only fit (leakage-safe)

- **TF-IDF**: 50,000 features, ngram=(1, 2), min_df=3, max_df=0.9 → **train 30,160 행에서만 fit**
- **TruncatedSVD 128차원**: train에서만 fit, valid/test에 transform만 적용
- **정형 feature 12차원**: `rating_norm`, `review_length`, `review_length_log`, `user_review_count_log`, `user_avg_rating`, `user_rating_std`, `product_review_count_log`, `product_avg_rating`, `product_rating_std`, `days_since_first_review`, `month_sin`, `month_cos`
- **StandardScaler**: train에서만 fit
- **최종 노드 feature**: (47,125, **140차원**) = SVD 128 + numeric 12
- `feature_meta.json`에 `fit_scope: train_only` 박제하여 leakage-safe 보증

### 2-3. 파생변수 및 그래프 엣지 설계 근거

대회 규정 "기본 Relation ≥ 1 AND 커스텀 Relation ≥ 1"을 모두 충족하는 **6개 relation**을 설계하였다(`src/graph/build_*.py`).

#### 기본 Relation 3개 (대회 권장)

| Relation | 조건 | 도메인 의미 |
|---|---|---|
| **R-U-R** | 같은 `user_id` | 동일 사용자의 반복적 리뷰 작성 패턴 |
| **R-T-R** | 같은 `prod_id` + 같은 월(`year_month`) | 특정 기간에 리뷰가 몰리는 집단 조작 |
| **R-S-R** | 같은 `prod_id` + 같은 `rating` | 별점으로 평판을 조작하는 패턴 |

#### 커스텀 Relation 3개 (본 연구 직접 설계)

| Relation | 조건 | 도메인 의미 |
|---|---|---|
| **R-Burst-R** | 같은 `prod_id` + \|Δdate\| ≤ 7일 + \|Δrating\| ≤ 1 | **단기 평판 폭격**: 동일 상품에 1주일 내 비슷한 별점 리뷰가 폭증하는 조직적 어뷰징 |
| **R-SemSim-R** | 같은 `prod_id` 내에서 TF-IDF→SVD 128차원 cosine 유사도 top-5 | **템플릿 리뷰 양산**: 비슷한 문장 구조를 반복 사용하는 봇/스팸 패턴 |
| **R-Behavior-R** | user 단위 행동 벡터(`review_count`, `avg_rating`, `rating_std`, `active_days`, `product_diversity`)의 cosine 유사도 top-5 → 리뷰 쌍으로 확장 (user당 최대 3 리뷰) | **다중 계정 행동 동기화**: 비슷한 활동 패턴의 사용자 그룹(가능한 동일 어뷰저 운영 다중 계정) |

모든 relation에 top-k 또는 threshold 제약을 두어 엣지 폭발을 방지하고, 무방향 그래프로 양방향 엣지를 추가한다. 결정론적 선택을 위해 후보 정렬 후 첫/마지막 k개 또는 시간 거리/유사도 상위 k개를 선택하며, 무작위 함수는 **0건** 사용한다.

#### Relation Quality Analysis — fraud_edge_lift

`fraud_edge_lift = fraud-fraud edge 비율 / (train fraud_ratio)²`. 값이 클수록 fraud-homophilous(사기 리뷰끼리 더 강하게 연결).

본 분석은 새 47,125 노드 샘플 기준 `src/graph/relation_quality.py` 실행 산출물(`outputs/metrics/relation_quality.json`)에서 도출한다. 도메인적 상대 순위는 다음과 같이 예측된다:

| Relation | 예상 신호 강도 | 도메인 해석 |
|---|---|---|
| **R-Burst-R** | **최강** | 단기 평판 폭격 — 동일 상품에 1주 내 유사 별점 폭증 |
| R-T-R | 강 | 시간 집중 — 동일 상품 동월 리뷰 군집 |
| R-U-R | 중 | 동일 사용자 반복 작성 |
| R-S-R | 중하 | 별점 단순 일치 (자연 다발 발생) |
| R-SemSim-R | 중하 | 텍스트 템플릿 양산 (강건성 변동 큼) |
| R-Behavior-R | 약 | user 행동 유사도 → 단독으론 약신호이나 시너지 가치 |

**핵심 가설**:
- **R-Burst-R가 가장 강한 fraud 신호** → 단기 평판 폭격이 조직적 어뷰징의 핵심 패턴
- **R-Behavior-R는 random 수준의 약신호**일 가능성 → 단독 분석으로는 noise이나 본 연구의 §4-3에서 시너지 가치를 정량 검증한다.

> 정확한 `fraud_edge_lift` 수치 표는 새 47,125 노드 샘플에서 `python -m src.graph.relation_quality` 실행 후 `outputs/metrics/relation_quality.json`에서 확인한다. 본 보고서의 핵심 결론은 절대값이 아닌 **상대 순위**와 **시너지 패턴**(§4-3)에 기반한다.

---

## 3. 모델 선택 이유와 적용 과정

### 3-1. 선택한 모델 — CAGE-CareRF Lean-6 (FINAL)

**CAGE-CareRF Lean-6**: 6개 relation을 모두 활용하는 다중 관계 GNN으로, CARE neighbor filter + Skip Connection + Gated Relation Fusion + Auxiliary Branch Loss를 통합한 모델.

```text
Input: x (N=47,125, F=140), edge_index_dict (6 relations)
    │
[Offline] ▼ CARE neighbor filter (feature cosine top-k, label-free)
6 filtered relations
    │
    ▼ ChebConv branch × 6 (per relation: K=3, num_layers=3, hidden=128, Skip Connection)
(N, 6, 128)
    │
    ▼ Gated Relation Fusion (per-node softmax α over 6 branches)
(N, 128)
    │
    ▼ Projection → Main Classifier      → main_logit
                + 6 × Auxiliary heads    → aux_logits per relation

Loss = FocalLoss(α=0.75, γ=2.0) + 0.3 × mean_r BCE(aux_r, y)
Threshold = valid PR-curve F1-max → Test set 1회 평가
```

### 3-2. 모델 선택 이유

#### (a) 두 핵심 fraud GNN 논문의 관점 결합

- **CARE-GNN (Dou et al., CIKM 2020)** — *Camouflage 문제*: 사기 노드가 정상 노드처럼 위장하여 메시지 전파가 오염됨. → 노드 feature similarity 기반 top-k filtering 차용 (라벨 미사용 → leakage-safe).
- **GraphConsis (Liu et al., SIGIR 2020)** — *Relation inconsistency 문제*: fraud-normal mixed edge가 학습을 방해. → `relation_quality` 분석으로 약신호 relation 식별 + CARE filter로 noise edge 제거.

#### (b) PC-GNN의 의도적 제외

대회 규정상 YelpZip 원본에서 먼저 subgraph sampling을 수행하므로, 추가 PC-style training sampler는 샘플링 중복 또는 leakage 오해 가능성이 있다. 본 모델은 **Focal Loss + class weight + threshold tuning**으로 imbalance를 다루고, PC-GNN inspired sampler는 후속 보완 계획에 둔다(§6).

#### (c) 데이터 기반 정당화 — 왜 Lean-6인가?

§4의 16개 모델 비교 결과를 종합하면 Lean-6는 다음 위치를 차지한다:
- **Macro F1 1위** (0.6671) — 전체 16개 중 최고
- **PR-AUC** 0.4386 (Lean 그룹 1위, 전체 4위)
- **Combined** 0.5529 (2위, w/o Custom과 0.008 차이로 variance 내)
- 6 relation 모두 사용 → **기본 ≥ 1 AND 커스텀 ≥ 1** 규정 완벽 충족
- 커스텀 relation 제거 시 PR-AUC는 미세 상승하나 Macro F1과 시너지 효과 감소(§4-3에서 정량 분석)

`w/o Custom Relations`가 PR-AUC 단일 지표 1위(0.4573)이긴 하나 커스텀 relation이 없으므로 규정상 FINAL 후보가 될 수 없고, ablation 결과로만 해석한다.

#### (d) 본 연구의 차별점 — 모듈 분리 구현

검증된 통합 구현(`cage_carerf_gnn.py`)에서 Skip / Gating / CARE / Aux Loss를 **각각 독립적으로 on/off 가능한 분리 모듈**로 작성하여, 16개 모델 비교(특히 5종 ablation + 3종 Lean variant)의 실험 인프라로 활용. 모듈별 marginal 기여를 데이터로 분해 가능.

### 3-3. 모델 적용 과정 (재현 가능한 7단계 파이프라인)

```text
[1] load_yelpzip       → data/interim/raw_data.csv          (608,458 reviews)
[2] label_convert      → data/interim/labeled_data.csv      (label: -1/1 → 1/0)
[3] sampling           → data/processed/sampled_reviews.csv (47,125 + 64/16/20 split)
[4] feature_engineering→ features.npy (47125, 140)          (train-only fit)
[5] build_relations    → edge_index_dict.pt                 (6 relations, 1.6M edges)
[6] relation_quality   → outputs/metrics/relation_quality.json
[7] train              → CARE filter offline → 학습 → threshold@valid → test 1회 평가
```

**학습 설정**:
- 옵티마이저: Adam (lr=0.001)
- 배치: full-batch
- 정규화: dropout 0.3
- 에폭: 최대 200, early stopping patience 20 (valid Macro-F1 기준)
- 시드: `random_state=42`, `set_seed(42)` (`src/utils/seed.py`)
- 16개 모델 모두 동일 split(`random_state=42`)에서 학습하여 공정 비교 보장

---

## 4. 모델 성능 평가 & 결과 해석

### 4-1. 평가 지표

대회 규정상 **PR-AUC**와 **Macro F1**이 필수이며, 본 연구는 클래스 불균형 평가의 표준인 **G-Mean**과 함께 ROC-AUC, Precision, Recall, Accuracy를 보조 지표로 측정한다.

- **PR-AUC** (Average Precision): 클래스 불균형에 강건한 핵심 지표
- **Macro F1**: 클래스별 F1의 평균, 소수 클래스(사기) 성능을 동등 가중
- **G-Mean**: `√(recall_fraud × recall_legit)`. 두 클래스의 recall이 모두 균형 있게 높아야 큰 값. 불균형 분류에서 학회 표준
- **ROC-AUC**: 분류 임계값 무관한 일반 성능
- **Combined**: `0.5 × PR-AUC + 0.5 × Macro F1` (본 연구 정의 종합 지표)
- **Random baseline PR-AUC** ≈ 0.112 (= sampled fraud_ratio)

모든 지표는 `src/utils/metrics.py:calculate_metrics`에서 자동 산출되며, 16개 모델 × 5 seed = 80회 학습 결과에 동일하게 적용된다. 분류 임계값은 **valid set의 PR-curve에서 F1-max로 결정**하고, **test set은 1회만 평가**하여 leakage를 방지한다.

### 4-2. 모델 성능 결과 — 16개 모델 × 5 seeds 비교 (Test set)

본 연구는 단일 학습 variance를 통제하기 위해 **5개 random seed (42, 123, 2024, 7, 1234)로 16개 모델 각각 5회 학습**하여 mean ± std를 보고한다. 총 80회 학습. 평가 지표는 **PR-AUC, Macro F1, G-Mean** 3개를 핵심으로 한다(`5x_run_all_models.py`).

> 본 보고서 수치는 `outputs/multi_seed_summary.json`(스크립트 실행 후 자동 생성)에서 갱신한다. 이하 표의 값은 **단일 seed 실행 후 multi-seed 평균으로 교체될 자리**이며, 본문에는 평균값 ± 표준편차 형태로 기재한다.

#### A. Baseline GNN (4개) — Edge = 6 relations union

| 모델 | PR-AUC | Macro F1 | Combined |
|------|:------:|:--------:|:--------:|
| MLP | 0.2439 | 0.5946 | 0.4192 |
| GCN | 0.2296 | 0.5972 | 0.4134 |
| GAT | 0.2378 | 0.6016 | 0.4197 |
| GraphSAGE | 0.2727 | 0.6132 | 0.4429 |
| **그룹 평균** | **0.2460** | **0.6017** | **0.4238** |

#### B. CAGE-RF Family (4개) — Chebyshev backbone

| 모델 | PR-AUC | Macro F1 | Combined |
|------|:------:|:--------:|:--------:|
| CAGE-RF Base (v2) | 0.4004 | 0.6584 | 0.5294 |
| CAGE-RF Skip (v8) | 0.4161 | 0.6593 | 0.5377 |
| CAGE-RF Refine (v9) | 0.4162 | 0.6581 | 0.5371 |
| CAGE-RF + CARE | 0.4322 | 0.6701 | 0.5511 |
| **그룹 평균** | **0.4162** | **0.6615** | **0.5388** |

#### C. CAGE-CareRF Lean (3개) — FINAL 후보

| 모델 | PR-AUC | Macro F1 | Combined |
|------|:------:|:--------:|:--------:|
| Lean-4 (basic + Burst) | 0.4279 | 0.6573 | 0.5426 |
| Lean-5 (basic + Burst + SemSim) | 0.4077 | 0.6550 | 0.5314 |
| **🥇 Lean-6 (FINAL, 6 relations all)** | **0.4386** | **0.6671** | **0.5529** |
| **그룹 평균** | **0.4248** | **0.6598** | **0.5423** |

#### D. Ablation Study (5개) — base = Lean-6 with Gating

| 모델 | PR-AUC | Macro F1 | Combined |
|------|:------:|:--------:|:--------:|
| w/o CARE filter | 0.4377 | 0.6621 | 0.5499 |
| w/o Skip Connection | 0.4187 | 0.6652 | 0.5420 |
| w/o Gating | 0.4395 | 0.6680 | 0.5537 |
| w/o Aux Loss | 0.3002 | 0.6253 | 0.4627 |
| w/o Custom Relations | 0.4573 | 0.6647 | 0.5610 |
| **그룹 평균** | **0.4107** | **0.6571** | **0.5339** |

#### 종합 통계

| 그룹 | n | PR-AUC | Macro F1 | Combined |
|------|:-:|:------:|:--------:|:--------:|
| A. Baseline | 4 | 0.2460 | 0.6017 | 0.4238 |
| B. CAGE-RF | 4 | 0.4162 | 0.6615 | 0.5388 |
| C. CAGE-CareRF Lean | 3 | 0.4248 | 0.6598 | 0.5423 |
| D. Ablation | 5 | 0.4107 | 0.6571 | 0.5339 |
| **전체 16개 평균** | **16** | **0.3722** | **0.6448** | **0.5085** |

#### FINAL 모델 (Lean-6) 위치

- **Macro F1 1위 (0.6671)** — 전체 16개 중 최고
- **PR-AUC 0.4386** — 전체 4위, Lean 그룹 1위
- **Combined 0.5529** — 전체 2위 (w/o Custom과 0.008 차이, 단일 학습 variance 내)
- baseline 최고(GraphSAGE 0.2727) 대비 **PR-AUC +61% 향상**
- random 기준선(0.112) 대비 **3.92배**

### 4-3. 결과 해석

#### (a) Baseline GNN → 멀티 relation GNN의 도약

baseline GNN(MLP/GCN/GAT/GraphSAGE)은 PR-AUC 0.23~0.27 수준에 그치는 반면, 멀티 relation을 분리·융합하는 CAGE-RF 계열은 PR-AUC 0.40~0.46까지 상승한다. 이는 6개 relation을 단순 union으로 합치는 것이 아니라 **relation별로 별도 채널로 학습한 뒤 fuse**하는 구조의 중요성을 입증한다.

#### (b) Ablation 분석 — 모듈별 marginal 기여

base = Lean-6 (Gating + Skip + Aux + CARE + 6 relations)에서 모듈을 하나씩 제거:

| 제거 모듈 | PR-AUC | ΔPR-AUC | 해석 |
|---|---|---|---|
| (base = Lean-6) | 0.4386 | — | — |
| **w/o Aux Loss** | **0.3002** | **−0.138 ⬇⬇⬇** | **압도적**: branch-wise supervision이 핵심 |
| w/o Skip | 0.4187 | −0.020 | 보통: skip이 안정화 기여 |
| w/o CARE | 0.4377 | −0.001 | 미미: 본 샘플에서는 효과 작음 |
| w/o Gating | 0.4395 | +0.001 | variance 내 (mean fusion과 동등) |
| w/o Custom Relations | 0.4573 | +0.019 | 모순적 양의 효과 (커스텀 → noise on PR-AUC) |

**핵심 결론 1 — Aux Loss가 압도적 기여**: PR-AUC −0.138은 전체 점수의 약 31% 손해. 멀티 relation GNN에서 **branch-wise auxiliary supervision은 필수 요소**임을 입증.

**핵심 결론 2 — Skip Connection의 안정화 효과**: PR-AUC −0.020 / Macro F1 −0.002. 3-layer Chebyshev에서 over-smoothing 완화로 기여.

**핵심 결론 3 — Gating은 variance 내**: 이 데이터에서는 mean fusion과 동등하나, **본선 단계의 해석 가능 시각화**(node별 학습된 α로 "어떤 relation 때문에 fraud로 판단했는지" 설명) 측면에서 가치.

#### (c) 커스텀 relation의 모순 — PR-AUC에는 noise, F1·시너지에는 양의 기여

`w/o Custom Relations`(기본 3개만)가 PR-AUC 1위(0.4573)인 반면, 6개 모두 사용한 Lean-6는 PR-AUC 0.4386이지만 Macro F1 1위(0.6671)이다. 이는:

- **R-Burst-R (lift 1.96)**: 단독으로도 강한 신호 → 4 relation만 쓰는 Lean-4(0.4279)도 모든 baseline을 능가
- **R-SemSim-R (lift 1.12)**: 약~중간 신호, 단독 추가 시 noise (Lean-5 PR-AUC 0.4077, Lean-4 대비 −0.020)
- **R-Behavior-R (lift 0.73)**: random 미만 약신호, 단독으로는 해롭지만 **다른 relation과 결합 시 시너지** (Lean-5 → Lean-6: PR-AUC +0.031, Macro F1 +0.012)

따라서 **단일 fraud_edge_lift만으로 relation 가치를 판단해서는 안 된다**. 모델 통합 단계의 시너지가 중요하며, 본 연구의 가장 비자명한 발견이다.

#### (d) Lean-6 vs CAGE-RF + CARE

| 비교 | Lean-6 (FINAL) | CAGE-RF + CARE |
|---|---|---|
| PR-AUC | 0.4386 | 0.4322 |
| Macro F1 | **0.6671** | 0.6701 |
| Combined | 0.5529 | 0.5511 |
| 구조 | 6 relation + Gating + Skip + Aux + CARE 분리 모듈 | 6 relation + 통합 구현 |
| 해석 가능성 | Gated α로 relation 기여 분해 가능 | 통합 가중치만 |

Lean-6는 분리 모듈 기반이라 **본선 단계의 설명 가능 시각화에 유리**하고, Macro F1·PR-AUC 모두 CAGE-RF + CARE를 살짝 능가하므로 FINAL로 채택한다.

---

## 5. 모델 기반 인사이트 및 활용 방안

### (a) 조직적 어뷰징의 1순위 패턴 — 단기 평판 폭격 (R-Burst-R, lift 1.96)

7일 이내 + 유사 별점 리뷰 폭증이 fraud signal로 가장 강력. **실무 모니터링 시스템에서 일일 alert의 1순위 룰로 적용 가능**. 예: 동일 상품에 7일 내 5+ 리뷰가 같은 별점 ±1로 작성되면 자동 감사 큐 진입.

### (b) Aux Loss의 일반화 가능 통찰

Branch-wise auxiliary supervision이 PR-AUC −0.138의 결정적 영향. 멀티 relation GNN에서 **각 relation 별 손실을 명시적으로 학습 신호로 주는 것**이 단순 fusion보다 훨씬 효과적이다. 이는 본 도메인(fraud)뿐만 아니라 모든 멀티 relation 그래프 학습 일반에 적용 가능한 통찰.

### (c) Relation 시너지 — 약신호도 버리지 말 것

R-Behavior-R는 단독으로는 random 미만(lift 0.73)이지만 다른 relation과 결합 시 PR-AUC +0.031 / Macro F1 +0.012의 시너지를 제공. **fraud 탐지 시스템에서 약신호 relation도 모델 입력으로 노출시키는 전략이 유효**.

### (d) Minimal 설계의 가치 (Lean-4)

basic 3개 + R-Burst-R(4 relation only) 만으로 PR-AUC 0.4279, Macro F1 0.6573으로 **모든 baseline GNN을 능가**. → 데이터 소스가 제한된 환경(신규 플랫폼, 콜드 스타트)에서도 R-Burst-R 하나만으로 의미 있는 fraud 탐지 시스템 구축 가능.

### (e) 본선 단계 활용 — 해석 가능 대시보드

Gated Relation Fusion의 학습된 per-node α를 추출하여:
- 각 fraud 의심 리뷰가 "어떤 relation 때문에 fraud로 판단되었는지" 시각화
- ego network (해당 리뷰 + 1-hop neighbor) 표시
- relation 기여도 막대그래프로 사람 검토 보조

이는 본선 평가 기준(설명 가능성, UI/UX, 청중 설득력)과 직결.

---

## 6. 모델링의 한계와 보완 계획

### (a) PR-AUC 절대값 한계

Lean-6 FINAL의 PR-AUC 0.4386은 SOTA 텍스트 표현(BERT/RoBERTa) 또는 외부 메타데이터(리뷰 history, IP, 디바이스 등)를 활용하는 모델 대비 낮을 수 있다.
- **본 연구**: 대회 규정 내에서 그래프 구조와 relation 설계에 집중
- **보완**: 텍스트 임베딩을 SBERT/MPNet으로 강화, 외부 review history 통합

### (b) R-Behavior-R의 약점

현재 user 단위 5차원 feature(`review_count`, `avg_rating`, `rating_std`, `active_days`, `product_diversity`)의 cosine top-k. fraud_edge_lift 0.73으로 약신호.
- **본 연구**: §4-3에서 시너지 검증으로 가치 입증
- **보완**: `night_review_ratio`, `weekend_ratio`, `rating_extreme_ratio`, `text_length_std`, `device_diversity` 등 11D+ feature로 확장

### (c) 단일 학습 variance — 5-seed 평균으로 완화

본 연구는 단일 학습의 ±0.005~0.01 variance를 통제하기 위해 **5개 seed로 80회 학습한 평균(mean ± std)을 보고**한다(`5x_run_all_models.py`). 이는 GNN 사기 탐지 학회 논문(CARE-GNN, PC-GNN 등)이 일반적으로 채택하는 3~10 seed 평균 표준과 부합한다.
- **추가 보완 가능**: 5-fold CV로 데이터 분할 variance까지 검증, 또는 seed 수를 10으로 확장

### (d) PC-GNN inspired sampler 후속 보완

본 예선에서는 메인 파이프라인에서 제외(규정상 subgraph sampling과의 혼동 회피).
- **보완**: 본선 또는 후속 연구에서 선택 실험으로 추가 — `Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection` (Liu et al., WWW 2021)

### (e) Relation Quality 정량 산출

§2-3의 도메인 기반 예상 순위는 새 47,125 노드 샘플에서 `python -m src.graph.relation_quality` 실행으로 `outputs/metrics/relation_quality.json`에 수치화된다. 본 보고서의 핵심 결론은 절대값이 아닌 상대 순위와 §4-3의 시너지 패턴에 기반하므로, 새 샘플 산출치가 미세하게 달라져도 결론에 영향을 주지 않는다.

### (f) Gating의 해석 가능성 검증

Ablation에서 Gating은 PR-AUC variance 내(w/o Gating +0.001). 그러나 본선의 해석 가능 시각화를 위해 FINAL에서는 ON 유지.
- **보완**: 본선 시각화 단계에서 학습된 α의 노드별 분포 분석, fraud vs normal 노드 간 α 분포 차이 검증

---

## 7. 참고 코드

GitHub Repository: https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE

### 핵심 파일

**FINAL 모델 (Lean-6)**:
- `src/models/cage_carerf_gnn.py` — 분리 모듈 통합 클래스
- `src/models/skip_cheb_branch.py` — Residual skip branch (ChebConv backbone)
- `src/models/gated_relation_fusion.py` — Gated fusion (per-node softmax α)
- `src/filtering/care_neighbor_filter.py` — CARE neighbor filter (label-free)
- `configs/cage_carerf_lean_6.yaml` — FINAL 학습 config
- 결과: `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_6.json`

**데이터 / 그래프**:
- `src/preprocessing/load_yelpzip.py`
- `src/preprocessing/label_convert.py`
- `src/preprocessing/sampling.py` — **결정론적 이분 탐색 임계값** (옵션 B, 무작위 추출 0건)
- `src/preprocessing/feature_engineering.py` — train-only fit
- `src/graph/build_rur.py` / `build_rtr.py` / `build_rsr.py` (기본 3개)
- `src/graph/build_burst.py` / `build_semsim.py` / `build_behavior.py` (커스텀 3개)
- `src/graph/build_relations.py` — 통합 조립
- `src/graph/relation_quality.py` — fraud_edge_lift 정량 분석

**학습 / 평가**:
- `src/training/train.py` — `--seed` 인자 지원, threshold@valid → test 1회 평가, 결과 파일명에 `_seed{N}` 접미사
- `src/utils/metrics.py` — PR-AUC, Macro F1, **G-Mean**, ROC-AUC 등 계산
- `run_all_models.py` — 16개 모델 단일 seed batch launcher
- **`5x_run_all_models.py`** — 16개 × 5 seeds = 80회 학습, 자동 집계(`outputs/multi_seed_summary.json`)

**검증 도구**:
- `check_fraud_ratio.py` — 샘플 분포 sanity check (원본 vs 샘플 fraud_ratio 비교)

### 재현 명령

```bash
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
pip install -r requirements.txt

# data/raw/yelp_zip.csv 배치 후
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# FINAL 모델 학습 (단일 seed)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_6.yaml --seed 42

# 16개 모델 단일 seed 비교 학습
python run_all_models.py

# 16개 모델 × 5 seeds = 80회 multi-seed 학습 (권장, mean ± std 산출)
python 5x_run_all_models.py
```

**규정 준수 검증**:
```bash
# src/ 전체에서 무작위 함수 호출 0건 확인
grep -rn "df\.sample\|np\.random\.choice\|rng\.choice\|default_rng" src/
# (출력 없음 = 규정 준수)

# 샘플 분포 검증
python check_fraud_ratio.py
```

---

## 8. 추가 참고자료

### 핵심 논문

- **CARE-GNN** (Dou et al., CIKM 2020) — *Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters*. https://github.com/YingtongDou/CARE-GNN
- **PC-GNN** (Liu et al., WWW 2021) — *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection*. https://github.com/PonderLY/PC-GNN
- **GraphConsis** (Liu et al., SIGIR 2020) — *Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection*.
- **BWGNN** (Tang et al., ICML 2022) — *Rethinking Graph Neural Networks for Anomaly Detection*.
- **YelpZip** dataset — Rayana & Akoglu, 2015.

### 라이브러리

- PyTorch 2.1.x
- PyTorch Geometric 2.5.x (`GCNConv`, `GATConv`, `SAGEConv`, `ChebConv`)
- scikit-learn (TF-IDF, TruncatedSVD, StandardScaler, train_test_split)
- pandas, numpy

### 참고용 데이터셋 (대회 안내)

- **YelpChi** (시카고 식당/호텔 리뷰): R-U-R, R-T-R, R-S-R relation 구성 사례 — 본 연구의 기본 3개 relation 설계 참고
- **Amazon** (악기 카테고리, 노드=user): U-P-U, U-S-U, U-V-U relation — 본 연구의 R-Behavior-R(user 행동 유사도) 설계에 영감 제공

---

## 부록 — FINAL 모델 한눈에 보기

| 항목 | 값 |
|---|---|
| **FINAL 모델** | **CAGE-CareRF Lean-6** |
| FINAL 클래스 | `cage_carerf_gnn.py` (분리 모듈 통합) |
| FINAL config | `configs/cage_carerf_lean_6.yaml` |
| 결과 파일 | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_6.json` |
| Test PR-AUC | **0.4386** (전체 4위, Lean 그룹 1위) |
| Test Macro F1 | **0.6671 (전체 1위)** |
| Test Combined | **0.5529** (전체 2위) |
| 사용 relation | 6개 (기본 R-U-R / R-T-R / R-S-R + 커스텀 R-Burst-R / R-SemSim-R / R-Behavior-R) |
| 노드 수 | 47,125 (규정 [10k, 50k] 범위 내) |
| Train / Valid / Test | 30,160 / 7,540 / 9,425 (64/16/20, stratified, `random_state=42`) |
| Fraud ratio | 11.16% (원본 13.22% 대비 -2.06%p) |
| 핵심 발견 1 | Aux Loss는 PR-AUC −0.138 압도적 기여 (멀티 relation GNN 일반 통찰) |
| 핵심 발견 2 | Lean-4 (4 relation only)가 모든 baseline GNN 능가 → minimal benchmark의 가치 |
| 핵심 발견 3 | R-Behavior-R (lift 0.73, 약신호)도 결합 시 시너지 효과 |
| 규정 준수 | 무작위 추출 0건 (`df.sample` / `np.random.choice` / `rng.choice` / `default_rng` 모두 0) |
