# CAGE-CareRF GNN — YelpZip 사기 리뷰 탐지

**Multi-Relation GNN with Camouflage-Aware Filtering for Organized Abusing Network Detection**
ITDA Networking Day 2026 학술제 본선 코드 — Team UnivConcat (김 재 현 · 백 수 연 · 홍 예 진)

본 repo는 본 연구의 **메인 코드 베이스이자 공식 제출 대상**이다. 텍스트 인코더 5종 비교 후 최종 채택된 시스템(backbone = CAGE-RF + CARE / encoder = concat)을 동일 파이프라인 위에서 재현할 수 있도록 구성되어 있다.

---

## 0. 공식 제출 자료

| 항목 | 위치 |
|---|---|
| **메인 코드 (본 repo)** | https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE |
| **시각화 결과물 (Streamlit)** | https://github.com/ITDA-Team-C/Streamlit_CAGE-RF-with-CARE |
| **분석 보고서** | `연합1조_분석보고서.pdf` |
| **시각화 요약 보고서** | `연합1조_시각화요약보고서.pdf` |

### Encoder ablation 단계별 reference repo

| 단계 | 변형 | Repo |
|:---:|---|---|
| 1단계 | frozen-SBERT 단독 | https://github.com/ITDA-Team-C/CARE-RF-GNN_With_BERT |
| 2단계 | TF-IDF ⊕ SBERT concat (최종 채택) | https://github.com/ITDA-Team-C/CAGE-RF-GNN-CONCAT-BERT_WITH_TF_DF |
| 추가 실험 | Linear projection (sbert_proj · concat_proj) | https://github.com/MeDeoDuck/FINAL_FINAL_EXPERIMENT |

---

## 1. 공식 최종 시스템

| 구성요소 | 채택 |
|---|---|
| **Backbone (모델)** | **CAGE-RF + CARE** (Skip + Gated Fusion + Aux Loss + offline CARE filter) |
| **Encoder (텍스트 → 노드 feature)** | **concat** (TF-IDF SVD-128 ⊕ frozen-SBERT SVD-128 → 모달리티별 train-only z-score → 공동 train-only SVD-128) |

### 5 seeds 평균 성능 (Test set)

| 지표 | 값 |
|---|:---:|
| **PR-AUC** | **0.4439 ± 0.0152** |
| **Macro F1** | **0.6670 ± 0.0036** |
| G-Mean | 0.6131 ± 0.0257 |
| ROC-AUC | 0.8230 ± 0.0040 |

PR-AUC(사기 탐지력)와 Macro F1(양 클래스 균형) 두 핵심 지표 모두에서 1위와 std 내 통계적 동률을 동시에 달성한 균형형 인코더로 concat을 채택하였다.

---

## 2. 핵심 결과 — 75회 학습(15모델 × 5 seeds) + 인코더 5종 ablation

### 2-1. 15모델 ablation (TF-IDF 인코더 기준, PR-AUC 내림차순 상위)

| Rank | Model | PR-AUC | Macro F1 | G-Mean | ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **CAGE-RF + CARE (backbone)** | 0.4447 ± 0.0061 | 0.6647 ± 0.0066 | 0.6049 ± 0.0405 | 0.8250 ± 0.0021 |
| 2 | w/o Gating (≡ Lean-6) | 0.4334 ± 0.0102 | 0.6622 ± 0.0039 | 0.6063 ± 0.0174 | 0.8201 ± 0.0011 |
| 3 | w/o Skip | 0.4301 ± 0.0163 | 0.6650 ± 0.0048 | 0.6305 ± 0.0238 | 0.8212 ± 0.0020 |
| 4 | CAGE-CareRF Lean-4 | 0.4296 ± 0.0136 | 0.6625 ± 0.0050 | 0.6124 ± 0.0153 | 0.8179 ± 0.0043 |
| 9 | w/o Aux Loss | 0.2982 ± 0.0203 | 0.6212 ± 0.0067 | 0.5817 ± 0.0357 | 0.7700 ± 0.0152 |
| 10 | ChebConv (baseline 최고) | 0.2752 ± 0.0055 | 0.6128 ± 0.0036 | 0.6058 ± 0.0128 | 0.7556 ± 0.0035 |

핵심: Aux Loss 제거 시 −0.1465(33% 손실, 압도적), CARE 제거 시 −0.0203(결정타), Skip·Gating은 variance 내. Baseline 평균 대비 PR-AUC +62%.

### 2-2. 인코더 5종 비교 (FINAL backbone 위에서, 5 seeds)

| Rank | 변형 | PR-AUC | Macro F1 | G-Mean | ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | TF-IDF | 0.4447 ± 0.0061 | 0.6647 ± 0.0066 | 0.6049 ± 0.0405 | 0.8250 ± 0.0021 |
| 2 | **concat ⭐ 채택** | 0.4439 ± 0.0152 | **0.6670 ± 0.0036** | **0.6131 ± 0.0257** | 0.8230 ± 0.0040 |
| 3 | frozen-SBERT | 0.4313 ± 0.0131 | **0.6686 ± 0.0074** | **0.6179 ± 0.0223** | 0.8159 ± 0.0038 |
| 4 | frozen-SBERT + Linear | 0.4083 ± 0.0176 | 0.6604 ± 0.0112 | 0.5917 ± 0.0246 | 0.7938 ± 0.0058 |
| 5 | concat + Linear | 0.3616 ± 0.0076 | 0.6450 ± 0.0056 | 0.5837 ± 0.0311 | 0.7846 ± 0.0042 |

핵심: TF-IDF가 PR-AUC 1위, frozen-SBERT가 Macro F1 1위로 단일 인코더는 한 지표에 편중. concat은 두 지표 모두에서 1위와 std 내 동률 → 균형형 채택. 학습 가능 Linear 변형 2종은 적은 라벨 환경에서 SVD를 능가하지 못해 인코더 fine-tune류 적응의 한계 실증.

---

## 3. 환경

| 항목 | 값 |
|---|---|
| Python | 3.11+ 권장 (검증 3.13) |
| OS | Linux / macOS / Windows |
| GPU | 강력 추천 (CPU 학습은 매우 느림) |
| 핵심 의존성 | PyTorch · PyG · scikit-learn · sentence-transformers · pandas |

```bash
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. 빠른 실행 — 공식 최종 시스템 재현

```bash
# 1) 데이터 배치 (~410MB, repo 미포함)
mkdir -p data/raw
# yelp_zip.csv (user_id, prod_id, rating, label, date, text, tag) 를 data/raw/ 에 둠

# 2) 전처리 → 그래프 빌드 → 학습 (TEXT_ENCODER=concat 이 기본값)
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
TEXT_ENCODER=concat python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# 3) FINAL 단일 학습
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42

# 4) 학회 표준 multi-seed (15 × 5 = 75회) — 권장
python 5x_run_all_models.py
```

### 인코더 교체 (TEXT_ENCODER 환경변수)

| TEXT_ENCODER | 설명 |
|---|---|
| `concat` (기본값, 공식 최종) | TF-IDF ⊕ frozen-SBERT → joint SVD-128 |
| `tfidf` | TF-IDF → train-only SVD-128 |
| `sbert` | frozen SBERT → train-only SVD-128 |
| `sbert_proj` | frozen SBERT + 학습 가능 Linear(384→128) |
| `concat_proj` | [frozen SBERT, TF-IDF] + 학습 가능 Linear(512→128) |

`sbert_proj` · `concat_proj` 두 변형은 모델에 `TextProjectionWrapper`가 자동 적용된다.

```bash
# 학습 가능 Linear projection 두 변형 × 5 seeds
python run_proj_experiments.py
```

---

## 5. FINAL 모델 구조 — CAGE-RF + CARE

```
Reviews (N = 47,125, F = 140 = text 128 + numeric 12)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
6 relation graphs (basic R-U-R / R-T-R / R-S-R + 커스텀 R-Burst-R / R-SemSim-R / R-Behavior-R)
    │
    ▼
ChebConv K=3 branch × 6 (per relation, hidden=128, num_layers=3, residual Skip)
    │
    ▼   stack → (N, 6, 128)
Gated Relation Fusion (per-node softmax α over 6 branches)
    │
    ▼   (N, 128)
Projection → Main Classifier            → main_logit
            + 6 × Auxiliary heads        → aux_logits per relation

Loss = FocalLoss(α=0.75, γ=2.0) + 0.3 × mean_r BCE(aux_r, y)
threshold @ valid PR-curve F1-max → Test set 1회 평가
```

| 구성 | 값 |
|---|---|
| 모델 클래스 | `src/models/cage_rf_gnn_cheb.py` (Skip + Gating + Aux 통합) |
| Offline filter | `src/filtering/care_neighbor_filter.py` |
| Trainable projection wrapper | `src/models/text_projection_wrapper.py` (proj 변형 전용) |
| Config | `configs/cage_rf_skip_care.yaml` |
| 사용 relation | 6개 모두 |
| 활성 모듈 | Skip + Gated Fusion + Auxiliary Loss + CARE filter 모두 ON |

---

## 6. 디렉토리 구조

```text
.
├── src/
│   ├── preprocessing/   load_yelpzip / label_convert / sampling / feature_engineering
│   ├── sampling/        cascade_pipeline / hdbscan_stratified_split / grouped_stratified_split
│   ├── graph/           build_{rur,rtr,rsr,burst,semsim,behavior,relations,relation_quality}
│   ├── filtering/       care_neighbor_filter
│   ├── models/          baseline_{mlp,gcn,gat,graphsage,cheb,tag} / cage_rf_gnn_cheb /
│   │                    skip_cheb_branch / gated_relation_fusion / cage_carerf_gnn /
│   │                    text_projection_wrapper / losses
│   ├── training/        train / evaluate / threshold /
│   │                    lgbm_stacking / stacked_ensemble / aggregate_final
│   └── utils/           seed / io / metrics
├── configs/             default, v8_skip, v9_twostage, cage_rf_skip_care,
│                        cage_carerf{,_lean,_lean_5}, ablation_no_{skip,gating,aux,custom}
├── amazon/              참고 데이터셋: Amazon — 7모델 × 5 seeds
├── yelchi/              참고 데이터셋: YelpChi — 7모델 × 5 seeds
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── run_all_models.py            YelpZip 단일 seed launcher
├── 5x_run_all_models.py         YelpZip 15 × 5 = 75회 launcher
├── run_proj_experiments.py      Linear projection 변형 2 × 5 = 10회 launcher
├── run_all_amazon.py            Amazon 7모델 launcher
└── run_all_yelchi.py            YelpChi 7모델 launcher
```

`.gitignore` 제외: `data/`, `outputs/`, 외부 reference 라이브러리(`CARE-GNN/`, `PC-GNN/`, `DGFraud/`).

---

## 7. 데이터 핵심 수치

- 원본: YelpZip 608,458 리뷰 / 5,044 상품 / 260,239 사용자 / fraud_ratio 13.22%
- 샘플 (결정론적 이분 탐색): **47,125 nodes** (Train 30,160 / Valid 7,540 / Test 9,425)
- 샘플 fraud_ratio: 11.16% (train/valid/test 동일, stratified)
- 임계값: `head_pu = 6`, `head_m = 1` → 자연스럽게 ≤ 50,000
- Node feature: 140D = 128 (텍스트 → SVD) + 12 (numeric), 모든 fit은 train-only

### 6 Relations 도메인 가설

| Relation | 도메인 의미 | 예상 신호 강도 |
|---|---|---|
| R-Burst-R | 단기 평판 폭격 (7일/±1점) | 최강 |
| R-T-R | 동일 상품 동월 시간 집중 | 강 |
| R-U-R | 동일 사용자 반복 작성 | 중 |
| R-S-R | 동일 상품 동일 별점 | 중하 |
| R-SemSim-R | 텍스트 코사인 top-5 | 중하 |
| R-Behavior-R | user 행동 벡터 cosine top-5 | 약 |

---

## 8. 예선 규정 준수 요약

- Node = Review 유지 (분류 타깃 = 리뷰, 단일 노드 타입 homogeneous 그래프)
- YelpZip 원본 → 결정론적 이분 탐색 임계값 샘플링 → 47,125 노드 → 64/16/20 stratified split (`random_state=42`)
- **무작위 추출 0건** (`df.sample` / `np.random.choice` / `rng.choice` / `np.random.default_rng` 모두 `src/` 에 없음)
- 라벨 변환 `-1→1, 1→0` (`src/preprocessing/label_convert.py`)
- 기본 relation 3 + 커스텀 relation 3 = 6 relations, 모두 결정론적 top-k 또는 threshold 적용
- TF-IDF/SVD/Scaler **train-only fit** (leakage-safe), SBERT는 frozen
- relation quality 계산 시 train labels only
- threshold 는 valid PR-curve에서 결정, **test set 은 1회만 평가**
- PR-AUC / Macro F1 / G-Mean / ROC-AUC / Precision / Recall / Accuracy 모두 저장
- 학회 표준 **multi-seed (5개) 평균 ± std** 보고

---

## 9. 도메인 일반화 — Cross-dataset 검증 (참고)

동일 backbone(CAGE-RF + CARE)을 다른 fraud 도메인에 이식한 결과 (5 seeds 평균):

| 데이터셋 | 제안 모델 | 최고 baseline | PR-AUC (제안) | PR-AUC Δ |
|---|---|---|:---:|:---:|
| YelpZip (메인) | CAGE-RF + CARE | ChebConv | 0.4447 | +0.1696 |
| Amazon | CAGE-CareRF | MLP | 0.8162 | -0.0041 |
| YelpChi | CAGE-CareRF | GraphSAGE | 0.7114 | +0.0936 |

큰 재설계 없이 다른 fraud 도메인으로 이식 가능. 자세한 표는 `amazon/` 및 `yelchi/` 폴더 참조.

---

## 10. 본선 확장 — Behavioral Stacking & Meta-Ensemble

본 섹션은 본선 단계에서 추가된 *후속 실험* 으로, FINAL GNN 의 출력을 **트리 기반 학습기 3종 (LightGBM · XGBoost · CatBoost) 의 행동 피처 모델** 과 결합하여 한 단계 더 성능 마진을 짜내는 stacking 파이프라인이다. 위 1~5 섹션의 공식 FINAL 결과 (`PR-AUC 0.4439 ± 0.0152`) 는 변경되지 않는다.

### 10-1. 누수 차단 — 모든 aggregate 는 train 만으로 계산

```python
# src/training/lgbm_stacking.py  (build_features)
train_df = df.loc[train_mask].copy()
user_agg = _user_aggregates(train_df)   # train 만 사용
prod_agg = _prod_aggregates(train_df)   # train 만 사용
# valid/test 행은 user_id / prod_id 로 *train aggregate* 만 조회
# train 에 없던 user/prod 는 NaN → 0 + unseen_in_train 별도 flag
```

YelpZip fraud 탐지에서 흔히 보이는 `user_fraud_rate` / `prod_fraud_rate` 류의 **전역 집계 leakage** 를 구조적으로 차단. 본 파이프라인 결과는 hold-out test 에 대해 재현 가능한 수치임을 보장한다.

### 10-2. 행동 피처 구성 (총 32개)

| 그룹 | 피처 |
|---|---|
| **self** | rating, rating_is_extreme, text_len, n_words, n_exclaim, n_question, caps_ratio, avg_word_len, ttr, dow, month, year |
| **user_agg** (train only) | n_reviews, rating_{mean,std}, rating_extreme_ratio, text_len_{mean,std}, unique_prods, review_per_prod, unseen_in_train |
| **prod_agg** (train only) | n_reviews, rating_{mean,std,skew}, extreme_ratio, unique_users, review_per_user, unseen_in_train |
| **burst** | 같은 prod 의 ±7일 윈도우 안의 리뷰 수 (binary-search O(N log N)) |
| **relative** | rating − user_mean, rating − prod_mean, text_len − user_mean |

### 10-3. 아키텍처 — 2-Level Stacking

```text
                   ┌─────────────────────────┐
                   │  Behavioral Features    │
                   │  (32 cols, train-only)  │
                   └────────────┬────────────┘
                                │
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
   ┌────────┐     ┌──────────┐    ┌──────────┐    ┌────────┐
   │ LGBM   │     │ XGBoost  │    │ CatBoost │    │  GNN   │ ← saved npy
   │ Level1 │     │ Level1   │    │ Level1   │    │ FINAL  │
   └───┬────┘     └────┬─────┘    └────┬─────┘    └───┬────┘
       │               │               │               │
       └───────────────┴────┬──────────┴───────────────┘
                            ▼
                  ┌──────────────────────┐
                  │  Level-2 Meta-Learner│
                  │  Logistic Regression │  ← valid 에서만 학습
                  │   (4 prob inputs)    │
                  └──────────┬───────────┘
                             ▼
                       Test PR-AUC / Macro-F1
```

- **Level 1**: 네 모델 각각 *독립* 학습. LGBM/XGB/Cat 은 같은 32D 행동 피처 사용, GNN 은 이미 학습된 5-seed 결과의 `probs_*_seed{N}.npy` 를 로드.
- **Level 2**: Logistic Regression 메타 학습기가 `[lgbm_v, xgb_v, cat_v, gnn_v]` 4개 확률을 입력으로 받아 *valid 에서만* 학습 (test 한 번도 미접촉). 가중치는 base learner 별 신뢰도를 *피처 별로* 학습하므로 단순 weighted average 보다 정교.
- **Alternate**: `--meta xgb` 옵션으로 얕은 XGBoost (max_depth=3) 메타 학습기 대안 제공.

### 10-4. 실행

```bash
# 의존성 (한 번만)
pip install lightgbm xgboost catboost hdbscan

# (선행) FINAL GNN 5-seed 학습 — probs_*_seed{N}.npy 자동 저장
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42   # 42, 123, 2024, 7, 1234

# 단순 weighted blend (LGBM only + GNN, 5 seeds)
python -m src.training.lgbm_stacking --seed 42 \
    --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy \
    --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy

# Level-2 meta-ensemble (LGBM + XGB + Cat + GNN, 5 seeds)
python -m src.training.stacked_ensemble --seed 42 \
    --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy \
    --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy

# 5-seed 통합 리포트 (GNN / LGBM-only / Blend / Meta 한 표)
python -m src.training.aggregate_final
```

### 10-5. 보조 — Cascade Sampling Pipeline (`src/sampling/`)

본선 단계 부수 산출물. 기존 *fraud-blind* hybrid sampling 대비 **자연스럽게 fraud 비율을 25%+ 로 끌어올리는** 3-stage 캐스케이드를 제공한다 (`src/preprocessing/sampling.py` CONFIG 의 `sampling_strategy` 로 선택):

1. **`group_dense`** — fraud-density × log-activity 점수 상위 (user / prod / month) 그룹 선택
2. **`cascade`** — (1) → HDBSCAN semantic 필터링 → R-U-R / burst window 기반 normal reseed (그래프 연결성 회복)
3. 보조 분할 도구: `hdbscan_stratified_split` (의미 군집 + 라벨 동시 stratify), `grouped_stratified_split` (StratifiedGroupKFold, shuffle / time_ordered 모드)

공식 FINAL 결과 (`PR-AUC 0.4439`) 는 기존 hybrid sampling 기준이며, 본 캐스케이드는 본선 분석·실험용으로 분리해 둔다.

---

## 11. 라이선스 / 팀

ITDA Team UnivConcat — 2026 ITDA Networking Day 학술제 본선 제출용.
외부 reference 라이브러리(CARE-GNN, PC-GNN, DGFraud)는 본 repo에 포함되지 않으며 각 원본 라이선스에 따른다.
