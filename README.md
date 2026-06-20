<div align="center">

# 🕵️ CAGE-CareRF GNN

### Multi-Relation GNN으로 조직적 사기 리뷰 네트워크를 잡아냅니다

**Camouflage-Aware Filtering for Organized Abusing Network Detection** · YelpZip
ITDA Networking Day 2026 학술제 본선 — **Team UnivConcat** (김재현 · 백수연 · 홍예진)

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.7-3C2179?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-stacking-9ACD32?style=for-the-badge)

<br>

<h4>
가짜 리뷰는 혼자 움직이지 않습니다 — 같은 IP·시간대·문체로 <b>떼지어</b> 움직이죠.<br>
이 repo는 리뷰를 <b>6가지 관계 그래프</b>로 엮고, 관계마다 다른 신호를 노드별 게이트로 융합해<br>
<b>조직적 어뷰징(organized abusing)</b>을 탐지하는 <b>공식 제출 메인 코드 베이스</b>입니다.
<br><br>
표준 GNN baseline 대비 <b>PR-AUC +7.4%</b> · 학회 표준 5-seed 평균±표준편차로 검증.
</h4>

</div>

> [!NOTE]
> 본 README의 모든 성능 수치는 **2026-06 재집계 결과**(`multi_seed_summary.json` 17모델 · `summary.json` 인코더 study)이며, **5개 random seed(7·42·123·2024·1234)** 의 평균 ± 표준편차입니다. 정리·시각화는 [`results_summary.html`](results_summary.html)에서 표·바 차트로 볼 수 있습니다.

<br>

## 0. 공식 제출 자료

| 항목 | 위치 |
|---|---|
| **메인 코드 (본 repo)** | https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE |
| **시각화 결과물 (Streamlit)** | https://github.com/ITDA-Team-C/Streamlit_CAGE-RF-with-CARE |
| **분석 보고서** | `연합1조_분석보고서.pdf` |
| **시각화 요약 보고서** | `연합1조_시각화요약보고서.pdf` |

### 인코더 ablation 단계별 reference repo

| 단계 | 변형 | Repo |
|:---:|---|---|
| 1단계 | frozen-SBERT 단독 | [CAGE-CareRF_With_Bert_Frozen](https://github.com/ITDA-Team-C/CAGE-CareRF_With_Bert_Frozen) |
| 2단계 | TF-IDF ⊕ SBERT concat | [CAGE-CareRF-CONCAT-BERT_WITH_TF_IDF](https://github.com/ITDA-Team-C/CAGE-CareRF-CONCAT-BERT_WITH_TF_IDF) |
| 시각화 | 실시간 threshold·네트워크 탐색 | [Streamlit_CAGE-RF-with-CARE](https://github.com/ITDA-Team-C/Streamlit_CAGE-RF-with-CARE) |

<br>

## 1. 공식 최종 시스템

| 구성요소 | 채택 |
|---|---|
| **Backbone (모델)** | **CAGE-RF + CARE** (Skip + Gated Fusion + Aux Loss + offline CARE filter) |
| **Encoder (텍스트 → 노드 feature)** | **concat** (TF-IDF SVD-128 ⊕ frozen-SBERT SVD-128 → 모달리티별 train-only z-score → 공동 train-only SVD-128) |
| **Config** | `configs/cage_rf_skip_care.yaml` (집계 ID `cage_rf_gnn_cheb_cage_rf_skip_care`) |

### 5 seeds 평균 성능 (Test set, 2026-06 재집계)

<div align="center">

| 지표 | 최종 모델 | 최고 baseline (GAT) | MLP (그래프 미사용) |
|:---|:---:|:---:|:---:|
| **PR-AUC** | **0.789 ± 0.003** | 0.734 | 0.633 |
| **Macro-F1** | **0.792 ± 0.004** | 0.759 | 0.714 |
| **G-Mean** | **0.776 ± 0.005** | 0.734 | 0.693 |
| ROC-AUC | ~0.885–0.890 *(인코더 study 기준)* | — | — |

</div>

표준 GNN baseline(GAT) 대비 **PR-AUC +0.054 (상대 +7.4%)**, 그래프 구조를 쓰지 않는 MLP 대비 **+24.7%**. 최종 모델은 최상위권과 **통계적으로 동급**(표준편차 내)이면서 세 지표 모두에서 균형이 가장 좋은 설정이다.

> [!TIP]
> 전체 17개 모델(제안 변형·ablation·baseline) × 5-seed 비교는 [`results_summary.html`](results_summary.html)에서 정렬·바 차트로 한눈에 볼 수 있습니다.

<br>

## 2. 핵심 결과 — 17모델 × 5 seeds + 인코더 ablation

### 2-1. 주요 모델 비교 (PR-AUC 내림차순, 발췌)

<div align="center">

| Model | PR-AUC | Macro-F1 | G-Mean |
|---|:---:|:---:|:---:|
| w/o Custom relation *(ablation)* | 0.792 ± 0.002 | 0.797 ± 0.003 | 0.777 ± 0.007 |
| w/o Gating *(ablation)* | 0.790 ± 0.004 | 0.790 ± 0.003 | 0.781 ± 0.015 |
| **★ CAGE-RF + CARE (FINAL)** | **0.789 ± 0.003** | **0.792 ± 0.004** | **0.776 ± 0.005** |
| CAGE-CareRF Lean-4 | 0.788 ± 0.005 | 0.794 ± 0.003 | 0.774 ± 0.009 |
| w/o Skip *(ablation)* | 0.782 ± 0.002 | 0.792 ± 0.003 | 0.782 ± 0.002 |
| **w/o Aux Loss** *(ablation)* | **0.739 ± 0.006** | 0.765 ± 0.007 | 0.757 ± 0.016 |
| GAT *(baseline 최고)* | 0.734 ± 0.003 | 0.759 ± 0.003 | 0.734 ± 0.005 |
| GCN *(baseline)* | 0.726 ± 0.004 | 0.758 ± 0.002 | 0.730 ± 0.008 |
| ChebConv *(baseline)* | 0.720 ± 0.003 | 0.752 ± 0.002 | 0.734 ± 0.016 |
| MLP *(no graph)* | 0.633 ± 0.002 | 0.714 ± 0.002 | 0.693 ± 0.007 |

</div>

> [!IMPORTANT]
> **per-relation Auxiliary Loss가 단연 핵심입니다.** 제거 시 PR-AUC가 **−0.05 (0.789 → 0.739)**, Macro-F1이 0.028 하락 — 단일 구성요소 제거 중 가장 큰 손실입니다. 반면 Skip은 소폭(−0.007), **Gating·Custom relation 제거는 표준편차(±0.002~0.004) 범위 내 변화(≈0)** 로, 이 데이터셋·설정에서는 기여가 통계적으로 유의하지 않습니다. (측정값 그대로 정직 보고)

### 2-2. 인코더 projection ablation (5 seeds)

학습 가능 Linear projection 변형의 비교 (`summary.json`).

<div align="center">

| 인코더 (projection) | PR-AUC | Macro-F1 | G-Mean | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|
| **SBERT** projection | **0.766** | **0.781** | 0.764 | **0.890** |
| TF-IDF ⊕ SBERT **concat** projection | 0.758 | 0.780 | **0.767** | 0.885 |

</div>

- 두 projection은 사실상 **동급**(차이 대부분 1~2 표준편차 이내). SBERT projection이 PR-AUC·ROC-AUC·Macro-F1 근소 우위, concat projection이 G-Mean 근소 우위.
- 두 학습형 projection 모두 **공식 최종 시스템의 SVD 기반 concat 인코더(PR-AUC 0.789)에는 미치지 못함** → "적은 라벨 환경에서 학습형 projection이 단순 SVD를 능가하지 못한다"는 결론을 재확인.

<br>

## 3. 환경

| 항목 | 값 |
|---|---|
| Python | 3.11+ 권장 (검증 3.13) |
| OS | Linux / macOS / Windows |
| GPU | 강력 추천 (CPU 학습은 매우 느림) |
| 핵심 의존성 | PyTorch · PyG 2.7 · scikit-learn · sentence-transformers · pandas · LightGBM |

```bash
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

<br>

## 4. 빠른 실행 — 공식 최종 시스템 재현

> [!IMPORTANT]
> 데이터(~410MB)는 용량 문제로 repo에 포함되지 않습니다. YelpZip 원본을 `data/raw/`에 직접 배치하세요.

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

# 4) 학회 표준 multi-seed (다중 모델 × 5 seeds) — 권장
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

```bash
# 학습 가능 Linear projection 두 변형 × 5 seeds (summary.json 생성)
python run_proj_experiments.py
```

<br>

## 5. FINAL 모델 구조 — CAGE-RF + CARE

```
Reviews (N = 47,125, F = 140 = text 128 + numeric 12)
    │
    ▼   [offline 1회] CARE neighbor filter (feature cosine top-k=10, label-free · leakage-safe)
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
| Offline filter | `src/filtering/care_neighbor_filter.py` (feature cosine top-k=10) |
| Trainable projection wrapper | `src/models/text_projection_wrapper.py` (proj 변형 전용) |
| Config | `configs/cage_rf_skip_care.yaml` |
| 활성 모듈 | Skip + Gated Fusion + Auxiliary Loss + CARE filter 모두 ON |

<br>

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
│                        cage_carerf{,_lean,_lean_5,_lean_6}, ablation_no_{skip,gating,aux,custom,care}
├── amazon/  yelchi/     cross-dataset 검증용 (Amazon · YelpChi)
├── run_all_models.py            YelpZip 단일 seed launcher
├── 5x_run_all_models.py         YelpZip 다중 모델 × 5 seeds launcher
├── run_baselines.py             baseline(GCN/GAT/…) 일괄 실행
├── run_proj_experiments.py      Linear projection 변형 × 5 seeds (summary.json)
├── run_all_amazon.py            Amazon launcher
└── run_all_yelchi.py            YelpChi launcher
```

`.gitignore` 제외: `data/`, `outputs/`, 외부 reference 라이브러리(`CARE-GNN/`, `PC-GNN/`, `DGFraud/`).

<br>

## 7. 데이터 핵심 수치

- 원본: YelpZip 608,458 리뷰 / 5,044 상품 / 260,239 사용자 / fraud_ratio 13.22%
- 샘플 (결정론적 이분 탐색): **47,125 nodes** (Train 30,160 / Valid 7,540 / Test 9,425)
- 샘플 fraud_ratio: **11.16%** (train/valid/test 동일, stratified)
- Node feature: **140D = 128 (텍스트 → SVD) + 12 (numeric)**, 모든 fit은 train-only

### 6 Relations 도메인 가설

| Relation | 도메인 의미 | 예상 신호 강도 |
|---|---|---|
| R-Burst-R | 단기 평판 폭격 (7일/±1점) | 최강 |
| R-T-R | 동일 상품 동월 시간 집중 | 강 |
| R-U-R | 동일 사용자 반복 작성 | 중 |
| R-S-R | 동일 상품 동일 별점 | 중하 |
| R-SemSim-R | 텍스트 코사인 top-5 | 중하 |
| R-Behavior-R | user 행동 벡터 cosine top-5 | 약 |

<br>

## 8. 예선 규정 준수 요약

- Node = Review 유지 (분류 타깃 = 리뷰, 단일 노드 타입 homogeneous 그래프)
- YelpZip 원본 → 결정론적 이분 탐색 임계값 샘플링 → 47,125 노드 → 64/16/20 stratified split (`random_state=42`)
- **무작위 추출 0건** (`df.sample` / `np.random.choice` / `rng.choice` 모두 `src/`에 없음)
- 라벨 변환 `-1→1, 1→0` (`src/preprocessing/label_convert.py`)
- 기본 relation 3 + 커스텀 relation 3 = 6 relations, 모두 결정론적 top-k 또는 threshold 적용
- TF-IDF/SVD/Scaler **train-only fit** (leakage-safe), SBERT는 frozen
- threshold는 valid PR-curve에서 결정, **test set은 1회만 평가**
- PR-AUC / Macro-F1 / G-Mean / ROC-AUC / Precision / Recall / Accuracy 모두 저장
- 학회 표준 **multi-seed (5개) 평균 ± std** 보고

<br>

## 9. 도메인 일반화 — Cross-dataset 검증 (참고)

동일 backbone(CAGE-RF + CARE)을 **Amazon · YelpChi** 등 다른 fraud 도메인에 큰 재설계 없이 이식할 수 있도록 `amazon/` · `yelchi/` 파이프라인을 함께 제공합니다. 각 데이터셋의 최신 재집계 수치는 해당 폴더의 결과 요약을 참조하세요. *(YelpZip 메인 결과는 위 1~2장 참조.)*

<br>

## 10. 본선 확장 — Behavioral Stacking & Meta-Ensemble

FINAL GNN의 출력을 **트리 기반 학습기 3종(LightGBM · XGBoost · CatBoost)의 행동 피처 모델**과 결합해 한 단계 더 마진을 짜내는 stacking 파이프라인입니다. *(위 1~2장의 공식 FINAL 결과는 본 확장과 무관하게 그대로 유지됩니다.)*

### 10-1. 누수 차단 — 모든 aggregate는 train만으로 계산

```python
# src/training/lgbm_stacking.py  (build_features)
train_df = df.loc[train_mask].copy()
user_agg = _user_aggregates(train_df)   # train 만 사용
prod_agg = _prod_aggregates(train_df)   # train 만 사용
# valid/test 행은 train aggregate 만 조회, unseen 은 NaN→0 + flag
```

`user_fraud_rate` / `prod_fraud_rate` 류의 **전역 집계 leakage**를 구조적으로 차단 → hold-out test에 대해 재현 가능한 수치를 보장.

### 10-2. 아키텍처 — 2-Level Stacking

```text
   ┌────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐
   │ LGBM   │  │ XGBoost  │  │ CatBoost │  │  GNN   │ ← saved npy
   │ Level1 │  │ Level1   │  │ Level1   │  │ FINAL  │
   └───┬────┘  └────┬─────┘  └────┬─────┘  └───┬────┘
       └────────────┴──────┬──────┴────────────┘
                           ▼
              ┌──────────────────────┐
              │  Level-2 Meta-Learner│  ← valid 에서만 학습
              │  Logistic Regression │     (4 prob inputs, test 미접촉)
              └──────────┬───────────┘
                         ▼  Test PR-AUC / Macro-F1
```

- **Level 1**: 네 모델 각각 독립 학습. 트리 3종은 행동 피처 32D 사용, GNN은 학습된 5-seed `probs_*_seed{N}.npy`를 로드.
- **Level 2**: Logistic Regression이 4개 확률을 입력으로 valid에서만 학습 (단순 평균보다 정교). `--meta xgb`로 얕은 XGBoost 대안 제공.

### 10-3. 실행

```bash
pip install lightgbm xgboost catboost hdbscan

# (선행) FINAL GNN 5-seed 학습 — probs_*_seed{N}.npy 자동 저장
python -m src.training.train --model cage_rf_gnn_cheb \
    --config configs/cage_rf_skip_care.yaml --seed 42   # 42,123,2024,7,1234

# Level-2 meta-ensemble (LGBM + XGB + Cat + GNN, 5 seeds)
python -m src.training.stacked_ensemble --seed 42 \
    --gnn-probs-valid outputs/benchmark/CHEB/probs_valid_seed42.npy \
    --gnn-probs-test  outputs/benchmark/CHEB/probs_test_seed42.npy

# 5-seed 통합 리포트 (GNN / LGBM-only / Blend / Meta 한 표)
python -m src.training.aggregate_final
```

### 10-4. 보조 — Cascade Sampling Pipeline (`src/sampling/`)

기존 fraud-blind hybrid sampling 대비 **자연스럽게 fraud 비율을 25%+로 끌어올리는** 3-stage 캐스케이드(`src/preprocessing/sampling.py`의 `sampling_strategy`로 선택): `group_dense`(fraud-density × log-activity 상위 그룹) → `cascade`(HDBSCAN semantic 필터링 + R-U-R/burst reseed) → 보조 분할 도구(`hdbscan_stratified_split`, `grouped_stratified_split`).

<br>

---

<div align="center">

<i>버그 제보·아이디어 환영합니다.</i><br>
<i>ITDA Team UnivConcat — 2026 ITDA Networking Day 학술제 본선 제출용</i><br>
<i>Contact : seankim0824@gmail.com</i>

</div>
