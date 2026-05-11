# 01. 코드 구조

YelpZip 기반 조직적 어뷰징 네트워크 탐지 — **CAGE-CareRF GNN** 예선 코드의 전체 디렉토리/파일 맵.

---

## 1. 최상위 디렉토리

```text
ITDA_TEAM_C_FINAL/
├── src/                      # 모든 소스 코드 (패키지)
├── configs/                  # 학습/실험 YAML config 10개
├── data/
│   ├── raw/yelp_zip.csv      # YelpZip 원본 (608,458 reviews)
│   ├── interim/              # 중간 산출물 (raw_data.csv, labeled_data.csv, raw_eda.txt)
│   └── processed/            # 최종 산출물 (features.npy, edge_index_*.pt 등)
├── outputs/
│   ├── metrics/              # *.json (relation_quality, filter_log, model metrics)
│   ├── checkpoints/          # 학습된 모델 .pt
│   ├── figures/              # 보고서 삽입용 PNG
│   ├── reports/              # 보고서 표 .csv
│   └── cage_rf_gnn/          # 학습별 metrics_*.json + report_*.html
└── docs/                     # 본 문서들
```

---

## 2. `src/` 패키지 구조

```text
src/
├── __init__.py
├── preprocessing/
│   ├── load_yelpzip.py          # P  raw CSV → interim/raw_data.csv
│   ├── label_convert.py         # P  -1→1, 1→0
│   ├── sampling.py              # P  Graph-Signal Preserving Hybrid Dense Sampling
│   └── feature_engineering.py   # N  TF-IDF/SVD/Scaler train-only fit, 140D feature
├── graph/
│   ├── build_rur.py             # P  R-U-R (같은 user)
│   ├── build_rtr.py             # P  R-T-R (같은 prod + 같은 month)
│   ├── build_rsr.py             # P  R-S-R (같은 prod + 같은 rating)
│   ├── build_burst.py           # P  R-Burst-R (|Δdate|≤7d & |Δrating|≤1)
│   ├── build_semsim.py          # P  R-SemSim-R (TF-IDF cosine top-k, 같은 prod 내)
│   ├── build_behavior.py        # N* user-level cosine → review-pair edge (메모리 안전 재작성)
│   ├── build_relations.py       # P  6개 호출 → edge_index_dict.pt 저장
│   └── relation_quality.py      # N  edge_count, avg_degree, fraud_*ratio, fraud_edge_lift
├── filtering/
│   └── care_neighbor_filter.py  # N  feature cosine top-k, label-free
├── sampling/
│   └── (empty)                  #    PC-GNN sampler는 예선 메인에서 제외
├── models/
│   ├── baseline_mlp.py          # P  MLP
│   ├── baseline_gcn.py          # N* union of 6 relations 사용으로 패치
│   ├── baseline_gat.py          # N* union 패치
│   ├── baseline_graphsage.py    # N* union 패치
│   ├── cage_rf_gnn.py           # P  (entry, no-op import shim)
│   ├── cage_rf_gnn_cheb.py      # P  Skip(v8) + Two-Stage(v9) + Gating + Aux 통합
│   ├── skip_cheb_branch.py      # N  분리 모듈 (CAGE-CareRF에서 사용)
│   ├── gated_relation_fusion.py # N  분리 모듈
│   ├── cage_carerf_gnn.py       # N  최종 제안 모델
│   └── losses.py                # P  WeightedBCE / Focal / Auxiliary
├── training/
│   ├── train.py                 # P+ create_model에 cage_carerf_gnn 추가, 오프라인 CARE 적용
│   ├── evaluate.py              # P  threshold @ valid 로드 → test 1회 평가
│   └── threshold.py             # P  Valid PR-curve에서 최적 threshold 결정
├── utils/
│   ├── seed.py                  # P  set_seed(42)
│   ├── io.py                    # P  load_config / save_object / save_json
│   ├── metrics.py               # P  PR-AUC / Macro-F1 / ROC-AUC / Precision / Recall / Accuracy
│   └── html_report.py           # P  학습 결과 HTML 리포트
└── reports/
    └── (예정) build_tables.py   #    보고서 csv 자동 생성 (Phase 8)
```

표기: **P** = ITDA_TEAM_C에서 포팅, **N** = 본 작업에서 신규, **N*** = 포팅 후 본 작업에서 재작성/보강, **P+** = 포팅 후 본 작업에서 옵션 추가.

---

## 3. `configs/` (10개 YAML)

| 파일 | 용도 |
|---|---|
| `default.yaml` | 기본 설정, CAGE-RF Base (no skip) 학습 |
| `v8_skip.yaml` | CAGE-RF Skip (use_skip_connection=true) |
| `v9_twostage.yaml` | CAGE-RF Refine (use_two_stage=true, focal+aux) |
| `cage_rf_skip_care.yaml` | v8 + CARE filter offline (Plan §7.1 #6) |
| **`cage_carerf.yaml`** | **최종 제안 모델 CAGE-CareRF GNN** |
| `ablation_no_care.yaml` | CARE filter 끔 |
| `ablation_no_skip.yaml` | use_skip=false |
| `ablation_no_gating.yaml` | use_gating=false (mean fusion) |
| `ablation_no_aux.yaml` | use_aux_loss=false, aux_weight=0 |
| `ablation_no_custom.yaml` | active_relations=["rur","rtr","rsr"] (기본 3개만) |

모든 cage_carerf 계열 yaml의 **`sampler.enabled: false`** + note: PC-GNN sampler는 메인에서 제외됨을 명시.

---

## 4. 데이터 흐름

```text
data/raw/yelp_zip.csv                            (608,458 reviews, 410MB)
        │
        │ load_yelpzip.py
        ▼
data/interim/raw_data.csv  +  raw_eda.txt
        │
        │ label_convert.py   (-1→1, 1→0)
        ▼
data/interim/labeled_data.csv                    (label ∈ {0,1})
        │
        │ sampling.py    (product/user/time hybrid dense)
        ▼
data/processed/sampled_reviews.csv               (50,000 nodes, split 컬럼 포함)
data/processed/sampling_stats.txt
        │
        │ feature_engineering.py  (TRAIN-only fit)
        ▼
data/processed/features.npy                      (50000, 140)
data/processed/node_samples.csv                  (sampled_reviews + 동일 split)
data/processed/feature_meta.json                 (fit_scope=train_only 명시)
        │
        │ build_relations.py
        ▼
data/processed/edge_index_{rur,rtr,rsr,burst,semsim,behavior}.pt
data/processed/edge_index_dict.pt                (6개 묶음)
data/processed/graph_meta.json                   (relation별 edge_count)
        │
        │ relation_quality.py   (train labels only)
        ▼
outputs/metrics/relation_quality.json
outputs/reports/relation_quality.csv             (보고서용 표)
        │
        │ train.py  (CARE filter offline → 모델 학습 → threshold@valid → test 평가)
        ▼
outputs/metrics/filter_log.json
outputs/cage_rf_gnn/metrics_*.json
outputs/cage_rf_gnn/best_model_*.pt
outputs/cage_rf_gnn/report_*.html
```

---

## 5. 모듈 의존성 다이어그램

```text
                   ┌───────────────────────────────┐
                   │  src.utils.{seed, io, metrics}│
                   └───────────────────────────────┘
                       ▲                    ▲
                       │                    │
   ┌───────────────────┴───────────┐        │
   │ preprocessing/{load,label,    │        │
   │   sampling,feature_eng}.py    │        │
   └─────────────────┬─────────────┘        │
                     ▼                      │
              data/processed/               │
                     │                      │
                     ▼                      │
   ┌───────────────────────────────┐        │
   │ graph/build_*.py              │        │
   │ graph/build_relations.py      │        │
   │ graph/relation_quality.py ────┼────────┘
   └─────────────────┬─────────────┘
                     ▼
              edge_index_dict.pt
                     │
                     ▼
    ┌───────────────────────────────────────────────────────┐
    │ training/train.py                                      │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
    │  │ create_model │→ │ CARE filter  │→ │ train loop   │ │
    │  └──────────────┘  │  (filtering/)│  │              │ │
    │                    └──────────────┘  └──────────────┘ │
    │           ↓ (모델 종류에 따라)                          │
    │  ┌──────────────────────────────────────────────────┐ │
    │  │ baselines/{gcn,gat,sage,mlp}                     │ │
    │  │ cage_rf_gnn_cheb (Skip/Two-Stage/Gating/Aux)     │ │
    │  │ cage_carerf_gnn:                                 │ │
    │  │   ├─ skip_cheb_branch.py                         │ │
    │  │   ├─ gated_relation_fusion.py                    │ │
    │  │   └─ care_neighbor_filter.py (inline 또는 offline)│ │
    │  └──────────────────────────────────────────────────┘ │
    │           ↓                                             │
    │  ┌────────────────┐   ┌────────────────┐              │
    │  │ models/losses  │   │ utils/metrics  │              │
    │  │ Focal+Aux      │   │ PR-AUC/MacroF1 │              │
    │  └────────────────┘   └────────────────┘              │
    │           ↓                                             │
    │  threshold @ valid PR-curve  →  test 1회 평가          │
    └────────────────────────────┬──────────────────────────┘
                                 ▼
                         outputs/cage_rf_gnn/metrics_*.json
```

---

## 6. 신규 작성 모듈 요약

| 파일 | 라인 수 | 핵심 책임 |
|---|---|---|
| `src/preprocessing/feature_engineering.py` | ~140 | TF-IDF/SVD/Scaler를 train_mask에서만 fit, valid/test는 transform-only |
| `src/graph/build_behavior.py` | ~95 | user 단위 cosine top-k (chunked, 메모리 안전) → user pair → review pair 확장 |
| `src/graph/relation_quality.py` | ~125 | relation별 quality (train labels only). fraud_edge_lift = ff_ratio / fraud_ratio² |
| `src/filtering/care_neighbor_filter.py` | ~110 | feature cosine top-k 필터링. before/after edge count 로그 |
| `src/models/skip_cheb_branch.py` | ~40 | CHEB + residual skip per layer |
| `src/models/gated_relation_fusion.py` | ~45 | softmax α per node, fused = Σ αᵢ·bᵢ + entropy regularizer |
| `src/models/cage_carerf_gnn.py` | ~150 | 6 branch + gating + main/aux classifier + optional inline CARE |

---

## 7. 산출물 위치 요약

| 파일 | 위치 | 생성 시점 |
|---|---|---|
| EDA | `data/interim/raw_eda.txt` | load_yelpzip |
| 샘플 통계 | `data/processed/sampling_stats.txt` | sampling |
| Feature 메타 | `data/processed/feature_meta.json` | feature_engineering (`fit_scope=train_only` 박제) |
| Graph 메타 | `data/processed/graph_meta.json` | build_relations |
| Relation Quality | `outputs/metrics/relation_quality.json` + `outputs/reports/relation_quality.csv` | relation_quality |
| CARE filter 로그 | `outputs/metrics/filter_log.json` | train (CARE 활성 시) |
| 모델 metrics | `outputs/cage_rf_gnn/metrics_<model>.json` | train |
| 모델 가중치 | `outputs/cage_rf_gnn/best_model_<model>.pt` | train |
| HTML 리포트 | `outputs/cage_rf_gnn/report_<model>.html` | train |

---

## 8. 외부 의존성

- Python 3.11+
- `torch` 2.x (+ CUDA 권장)
- `torch_geometric` (`ChebConv`, `GCNConv`, `GATConv`, `SAGEConv`)
- `scikit-learn` 1.x (`TfidfVectorizer`, `TruncatedSVD`, `StandardScaler`, metrics)
- `pandas`, `numpy`, `pyyaml`

`requirements.txt`는 옮길 때 ITDA_TEAM_C/requirements.txt 참조 또는 동일하게 구성.
