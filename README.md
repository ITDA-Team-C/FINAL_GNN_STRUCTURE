# CAGE-CareRF GNN

> **Camouflage-Aware Gated Edge Relation-Fusion GNN**
> YelpZip 기반 **그래프신경망 기반 조직적 어뷰징 네트워크 탐지** 공모전 예선 코드 (ITDA Team C).

YelpZip 리뷰를 노드로 두고 6개 relation(기본 3 + 커스텀 3)으로 그래프를 구성한 뒤, CARE-GNN 의 camouflage-resistant neighbor filtering + Skip GNN branch + Gated Relation Fusion + Auxiliary branch loss 를 결합한 다중 관계 GNN.

---

## 환경

- **Python**: **3.11+ 권장** (개발/검증은 3.13.12에서 진행)
- **OS**: Linux / macOS / Windows 모두 동작 (코드 내 yaml 로드는 UTF-8 명시)
- **GPU**: 권장 (200 epoch × 14 모델 학습 시 GPU에서 2~4시간, CPU에서는 하루 이상)

## 빠른 시작

```bash
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE

# Python 환경 (3.11+ 권장)
python -m venv .venv && source .venv/bin/activate

# 1) 의존성
pip install -r requirements.txt
# (GPU) PyG 가속 라이브러리 — CUDA 버전에 맞게:
# pip install pyg-lib torch-scatter torch-sparse \
#   -f https://data.pyg.org/whl/torch-2.11.0+cu121.html

# 2) 데이터 배치 (repo에는 포함되지 않음)
mkdir -p data/raw
# yelp_zip.csv 를 data/raw/ 에 둡니다.
# 기대 컬럼: user_id, prod_id, rating, label, date, text, tag

# 3) 전처리 + 그래프 빌드 (1회)
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# 4) 학습 (예: CAGE-CareRF 최종 모델)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf.yaml
```

전체 14개 모델 실행 명령은 [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) §3 참조.

---

## 디렉토리 구조

```text
.
├── src/
│   ├── preprocessing/   load_yelpzip / label_convert / sampling / feature_engineering
│   ├── graph/           build_{rur,rtr,rsr,burst,semsim,behavior,relations,relation_quality}
│   ├── filtering/       care_neighbor_filter
│   ├── models/          baseline_{mlp,gcn,gat,graphsage} / cage_rf_gnn_cheb /
│   │                    skip_cheb_branch / gated_relation_fusion / cage_carerf_gnn / losses
│   ├── training/        train / evaluate / threshold
│   └── utils/           seed / io / metrics / html_report
├── configs/             10 YAMLs (default, v8_skip, v9_twostage, cage_rf_skip_care,
│                                  cage_carerf, ablation_no_{care,skip,gating,aux,custom})
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── requirements.txt
└── run_baselines.py
```

데이터/학습 산출물/외부 reference 라이브러리는 `.gitignore`로 제외됩니다.

---

## 모델 개요

```text
Reviews (N, 140)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
6 relation graphs
    │
    ▼
SkipChebBranch ×6 (per relation)        ChebConv K=3, residual skip
    │
    ▼   stack → (N, 6, 128)
GatedRelationFusion (softmax α)         per-node α
    │
    ▼   (N, 128)
Projection → Main Classifier            logit
            + 6 × Auxiliary heads        aux_logits

Loss = Focal(logit, y) + 0.3 × mean_r BCE(aux_logit_r, y)
threshold @ valid PR-curve → Test 1회 평가 → PR-AUC / Macro-F1
```

자세한 구조는 [`docs/03_model_architecture.md`](docs/03_model_architecture.md).

---

## 예선 규정 준수 요약

- **Node = Review** 유지
- YelpZip 원본 → **Graph-Signal Preserving Hybrid Dense Sampling** → 64/16/20 split (stratified, seed=42)
- 라벨 변환 -1→1, 1→0
- 기본 relation 3개 (R-U-R, R-T-R, R-S-R) + 커스텀 relation 3개 (R-Burst-R, R-SemSim-R, R-Behavior-R)
- 모든 relation top-k / threshold 적용
- TF-IDF/SVD/Scaler **train-only fit** (transductive 가정 명시, `feature_meta.json`에 박제)
- relation quality 계산 시 **train labels only** (leakage-safe)
- threshold는 valid PR-curve에서 결정, **test set은 1회만 평가**
- PR-AUC / Macro-F1 / ROC-AUC / Precision / Recall / Accuracy 모두 저장

**PC-GNN inspired sampler는 메인 파이프라인에서 제외**: 대회 규정의 subgraph sampling 절차와 혼동 가능성을 피하기 위함. 클래스 불균형은 Focal Loss + class weight + threshold tuning으로 완화. `configs/cage_carerf.yaml`의 `sampler.enabled: false`에 의도가 박제되어 있음.

---

## 핵심 사실 (검증된 수치)

- **원본**: YelpZip 608,458 reviews, fraud_ratio 13.22%
- **샘플**: 50,000 nodes (Train 32k / Valid 8k / Test 10k, sampled fraud_ratio 11.16%)
- **Feature**: 140D = 128 (TF-IDF→SVD) + 12 (numeric)
- **6 Relations 총 1.65M edges**, `fraud_edge_lift`:
  R-Burst-R **1.96** (최강) · R-T-R 1.69 · R-U-R 1.26 · R-S-R 1.12 · R-SemSim-R 1.12 · R-Behavior-R 0.73 (약신호)
- **CARE filter 효과** (top_k 적용 후 edge 유지율):
  rur 97.5% · rtr 98.3% · rsr 65.2% · burst 99.7% · semsim 69.8% · behavior 42.7%

---

## 문서

- [`docs/01_code_structure.md`](docs/01_code_structure.md) — 코드 구조 / 파일별 역할 / 데이터 흐름
- [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) — 7단계 파이프라인 / 14개 모델 실행 명령 / leakage 차단 5중 방어
- [`docs/03_model_architecture.md`](docs/03_model_architecture.md) — CAGE-CareRF 모델 구조 / 각 모듈 명세 / Loss
- [`docs/04_setup_and_run.md`](docs/04_setup_and_run.md) — Setup, 의존성, 트러블슈팅

---

## 라이센스 / 팀

ITDA Team C — 2026 공모전 예선 제출용. 외부 reference 라이브러리(CARE-GNN, PC-GNN, DGFraud)는 본 repo에 포함되지 않으며, 각 원본 라이센스에 따릅니다.
