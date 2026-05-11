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
# torch-geometric 2.7은 ChebConv/GCNConv/GATConv/SAGEConv를 자체 구현하므로
# pyg-lib 없이도 학습 가능. 추가 가속이 필요하면 (Linux + CUDA 12.1 예시):
# pip install torch-scatter torch-sparse \
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

# 4) 학습 (FINAL 모델: CAGE-CareRF-Lean)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml
```

> **FINAL 모델은 `cage_carerf_lean.yaml`을 사용합니다.** Ablation 결과 Gated Fusion과 Custom Relations가 PR-AUC/Macro-F1에 음(-)의 marginal 효과를 보여서, **mean fusion + 기본 relation 3개(R-U-R, R-T-R, R-S-R) + Skip + CARE filter + Aux Loss** 조합을 최종 모델로 선택했습니다. 기존 `cage_carerf.yaml` (with Gating + Custom)은 해석 가능성 비교 모델로 유지됩니다.

---

## 전체 14개 모델 학습 명령

실행 위치: repo 루트(`FINAL_GNN_STRUCTURE/`). 모든 산출물은 자동으로 다른 파일에 저장되어 덮어쓰기 위험 없음.

### A. Baseline 4종 (edge = union of 6 relations)

```bash
python -m src.training.train --model mlp        --config configs/default.yaml
python -m src.training.train --model gcn        --config configs/default.yaml
python -m src.training.train --model gat        --config configs/default.yaml
python -m src.training.train --model graphsage  --config configs/default.yaml
```

### B. CAGE-RF 계열 4종 (multi-relation GNN)

```bash
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml             # Base
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml             # + Skip
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml         # + Two-Stage Refine
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml   # + CARE filter
```

### C. CAGE-CareRF FINAL (제안 모델 — Lean)

```bash
# FINAL (옵션 A) — Lean: Gating off + 기본 relation 3개만 + Skip + CARE + Aux
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml

# (선택) 비교 모델 v1 — with Gating + Custom relations 6개
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf.yaml
```

### D. Ablation 5종 (FINAL에서 한 항목씩 제거)

```bash
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_care.yaml      # w/o CARE filter
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml      # w/o Skip
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml    # w/o Gated Fusion (mean fallback)
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml       # w/o Aux Loss
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_custom.yaml    # 기본 relation 3개만 (Burst/SemSim/Behavior 제거)
```

### 한 번에 돌리는 스크립트

**가장 간단한 방법 — 동봉된 `run_all_models.py` 사용:**

```bash
# 전체 15개 모델 순차 학습 (A + B + C + D)
python run_all_models.py

# 일부 그룹만 (예: ablation만)
python run_all_models.py --only ablation
python run_all_models.py --only baselines,carerf

# 일부 그룹 제외
python run_all_models.py --skip baselines

# 중간에 1개 실패해도 계속 (다른 모델은 진행)
python run_all_models.py --continue-on-error

# 명령만 확인 (실행 안 함)
python run_all_models.py --dry-run
```

스크립트는 끝에 모델별 exit code, 소요 시간, 결과 파일 목록을 요약 출력합니다.

**또는 inline bash:**

```bash
#!/bin/bash
set -e
for m in mlp gcn gat graphsage; do
  python -m src.training.train --model $m --config configs/default.yaml
done
for cfg in default v8_skip v9_twostage cage_rf_skip_care; do
  python -m src.training.train --model cage_rf_gnn_cheb --config configs/$cfg.yaml
done
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf.yaml
for abl in no_care no_skip no_gating no_aux no_custom; do
  python -m src.training.train --model cage_carerf_gnn --config configs/ablation_$abl.yaml
done
echo "ALL 15 MODELS DONE"
ls -la outputs/cage_rf_gnn/metrics_*.json outputs/benchmark/cheb/metrics_*.json
```

### 산출 파일 매핑표

| # | 모델 | 결과 파일 |
|---|---|---|
| 1 | MLP | `outputs/cage_rf_gnn/metrics_mlp.json` |
| 2 | GCN | `outputs/cage_rf_gnn/metrics_gcn.json` |
| 3 | GAT | `outputs/cage_rf_gnn/metrics_gat.json` |
| 4 | GraphSAGE | `outputs/cage_rf_gnn/metrics_graphsage.json` |
| 5 | CAGE-RF Base | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v2.json` |
| 6 | CAGE-RF Skip | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v8_skip.json` |
| 7 | CAGE-RF Refine | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v9_twostage.json` |
| 8 | CAGE-RF + CARE | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_cage_rf_skip_care.json` |
| 9 | **CAGE-CareRF FINAL (Lean)** | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean.json` |
| 9b | CAGE-CareRF v1 (with Gating/Custom, 비교용) | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_v1.json` |
| 10 | w/o CARE | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_care.json` |
| 11 | w/o Skip | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_skip.json` |
| 12 | w/o Gating | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_gating.json` |
| 13 | w/o Aux | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_aux.json` |
| 14 | w/o Custom | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_custom.json` |

더 자세한 설명(7단계 파이프라인, leakage 차단, 트러블슈팅)은 [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) 참조.

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

## 모델 개요 (FINAL = Lean)

```text
Reviews (N, 140)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
3 basic relation graphs (R-U-R, R-T-R, R-S-R)        ← ablation 후 custom 3개 제외
    │
    ▼
SkipChebBranch ×3 (per relation)        ChebConv K=3, residual skip
    │
    ▼   stack → (N, 3, 128)
Mean Fusion (no gating)                  ← ablation 후 gating 제외
    │
    ▼   (N, 128)
Projection → Main Classifier            logit
            + 3 × Auxiliary heads        aux_logits

Loss = Focal(logit, y) + 0.3 × mean_r BCE(aux_logit_r, y)
threshold @ valid PR-curve → Test 1회 평가 → PR-AUC / Macro-F1
```

비교 모델 v1 (with Gating + Custom Relations 6개)은 동일 코드(`cage_carerf_gnn.py`)의 토글로 학습되며, 보고서에서 해석 가능성 trade-off 분석용으로 사용됩니다.

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
