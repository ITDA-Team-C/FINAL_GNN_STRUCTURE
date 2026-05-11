# CAGE-CareRF GNN

> **Camouflage-Aware Gated Edge Relation-Fusion GNN**
> YelpZip 기반 **그래프신경망 기반 조직적 어뷰징 네트워크 탐지** 공모전 예선 코드 (ITDA Team C)

YelpZip 리뷰를 노드로 두고 6개 relation(기본 3 + 커스텀 3)으로 그래프를 구성한 뒤, CARE-GNN의 camouflage-resistant neighbor filtering + Skip GNN branch + Auxiliary branch loss를 결합한 다중 관계 GNN.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| **Python** | 3.11+ 권장 (개발/검증 3.13.12) |
| **OS** | Linux / macOS / Windows |
| **GPU** | 강력 추천 (CPU에서는 16모델 학습이 하루 이상) |
| 검증 환경 | Linux + CUDA 12.x |

---

## 2. 빠른 시작

```bash
# 1) Clone & venv
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
python -m venv .venv && source .venv/bin/activate

# 2) 의존성
pip install -r requirements.txt
# torch-geometric 2.7은 ChebConv/GCNConv/GATConv/SAGEConv를 자체 구현하므로
# pyg-lib 없이도 작동. 추가 가속 원하면 (Linux + CUDA 12.1):
# pip install torch-scatter torch-sparse \
#   -f https://data.pyg.org/whl/torch-2.11.0+cu121.html

# 3) 데이터 배치 (repo에 미포함, ~410MB)
mkdir -p data/raw
# yelp_zip.csv 를 data/raw/ 에 둠
# 기대 컬럼: user_id, prod_id, rating, label, date, text, tag

# 4) 전처리 + 그래프 빌드 (1회)
python -m src.preprocessing.load_yelpzip
python -m src.preprocessing.label_convert
python -m src.preprocessing.sampling
python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# 5) 16개 모델 한 번에 학습
python run_all_models.py
```

---

## 3. 16개 모델 학습 명령

실행 위치: repo 루트. 모든 산출물은 자동으로 다른 파일에 저장됨.

### A. Baseline 4종 (edge = union of 6 relations)

```bash
python -m src.training.train --model mlp        --config configs/default.yaml
python -m src.training.train --model gcn        --config configs/default.yaml
python -m src.training.train --model gat        --config configs/default.yaml
python -m src.training.train --model graphsage  --config configs/default.yaml
```

### B. CAGE-RF 계열 4종 (multi-relation GNN, 통합 구현)

```bash
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml             # Base (no skip)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml             # + Skip
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml         # + Two-Stage Refine
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml   # + CARE filter
```

### C. CAGE-CareRF FINAL 후보 (3개 Lean 변종)

```bash
# Lean-4: basic 3 + R-Burst-R           (rule-compliant minimal, fraud_edge_lift 1.96)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml

# Lean-5: basic 3 + R-Burst-R + R-SemSim-R  (drop noisy R-Behavior-R, lift 0.73)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_5.yaml

# Lean-6: 6 relations all              (= ablation_no_gating)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_6.yaml
```

> 3개 모두 `gating=off + Skip=on + Aux=on + CARE=on`. 차이는 `active_relations` 수만. 결과 보고 FINAL 1개 결정.

### D. Ablation 5종 (CAGE-CareRF v1 base = with Gating + 6 rel)

```bash
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_care.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_custom.yaml
```

### 한 번에 돌리기 — `run_all_models.py`

```bash
python run_all_models.py                       # 16개 전체
python run_all_models.py --only carerf         # Lean 3개만
python run_all_models.py --only ablation       # ablation만
python run_all_models.py --skip baselines      # baseline 제외
python run_all_models.py --continue-on-error   # 실패해도 계속
python run_all_models.py --dry-run             # 명령만 확인
```

---

## 4. 디렉토리 구조

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
├── configs/             12 YAMLs (default, v8_skip, v9_twostage, cage_rf_skip_care,
│                                  cage_carerf{,_lean,_lean_5,_lean_6},
│                                  ablation_no_{care,skip,gating,aux,custom})
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── requirements.txt     pandas/numpy/scikit-learn/torch/torch-geometric/pyyaml/scipy
├── run_all_models.py    16모델 일괄 학습 launcher
└── run_baselines.py     (legacy)
```

`.gitignore`로 제외: `data/`, `outputs/`, `CARE-GNN/`, `PC-GNN/`, `DGFraud/`, `markdown/`, `__pycache__/`.

---

## 5. 모델 개요

### FINAL 후보 (Lean 3변종)

```text
Reviews (N, 140)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
N basic + custom relation graphs       (N = 4 or 5 or 6 depending on variant)
    │
    ▼
SkipChebBranch ×N (per relation)        ChebConv K=3, residual skip
    │
    ▼   stack → (B, N, 128)
Mean Fusion (no gating)                 ← ablation 결과 gating은 음(-)의 효과
    │
    ▼   (B, 128)
Projection → Main Classifier            logit
            + N × Auxiliary heads        aux_logits

Loss = Focal(logit, y) + 0.3 × mean_r BCE(aux_logit_r, y)
threshold @ valid PR-curve → Test 1회 평가
```

### Lean 변종 비교

| 변종 | active_relations | n_rel | n_custom | 근거 |
|---|---|---|---|---|
| Lean-4 | rur, rtr, rsr, **burst** | 4 | 1 | 규정 충족 최소 — 가장 강한 custom 1개만 (lift 1.96) |
| Lean-5 | rur, rtr, rsr, **burst, semsim** | 5 | 2 | noisy Behavior(lift 0.73)만 제외 |
| Lean-6 | 6개 모두 | 6 | 3 | 다 유지 (= ablation_no_gating) |

자세한 구조는 [`docs/03_model_architecture.md`](docs/03_model_architecture.md) 참조.

---

## 6. 예선 규정 준수 요약

- ✅ **Node = Review** 유지
- ✅ YelpZip 원본 → **Graph-Signal Preserving Hybrid Dense Sampling** → 64/16/20 split (stratified, seed=42)
- ✅ 라벨 변환 `-1→1, 1→0`
- ✅ **기본 relation 3개**(R-U-R, R-T-R, R-S-R) + **커스텀 relation ≥1**(R-Burst-R 필수) — 규정의 "basic ≥ 1, custom ≥ 1" 충족
- ✅ 모든 relation top-k / threshold 적용
- ✅ TF-IDF/SVD/Scaler **train-only fit** (transductive 가정, `feature_meta.json`에 박제)
- ✅ relation quality 계산 시 **train labels only** (leakage-safe)
- ✅ threshold는 valid PR-curve에서 결정, **test set은 1회만 평가**
- ✅ PR-AUC / Macro-F1 / ROC-AUC / Precision / Recall / Accuracy 모두 저장

**PC-GNN inspired sampler는 메인 파이프라인에서 의도적 제외**: 대회 규정의 subgraph sampling 절차와 혼동 가능성을 피하기 위해. 클래스 불균형은 Focal Loss + class weight + threshold tuning으로 완화. `configs/cage_carerf*.yaml`의 `sampler.enabled: false`에 의도 명시.

---

## 7. 검증된 핵심 수치

### 데이터
- **원본**: YelpZip 608,458 reviews, fraud_ratio 13.22%
- **샘플**: 50,000 nodes (Train 32k / Valid 8k / Test 10k, sampled fraud_ratio 11.16%)
- **Feature**: 140D = 128 (TF-IDF→SVD) + 12 (numeric)

### 6 Relations 총 edges + fraud_edge_lift
| Relation | edges | fraud_edge_lift | 해석 |
|---|---|---|---|
| R-U-R | 49,754 | 1.26 | 보통 |
| R-T-R | 87,228 | 1.69 | 강함 |
| R-S-R | 597,432 | 1.12 | 약함 |
| **R-Burst-R** | 33,672 | **1.96** | **가장 강함** ★ |
| R-SemSim-R | 330,132 | 1.12 | 약함 |
| R-Behavior-R | 550,136 | **0.73** | random보다도 약함 ⚠️ |
| 합계 | 1,648,354 | — | — |

### CARE filter 효과 (top-k 적용 후 edge 유지율)
rur 97.5% · rtr 98.3% · rsr 65.2% · burst 99.7% · semsim 69.8% · behavior 42.7%

---

## 8. 16개 산출 파일 매핑

| # | 모델 | 결과 파일 |
|---|---|---|
| 1 | MLP | `outputs/cage_rf_gnn/metrics_mlp.json` |
| 2 | GCN | `outputs/cage_rf_gnn/metrics_gcn.json` |
| 3 | GAT | `outputs/cage_rf_gnn/metrics_gat.json` |
| 4 | GraphSAGE | `outputs/cage_rf_gnn/metrics_graphsage.json` |
| 5 | CAGE-RF Base | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v2.json` |
| 6 | CAGE-RF Skip (v8) | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v8_skip.json` |
| 7 | CAGE-RF Refine (v9) | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_v9_twostage.json` |
| 8 | CAGE-RF + CARE | `outputs/benchmark/cheb/metrics_cage_rf_gnn_cheb_cage_rf_skip_care.json` |
| 9 | **CAGE-CareRF Lean-4** | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_4.json` |
| 10 | **CAGE-CareRF Lean-5** | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_5.json` |
| 11 | **CAGE-CareRF Lean-6** | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_6.json` |
| 12 | w/o CARE | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_care.json` |
| 13 | w/o Skip | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_skip.json` |
| 14 | w/o Gating | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_gating.json` |
| 15 | w/o Aux Loss | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_aux.json` |
| 16 | w/o Custom | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_custom.json` |

---

## 9. 문서

- [`docs/01_code_structure.md`](docs/01_code_structure.md) — 디렉토리/파일 역할 / 데이터 흐름
- [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) — 7단계 파이프라인 / leakage 차단 / 트러블슈팅
- [`docs/03_model_architecture.md`](docs/03_model_architecture.md) — CAGE-CareRF 구조 / 각 모듈 / Loss
- [`docs/04_setup_and_run.md`](docs/04_setup_and_run.md) — 환경 setup / 의존성 / 재현성 체크리스트

---

## 10. 참고용 데이터셋 (Amazon / YelpChi)

YelpZip이 본 연구의 **메인 데이터셋**이며, **Amazon · YelpChi**는 모델 일반화 검증을 위한 **참고용 cross-dataset 실험** 입니다. 각각 독립된 폴더(`amazon/`, `yelchi/`)에 있으며 메인 코드(`src/`, `configs/`, `data/`)와는 완전히 분리되어 있습니다.

데이터 형식: **CARE-GNN / PC-GNN 표준 `.mat`** (3 relations 이미 처리됨).

### 10.1 데이터 준비 (gitignored, 사용자가 직접 추가)

```bash
# CARE-GNN repo에서 .mat 받기
git clone https://github.com/YingtongDou/CARE-GNN.git /tmp/CARE-GNN
unzip /tmp/CARE-GNN/data/Amazon.zip   -d amazon/data/
unzip /tmp/CARE-GNN/data/YelpChi.zip  -d yelchi/data/
```

배치 후 경로:
- `amazon/data/Amazon.mat`
- `yelchi/data/YelpChi.mat`

### 10.2 Amazon — 7개 모델 학습

#### 한 번에 (launcher)
```bash
python run_all_amazon.py                       # 7개 모두
python run_all_amazon.py --only cage_carerf    # 1개만
python run_all_amazon.py --mat-path /any/path/Amazon.mat
python run_all_amazon.py --epochs 100
python run_all_amazon.py --continue-on-error
python run_all_amazon.py --dry-run
```

#### 모델별 직접 실행
```bash
python -m amazon.src.train --model mlp                 --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model gcn                 --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model gat                 --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model graphsage           --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model cage_carerf         --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model cage_carerf_no_care --mat-path amazon/data/Amazon.mat
python -m amazon.src.train --model cage_carerf_no_aux  --mat-path amazon/data/Amazon.mat
```

산출:
```
amazon/outputs/metrics_mlp.json
amazon/outputs/metrics_gcn.json
amazon/outputs/metrics_gat.json
amazon/outputs/metrics_graphsage.json
amazon/outputs/metrics_cage_carerf.json          ← Amazon FINAL
amazon/outputs/metrics_cage_carerf_no_care.json
amazon/outputs/metrics_cage_carerf_no_aux.json
```

Amazon `.mat` 키: `features (N,25)`, `label`, `net_upu`, `net_usu`, `net_uvu` (3 relations).

### 10.3 YelpChi — 7개 모델 학습

#### 한 번에 (launcher)
```bash
python run_all_yelchi.py                       # 7개 모두
python run_all_yelchi.py --only cage_carerf    # 1개만
python run_all_yelchi.py --mat-path /any/path/YelpChi.mat
python run_all_yelchi.py --epochs 100
python run_all_yelchi.py --continue-on-error
python run_all_yelchi.py --dry-run
```

#### 모델별 직접 실행
```bash
python -m yelchi.src.train --model mlp                 --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model gcn                 --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model gat                 --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model graphsage           --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model cage_carerf         --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model cage_carerf_no_care --mat-path yelchi/data/YelpChi.mat
python -m yelchi.src.train --model cage_carerf_no_aux  --mat-path yelchi/data/YelpChi.mat
```

산출:
```
yelchi/outputs/metrics_mlp.json
yelchi/outputs/metrics_gcn.json
yelchi/outputs/metrics_gat.json
yelchi/outputs/metrics_graphsage.json
yelchi/outputs/metrics_cage_carerf.json          ← YelpChi FINAL
yelchi/outputs/metrics_cage_carerf_no_care.json
yelchi/outputs/metrics_cage_carerf_no_aux.json
```

YelpChi `.mat` 키: `features (N,32)`, `label`, `net_rur`, `net_rtr`, `net_rsr` (3 relations).

### 10.4 두 데이터셋 모두 한 번에 (전체 14개 모델)

```bash
python run_all_amazon.py && python run_all_yelchi.py
```

또는:
```bash
for ds in amazon yelchi; do
  python run_all_${ds}.py --continue-on-error
done
```

### 10.5 메인 YelpZip과의 비교

| 측면 | YelpZip (메인) | Amazon (참고) | YelpChi (참고) |
|---|---|---|---|
| Relations | 6 (basic 3 + custom 3) | 3 (UPU/USU/UVU) | 3 (RUR/RTR/RSR) |
| Node 단위 | Review | User | Review |
| Feature | 140D (TF-IDF+numeric, 직접 생성) | 25D (.mat 제공) | 32D (.mat 제공) |
| 전처리 | 7단계 파이프라인 | `.mat` load만 | `.mat` load만 |
| 학습 모델 수 | **16개** | **7개** | **7개** |

본 연구 모델의 핵심은 YelpZip 16모델 비교이며, Amazon/YelpChi는 동일 모델(CAGE-CareRF)이 **타 fraud dataset에서도 작동**하는지 검증하는 보조 실험입니다.

자세한 사항: [`amazon/README.md`](amazon/README.md), [`yelchi/README.md`](yelchi/README.md).

---

## 11. 라이센스 / 팀

ITDA Team C — 2026 공모전 예선 제출용. 외부 reference 라이브러리(CARE-GNN, PC-GNN, DGFraud)는 본 repo에 포함되지 않으며, 각 원본 라이센스에 따릅니다.
