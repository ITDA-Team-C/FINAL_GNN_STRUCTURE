# CAGE-CareRF GNN

> **Camouflage-Aware Gated Edge Relation-Fusion GNN**
> YelpZip 기반 **그래프신경망 기반 조직적 어뷰징 네트워크 탐지** 공모전 예선 코드 (ITDA Team C)

YelpZip 리뷰를 노드로 두고 6개 relation(기본 3 + 커스텀 3)으로 그래프를 구성한 뒤, CARE-GNN의 camouflage-resistant neighbor filtering + Skip GNN branch + Auxiliary branch loss를 결합한 다중 관계 GNN.

**최종 보고서**: [`REAL_FINAL_YELPZIP.md`](REAL_FINAL_YELPZIP.md) — 규정 §13 양식 8섹션.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| **Python** | 3.11+ 권장 (개발/검증 3.13.12) |
| **OS** | Linux / macOS / Windows |
| **GPU** | 강력 추천 (CPU에서는 16모델 × 5 seeds = 80회 학습이 매우 오래 걸림) |
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
python -m src.preprocessing.sampling          # 결정론적 이분 탐색 임계값
python -m src.preprocessing.feature_engineering
python -m src.graph.build_relations
python -m src.graph.relation_quality

# 5-a) 16개 모델 단일 seed 학습
python run_all_models.py

# 5-b) 16개 × 5 seeds = 80회 학습 (학회 표준, 권장)
python 5x_run_all_models.py

# 6) 샘플 분포 sanity check (선택)
python check_fraud_ratio.py
```

---

## 3. 16개 모델 학습 명령

실행 위치: repo 루트. 모든 산출물은 자동으로 다른 파일(`_seed{N}` 접미사 포함)에 저장됨.

### A. Baseline 4종 (edge = union of 6 relations)

```bash
python -m src.training.train --model mlp        --config configs/default.yaml --seed 42
python -m src.training.train --model gcn        --config configs/default.yaml --seed 42
python -m src.training.train --model gat        --config configs/default.yaml --seed 42
python -m src.training.train --model graphsage  --config configs/default.yaml --seed 42
```

### B. CAGE-RF 계열 4종 (multi-relation GNN, 통합 구현)

```bash
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml             --seed 42  # Base (no skip)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml             --seed 42  # + Skip
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml         --seed 42  # + Two-Stage Refine
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml   --seed 42  # + CARE filter
```

### C. CAGE-CareRF FINAL 후보 (3개 Lean 변종)

```bash
# Lean-4: basic 3 + R-Burst-R
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean.yaml --seed 42

# Lean-5: basic 3 + R-Burst-R + R-SemSim-R
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_5.yaml --seed 42

# Lean-6: 6 relations all (FINAL)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_lean_6.yaml --seed 42
```

### D. Ablation 5종 (CAGE-CareRF v1 base = with Gating + 6 rel)

```bash
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_care.yaml    --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml    --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml  --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml     --seed 42
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_custom.yaml  --seed 42
```

### 일괄 실행 — 단일 seed (`run_all_models.py`)

```bash
python run_all_models.py                       # 16개 전체
python run_all_models.py --only carerf         # Lean 3개만
python run_all_models.py --only ablation       # ablation만
python run_all_models.py --skip baselines      # baseline 제외
python run_all_models.py --continue-on-error   # 실패해도 계속
python run_all_models.py --dry-run             # 명령만 확인
```

### 일괄 실행 — Multi-seed × 5 (`5x_run_all_models.py`) **권장**

```bash
python 5x_run_all_models.py                                   # 16 × 5 = 80회
python 5x_run_all_models.py --seeds 42 123 2024 7 1234        # 커스텀 시드 (기본값과 동일)
python 5x_run_all_models.py --seeds 42 123                    # 2 시드만
python 5x_run_all_models.py --only carerf                     # 그룹 선택
python 5x_run_all_models.py --continue-on-error
python 5x_run_all_models.py --dry-run
```

실행 후 자동 집계: `outputs/multi_seed_summary.json` — 모델별 `pr_auc_mean ± std`, `macro_f1_mean ± std`, **`g_mean_mean ± std`** 저장.

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
│   └── utils/           seed / io / metrics (PR-AUC, Macro-F1, G-Mean, ...) / html_report
├── configs/             12 YAMLs (default, v8_skip, v9_twostage, cage_rf_skip_care,
│                                  cage_carerf{,_lean,_lean_5,_lean_6},
│                                  ablation_no_{care,skip,gating,aux,custom})
├── amazon/              참고 데이터셋: Amazon (CARE-GNN .mat)
├── yelchi/              참고 데이터셋: YelpChi (CARE-GNN .mat)
├── docs/                01_code_structure / 02_training_pipeline / 03_model_architecture /
│                        04_setup_and_run
├── REAL_FINAL_YELPZIP.md     최종 보고서 (규정 §13 8섹션 양식)
├── check_fraud_ratio.py      샘플 분포 sanity check
├── run_all_models.py         YelpZip 16모델 단일 seed launcher
├── 5x_run_all_models.py      YelpZip 16 × 5 seeds = 80회 launcher
├── run_all_amazon.py         Amazon 7모델 단일 seed launcher
├── 5x_run_all_amazon.py      Amazon 7 × 5 seeds = 35회 launcher
├── run_all_yelchi.py         YelpChi 7모델 단일 seed launcher
└── 5x_run_all_yelchi.py      YelpChi 7 × 5 seeds = 35회 launcher
```

`.gitignore`로 제외: `data/`, `outputs/`, `CARE-GNN/`, `PC-GNN/`, `DGFraud/`, `markdown/`, `__pycache__/`.

---

## 5. 모델 개요

### FINAL 후보 (Lean 3변종)

```text
Reviews (N=47,125, F=140)
    │
    ▼   [offline] CARE neighbor filter (feature cosine top-k, label-free)
N basic + custom relation graphs       (N = 4 or 5 or 6 depending on variant)
    │
    ▼
SkipChebBranch ×N (per relation)        ChebConv K=3, residual skip
    │
    ▼   stack → (B, N, 128)
Gated / Mean Fusion
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
| Lean-5 | rur, rtr, rsr, **burst, semsim** | 5 | 2 | noisy Behavior(lift 0.73) 제외 |
| **Lean-6 (FINAL)** | 6개 모두 | 6 | 3 | 약신호도 시너지 효과 → Macro F1 1위 |

자세한 구조는 [`docs/03_model_architecture.md`](docs/03_model_architecture.md) 참조.

---

## 6. 예선 규정 준수 요약

- ✅ **Node = Review** 유지
- ✅ YelpZip 원본 → **결정론적 이분 탐색 임계값 샘플링** → 47,125 노드 → 64/16/20 stratified split (`random_state=42`)
- ✅ **무작위 추출 0건** — `df.sample` / `np.random.choice` / `rng.choice` / `np.random.default_rng` 모두 `src/`에 없음 (`grep` 검증 완료)
- ✅ 라벨 변환 `-1→1, 1→0` (`src/preprocessing/label_convert.py:23`)
- ✅ **기본 relation 3개**(R-U-R, R-T-R, R-S-R) + **커스텀 relation 3개**(R-Burst-R, R-SemSim-R, R-Behavior-R) — 규정의 "basic ≥ 1, custom ≥ 1" 충족
- ✅ 모든 relation 결정론적 top-k 또는 threshold 적용 (무작위 후보 선택 X)
- ✅ TF-IDF/SVD/Scaler **train-only fit** (transductive 가정, `feature_meta.json`에 박제)
- ✅ relation quality 계산 시 **train labels only** (leakage-safe)
- ✅ threshold는 valid PR-curve에서 결정, **test set은 1회만 평가**
- ✅ **PR-AUC / Macro-F1 / G-Mean / ROC-AUC / Precision / Recall / Accuracy** 모두 저장
- ✅ **Multi-seed (5개) 평균** 보고 (`5x_run_all_models.py`) — 학회 표준

**PC-GNN inspired sampler는 메인 파이프라인에서 의도적 제외**: 대회 규정의 subgraph sampling 절차와 혼동 가능성을 피하기 위해. 클래스 불균형은 Focal Loss + class weight + threshold tuning으로 완화. `configs/cage_carerf*.yaml`의 `sampler.enabled: false`에 의도 명시.

---

## 7. 검증된 핵심 수치

### 데이터
- **원본**: YelpZip 608,458 reviews, 5,044 prod / 260,239 user, fraud_ratio 13.22%
- **샘플** (결정론적 이분 탐색 결과): **47,125 nodes** (Train 30,160 / Valid 7,540 / Test 9,425)
- **샘플 fraud_ratio**: 11.16% (Train/Valid/Test 모두 동일, stratified)
- **임계값**: `head_pu = 6` (상품·사용자 공통), `head_m = 1` (월) → union 자연스럽게 ≤ 50,000
- **Feature**: 140D = 128 (TF-IDF→SVD) + 12 (numeric), train-only fit

### 6 Relations 도메인 가설
| Relation | 예상 신호 | 도메인 의미 |
|---|---|---|
| R-Burst-R | **최강** (lift 예상 ≈ 1.9~2.0) | 단기 평판 폭격 (7일/±1점) |
| R-T-R | 강 | 동일 상품 동월 시간 집중 |
| R-U-R | 중 | 동일 사용자 반복 작성 |
| R-S-R | 중하 | 동일 상품 동일 별점 |
| R-SemSim-R | 중하 | TF-IDF→SVD 코사인 top-5 |
| R-Behavior-R | 약 | user 행동 벡터 cosine top-5 |

> 정확한 `fraud_edge_lift` 수치는 새 47,125 노드 샘플에서 `python -m src.graph.relation_quality` 실행 후 `outputs/metrics/relation_quality.json` 확인.

### CARE filter top-k 적용 후 edge 유지율 (참고)
rur 97.5% · rtr 98.3% · rsr 65.2% · burst 99.7% · semsim 69.8% · behavior 42.7%

---

## 8. 16개 산출 파일 매핑 (Multi-seed 형식)

`5x_run_all_models.py` 실행 시 각 모델당 5개 파일이 생성됨 (`_seed{42,123,2024,7,1234}`). 총 80개 + 집계 1개.

| # | 모델 | 결과 파일 패턴 |
|---|---|---|
| 1 | MLP | `outputs/cage_rf_gnn/metrics_mlp_seed{N}.json` |
| 2 | GCN | `outputs/cage_rf_gnn/metrics_gcn_seed{N}.json` |
| 3 | GAT | `outputs/cage_rf_gnn/metrics_gat_seed{N}.json` |
| 4 | GraphSAGE | `outputs/cage_rf_gnn/metrics_graphsage_seed{N}.json` |
| 5 | CAGE-RF Base | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v2_seed{N}.json` |
| 6 | CAGE-RF Skip (v8) | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v8_skip_seed{N}.json` |
| 7 | CAGE-RF Refine (v9) | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_v9_twostage_seed{N}.json` |
| 8 | CAGE-RF + CARE | `outputs/benchmark/CHEB/metrics_cage_rf_gnn_cheb_cage_rf_skip_care_seed{N}.json` |
| 9 | CAGE-CareRF Lean-4 | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_4_seed{N}.json` |
| 10 | CAGE-CareRF Lean-5 | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_5_seed{N}.json` |
| 11 | **CAGE-CareRF Lean-6 (FINAL)** | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_cage_carerf_lean_6_seed{N}.json` |
| 12 | w/o CARE | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_care_seed{N}.json` |
| 13 | w/o Skip | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_skip_seed{N}.json` |
| 14 | w/o Gating | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_gating_seed{N}.json` |
| 15 | w/o Aux Loss | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_aux_seed{N}.json` |
| 16 | w/o Custom | `outputs/cage_rf_gnn/metrics_cage_carerf_gnn_ablation_no_custom_seed{N}.json` |
| — | **집계** | `outputs/multi_seed_summary.json` (mean ± std) |

각 JSON에는 `seed`, `best_threshold`, `valid_metrics`, `test_metrics`가 포함되며, `test_metrics` 안에 PR-AUC / Macro F1 / **G-Mean** / ROC-AUC / Precision / Recall / Accuracy / recall_pos / recall_neg가 들어있음.

---

## 9. 문서

- [`REAL_FINAL_YELPZIP.md`](REAL_FINAL_YELPZIP.md) — 최종 보고서 (규정 §13 양식, 8섹션)
- [`docs/01_code_structure.md`](docs/01_code_structure.md) — 디렉토리/파일 역할 / 데이터 흐름
- [`docs/02_training_pipeline.md`](docs/02_training_pipeline.md) — 7단계 파이프라인 / leakage 차단 / 트러블슈팅
- [`docs/03_model_architecture.md`](docs/03_model_architecture.md) — CAGE-CareRF 구조 / 각 모듈 / Loss
- [`docs/04_setup_and_run.md`](docs/04_setup_and_run.md) — 환경 setup / 의존성 / 재현성 체크리스트

---

## 10. 참고용 데이터셋 (Amazon / YelpChi)

YelpZip이 본 연구의 **메인 데이터셋**이며, **Amazon · YelpChi**는 모델 일반화 검증을 위한 **참고용 cross-dataset 실험**입니다. 각각 독립된 폴더(`amazon/`, `yelchi/`)에 있으며 메인 코드(`src/`, `configs/`, `data/`)와는 완전히 분리되어 있습니다.

데이터 형식: **CARE-GNN / PC-GNN 표준 `.mat`** (3 relations + 노드/레이블 이미 포함). YelpZip처럼 별도 subgraph sampling 단계가 없고, `.mat` 로드 후 stratified split만 수행합니다.

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

### 10.2 Amazon — 7개 모델

#### 단일 seed (`run_all_amazon.py`)
```bash
python run_all_amazon.py                       # 7개 모두
python run_all_amazon.py --only cage_carerf    # 1개만
python run_all_amazon.py --epochs 100
python run_all_amazon.py --continue-on-error
```

#### Multi-seed × 5 (`5x_run_all_amazon.py`) **권장**
```bash
python 5x_run_all_amazon.py                          # 7 × 5 = 35회
python 5x_run_all_amazon.py --seeds 42 123
python 5x_run_all_amazon.py --only cage_carerf
python 5x_run_all_amazon.py --continue-on-error
```

#### 모델별 직접 실행 (seed 지정)
```bash
python -m amazon.src.train --model cage_carerf --mat-path amazon/data/Amazon.mat --seed 42
```

산출 (multi-seed):
```
amazon/outputs/metrics_<model>_seed{42,123,2024,7,1234}.json   ← 35개
amazon/outputs/multi_seed_summary.json                          ← 집계 mean ± std
```

Amazon `.mat` 키: `features (N,25)`, `label`, `net_upu`, `net_usu`, `net_uvu` (3 relations).

### 10.3 YelpChi — 7개 모델

#### 단일 seed (`run_all_yelchi.py`)
```bash
python run_all_yelchi.py                       # 7개 모두
python run_all_yelchi.py --only cage_carerf
python run_all_yelchi.py --epochs 100
python run_all_yelchi.py --continue-on-error
```

#### Multi-seed × 5 (`5x_run_all_yelchi.py`) **권장**
```bash
python 5x_run_all_yelchi.py                          # 7 × 5 = 35회
python 5x_run_all_yelchi.py --seeds 42 123
python 5x_run_all_yelchi.py --only cage_carerf
python 5x_run_all_yelchi.py --continue-on-error
```

#### 모델별 직접 실행 (seed 지정)
```bash
python -m yelchi.src.train --model cage_carerf --mat-path yelchi/data/YelpChi.mat --seed 42
```

산출 (multi-seed):
```
yelchi/outputs/metrics_<model>_seed{42,123,2024,7,1234}.json   ← 35개
yelchi/outputs/multi_seed_summary.json                          ← 집계 mean ± std
```

YelpChi `.mat` 키: `features (N,32)`, `label`, `net_rur`, `net_rtr`, `net_rsr` (3 relations).

### 10.4 세 데이터셋 모두 한 번에 (Multi-seed)

```bash
# YelpZip (메인) + Amazon + YelpChi 모두 5 seeds로
python 5x_run_all_models.py --continue-on-error && \
python 5x_run_all_amazon.py --continue-on-error && \
python 5x_run_all_yelchi.py --continue-on-error
```

총 학습 횟수: YelpZip 80 + Amazon 35 + YelpChi 35 = **150회**.

### 10.5 메인 YelpZip과의 비교

| 측면 | YelpZip (메인) | Amazon (참고) | YelpChi (참고) |
|---|---|---|---|
| Relations | 6 (basic 3 + custom 3) | 3 (UPU/USU/UVU) | 3 (RUR/RTR/RSR) |
| Node 단위 | Review | User | Review |
| Feature | 140D (TF-IDF+numeric, 직접 생성) | 25D (.mat 제공) | 32D (.mat 제공) |
| 전처리 | 7단계 파이프라인 (sampling 포함) | `.mat` load만 | `.mat` load만 |
| Subgraph 샘플링 | **결정론적 이분 탐색** (무작위 0건) | 해당 없음 (.mat에 노드 고정) | 해당 없음 (.mat에 노드 고정) |
| 학습 모델 수 | **16개** | **7개** | **7개** |
| Multi-seed 학습 | **80회** (16 × 5) | **35회** (7 × 5) | **35회** (7 × 5) |

본 연구 모델의 핵심은 YelpZip 16모델 비교이며, Amazon/YelpChi는 동일 모델(CAGE-CareRF)이 **타 fraud dataset에서도 작동**하는지 검증하는 보조 실험입니다.

자세한 사항: [`amazon/README.md`](amazon/README.md), [`yelchi/README.md`](yelchi/README.md).

---

## 11. 라이센스 / 팀

ITDA Team C — 2026 공모전 예선 제출용. 외부 reference 라이브러리(CARE-GNN, PC-GNN, DGFraud)는 본 repo에 포함되지 않으며, 각 원본 라이센스에 따릅니다.
