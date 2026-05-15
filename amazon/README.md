# Amazon Fraud Detection (CARE-GNN / PC-GNN format)

YelpZip 메인 파이프라인과 독립된, Amazon 데이터셋(.mat 포맷)용 학습 코드.
Cross-dataset 일반화 검증용 보조 실험 — **7모델 × 5 seeds = 35회 학습**.

> **결과**: 상위 5개 모델이 PR-AUC 0.80~0.82 범위로 거의 동등 (std 내). MLP가 1위 — Amazon은 user-level 25D feature에 사기 신호가 강하게 인코딩되어 그래프 message passing의 한계 효용이 낮음 (YelpZip과 대조적).

---

## 0. 핵심 결과 (5 seeds 평균)

| Rank | Model | PR-AUC | Macro F1 | G-Mean |
|:----:|-------|:------:|:--------:|:------:|
| 🥇 1 | MLP | **0.8203 ± 0.0242** | **0.9037 ± 0.0026** | **0.8622 ± 0.0043** |
| 2 | CAGE-CareRF | 0.8162 ± 0.0137 | 0.8996 ± 0.0065 | 0.8564 ± 0.0049 |
| 3 | CAGE-CareRF w/o CARE | 0.8117 ± 0.0348 | 0.8944 ± 0.0104 | 0.8410 ± 0.0205 |
| 4 | GraphSAGE | 0.8112 ± 0.0186 | 0.9002 ± 0.0062 | 0.8538 ± 0.0104 |
| 5 | CAGE-CareRF w/o Aux | 0.8043 ± 0.0229 | 0.8905 ± 0.0123 | 0.8538 ± 0.0092 |
| 6 | GCN | 0.2474 ± 0.0134 | 0.6201 ± 0.0091 | 0.5513 ± 0.0434 |
| 7 | GAT | 0.1491 ± 0.0774 | 0.5431 ± 0.0478 | 0.4734 ± 0.1395 |

**관찰**:
- 상위 5개 모델은 PR-AUC 0.80~0.82 범위로 거의 동등 (std 내)
- **MLP가 1위** — Amazon은 user-level 노드 25D feature에 사기 신호가 강하게 인코딩되어 그래프 message passing의 한계 효용이 낮음
- GCN/GAT는 망함 (PR-AUC < 0.25, GAT는 std 0.077로 극도로 불안정) → 단일 그래프(union)로는 user-level 사기 신호를 잡지 못함

---

## 1. 데이터 준비

CARE-GNN 또는 PC-GNN repo에서 `Amazon.mat`을 받아 다음 위치에 둡니다:

```
amazon/data/Amazon.mat
```

`Amazon.mat` 필수 키:
- `features` : sparse (N, 25) — node feature (이미 생성됨)
- `label`    : (N,) — {0, 1}, 1 = fraud
- `net_upu`  : sparse (N, N) — user-product-user adjacency
- `net_usu`  : sparse (N, N) — user-star-user adjacency
- `net_uvu`  : sparse (N, N) — user-vote-user adjacency

다운로드 예시:
```bash
git clone https://github.com/YingtongDou/CARE-GNN.git /tmp/CARE-GNN
unzip /tmp/CARE-GNN/data/Amazon.zip -d amazon/data/
```

---

## 2. 실행

### 단일 seed (`run_all_amazon.py`)

```bash
python run_all_amazon.py                       # 7개 모델 전부
python run_all_amazon.py --only cage_carerf    # 1개만
python run_all_amazon.py --mat-path /any/path/Amazon.mat
python run_all_amazon.py --epochs 100
python run_all_amazon.py --continue-on-error
```

### Multi-seed × 5 (`5x_run_all_amazon.py`) **권장**

```bash
python 5x_run_all_amazon.py                          # 7 × 5 = 35회
python 5x_run_all_amazon.py --seeds 42 123 2024 7 1234
python 5x_run_all_amazon.py --only cage_carerf
python 5x_run_all_amazon.py --continue-on-error
```

### 모델별 직접 실행 (seed 지정)

```bash
python -m amazon.src.train --model cage_carerf --mat-path amazon/data/Amazon.mat --seed 42
```

---

## 3. 학습 모델 7종

| # | 모델 | 설명 |
|---|---|---|
| 1 | MLP | graph-free 텍스트 baseline (실측 1위) |
| 2 | GCN | 3 relation의 union edge |
| 3 | GAT | 동일 (multi-head attention) |
| 4 | GraphSAGE | 동일 (inductive aggregation) |
| 5 | **CAGE-CareRF** | 본 연구 모델 (Skip + Gating + Aux + CARE) — 분리 모듈 구현 |
| 6 | CAGE-CareRF w/o CARE | CARE filter 제거 ablation |
| 7 | CAGE-CareRF w/o Aux | Aux loss 제거 ablation |

> YelpZip의 6 relation 구조와 달리 Amazon은 3 relation(UPU/USU/UVU)이라 Lean-4/5 같은 변종은 적용 불가.
> 모델명 `CAGE-CareRF`는 분리 모듈 구현(`cage_carerf_gnn.py`)의 코드 클래스명을 그대로 표기한 것 — 알고리즘적으로는 메인 YelpZip의 CAGE-RF + CARE와 동일 backbone.

---

## 4. 산출물 (Multi-seed)

```
amazon/outputs/
├── metrics_mlp_seed{42,123,2024,7,1234}.json                  ← 5
├── metrics_gcn_seed{N}.json                                    ← 5
├── metrics_gat_seed{N}.json                                    ← 5
├── metrics_graphsage_seed{N}.json                              ← 5
├── metrics_cage_carerf_seed{N}.json                            ← 5  (FINAL)
├── metrics_cage_carerf_no_care_seed{N}.json                    ← 5
├── metrics_cage_carerf_no_aux_seed{N}.json                     ← 5
└── multi_seed_summary.json                                     ← 집계 (mean ± std)
```

총 35개 + 집계 1개. 단일 seed 실행 시에는 `_seed{N}` 접미사 없이 모델당 1개만 생성.

각 JSON 구조:
```json
{
  "dataset": "amazon",
  "model": "cage_carerf",
  "seed": 42,
  "best_threshold": 0.42,
  "valid_metrics": {"pr_auc": ..., "macro_f1": ..., "roc_auc": ..., ...},
  "test_metrics":  {"pr_auc": ..., "macro_f1": ..., "g_mean": ..., ...}
}
```

---

## 5. 코드 구조

```
amazon/
├── data/Amazon.mat              ← 사용자가 추가
├── configs/default.yaml
├── src/
│   ├── data_loader.py           (.mat → torch + train/valid/test split)
│   ├── models.py                (MLP/GCN/GAT/SAGE/CAGE-CareRF + Focal+Aux loss)
│   ├── train.py                 (학습 루프 + threshold@valid + test 1회 평가)
│   └── metrics.py               (PR-AUC, Macro F1, G-Mean, ROC-AUC, ...)
└── outputs/                     (실행 시 자동 생성)

run_all_amazon.py                ← repo 루트 (7모델 단일 seed launcher)
5x_run_all_amazon.py             ← repo 루트 (7 × 5 = 35회 launcher)
```

YelpZip 메인 파이프라인(`src/`, `configs/`, `data/raw/`)과는 **완전히 분리**되어 있으며, 코드 공유 없음.
