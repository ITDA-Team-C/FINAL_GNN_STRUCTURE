# YelpChi Fraud Detection (CARE-GNN / PC-GNN format)

YelpZip 메인 파이프라인과 독립된, YelpChi 데이터셋(.mat 포맷)용 학습 코드.
Cross-dataset 일반화 검증용 보조 실험 — **7모델 × 5 seeds = 35회 학습**.

> **결과**: CAGE-CareRF 계열 3개가 PR-AUC 0.71~0.73으로 압도. Baseline GraphSAGE 0.62, MLP 0.51 → 멀티 관계 그래프 신호가 결정적 (Amazon과 정반대). 본 연구의 backbone(다중 relation 분리·융합)이 review-level fraud detection에서 일반화됨을 입증.

---

## 0. 핵심 결과 (5 seeds 평균)

| Rank | Model | PR-AUC | Macro F1 | G-Mean |
|:----:|-------|:------:|:--------:|:------:|
| 🥇 1 | CAGE-CareRF w/o CARE | **0.7309 ± 0.0171** | **0.8006 ± 0.0074** | 0.8043 ± 0.0194 |
| 2 | CAGE-CareRF w/o Aux | 0.7198 ± 0.0155 | 0.8002 ± 0.0060 | **0.8094 ± 0.0135** |
| 3 | CAGE-CareRF | 0.7114 ± 0.0148 | 0.7969 ± 0.0058 | 0.8017 ± 0.0150 |
| 4 | GraphSAGE | 0.6178 ± 0.0162 | 0.7412 ± 0.0080 | 0.7192 ± 0.0209 |
| 5 | MLP | 0.5080 ± 0.0227 | 0.7012 ± 0.0115 | 0.6742 ± 0.0111 |
| 6 | GCN | 0.2486 ± 0.0095 | 0.5660 ± 0.0052 | 0.4548 ± 0.0197 |
| 7 | GAT | 0.2401 ± 0.0086 | 0.5596 ± 0.0043 | 0.4636 ± 0.0229 |

**관찰**:
- CAGE-CareRF 계열 3개가 압도 (PR-AUC 0.71~0.73)
- Baseline GraphSAGE 0.62, MLP 0.51 → **멀티 관계 그래프 신호가 결정적** (Amazon과 대조적 — Amazon은 노드 feature 단독으로도 잘 풀림)
- GCN/GAT는 망함 (둘 다 0.24) → 단일 그래프(union)로는 review-level 사기 신호를 잡지 못함
- YelpChi에서는 1위가 `w/o CARE` — CARE filter가 review-level + 3 relation 환경에서는 오히려 noise를 일부 잘라낸 결과로 추정 (YelpZip 6 relation에서는 CARE가 결정타였음과 대조)

---

## 1. 데이터 준비

CARE-GNN 또는 PC-GNN repo에서 `YelpChi.mat`을 받아 다음 위치에 둡니다:

```
yelchi/data/YelpChi.mat
```

`YelpChi.mat` 필수 키:
- `features` : sparse (N, 32) — node feature (이미 생성됨)
- `label`    : (N,) — {0, 1}, 1 = fraud
- `net_rur`  : sparse (N, N) — review-user-review adjacency
- `net_rtr`  : sparse (N, N) — review-time-review adjacency
- `net_rsr`  : sparse (N, N) — review-star-review adjacency

다운로드 예시:
```bash
git clone https://github.com/YingtongDou/CARE-GNN.git /tmp/CARE-GNN
unzip /tmp/CARE-GNN/data/YelpChi.zip -d yelchi/data/
```

---

## 2. 실행

### 단일 seed (`run_all_yelchi.py`)

```bash
python run_all_yelchi.py                       # 7개 모델 전부
python run_all_yelchi.py --only cage_carerf    # 1개만
python run_all_yelchi.py --mat-path /any/path/YelpChi.mat
python run_all_yelchi.py --epochs 100
python run_all_yelchi.py --continue-on-error
```

### Multi-seed × 5 (`5x_run_all_yelchi.py`) **권장**

```bash
python 5x_run_all_yelchi.py                          # 7 × 5 = 35회
python 5x_run_all_yelchi.py --seeds 42 123 2024 7 1234
python 5x_run_all_yelchi.py --only cage_carerf
python 5x_run_all_yelchi.py --continue-on-error
```

### 모델별 직접 실행 (seed 지정)

```bash
python -m yelchi.src.train --model cage_carerf --mat-path yelchi/data/YelpChi.mat --seed 42
```

---

## 3. 학습 모델 7종

| # | 모델 | 설명 |
|---|---|---|
| 1 | MLP | graph-free baseline |
| 2 | GCN | 3 relation union edge |
| 3 | GAT | 동일 (multi-head attention) |
| 4 | GraphSAGE | 동일 (inductive) |
| 5 | **CAGE-CareRF** | 본 연구 모델 (Skip + Gating + Aux + CARE) — 분리 모듈 구현 |
| 6 | CAGE-CareRF w/o CARE | CARE filter ablation (실측 1위) |
| 7 | CAGE-CareRF w/o Aux | Aux loss ablation |

> YelpChi는 review-level node + 3 relation (R-U-R, R-T-R, R-S-R) — YelpZip의 기본 relation 3개와 동일한 의미. 단 CARE-GNN repo에서 이미 노드/엣지가 생성되어 있어 별도 sampling 단계가 없음.
> 모델명 `CAGE-CareRF`는 분리 모듈 구현(`cage_carerf_gnn.py`)의 코드 클래스명을 그대로 표기한 것 — 알고리즘적으로는 메인 YelpZip의 CAGE-RF + CARE와 동일 backbone (단 6 relation 대신 3 relation).

---

## 4. 산출물 (Multi-seed)

```
yelchi/outputs/
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
  "dataset": "yelchi",
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
yelchi/
├── data/YelpChi.mat             ← 사용자가 추가
├── configs/default.yaml
├── src/
│   ├── data_loader.py           (.mat → torch + train/valid/test split)
│   ├── models.py                (MLP/GCN/GAT/SAGE/CAGE-CareRF + Focal+Aux loss)
│   ├── train.py                 (학습 + threshold@valid + test 1회 평가)
│   └── metrics.py               (PR-AUC, Macro F1, G-Mean, ROC-AUC, ...)
└── outputs/

run_all_yelchi.py                ← repo 루트 (7모델 단일 seed launcher)
5x_run_all_yelchi.py             ← repo 루트 (7 × 5 = 35회 launcher)
```

YelpZip 메인 파이프라인과 **완전히 분리**.
