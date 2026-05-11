# YelpChi Fraud Detection (CARE-GNN / PC-GNN format)

YelpZip 메인 파이프라인과 독립된, YelpChi 데이터셋(.mat 포맷)용 학습 코드.

---

## 데이터 준비

CARE-GNN 또는 PC-GNN repo에서 `YelpChi.mat`을 받아 다음 위치에 둡니다:

```
yelchi/data/YelpChi.mat
```

`YelpChi.mat` 필수 키:
- `features` : sparse (N, 32) — node feature
- `label`    : (N,) — {0, 1}, 1 = fraud
- `net_rur`  : sparse (N, N) — review-user-review adjacency
- `net_rtr`  : sparse (N, N) — review-time-review adjacency
- `net_rsr`  : sparse (N, N) — review-star-review adjacency

다운로드 예시:
```bash
git clone https://github.com/YingtongDou/CARE-GNN.git
unzip CARE-GNN/data/YelpChi.zip -d yelchi/data/
```

---

## 실행

```bash
python run_all_yelchi.py                      # 7개 모델 전부
python run_all_yelchi.py --only cage_carerf   # 1개만
python run_all_yelchi.py --mat-path /path/to/YelpChi.mat
python run_all_yelchi.py --epochs 100
```

또는 모델 1개 직접:
```bash
python -m yelchi.src.train --model cage_carerf --mat-path yelchi/data/YelpChi.mat
```

---

## 학습 모델 7종

| # | 모델 | 설명 |
|---|---|---|
| 1 | MLP | graph-free baseline |
| 2 | GCN | 3 relation union edge |
| 3 | GAT | 동일 (multi-head attention) |
| 4 | GraphSAGE | 동일 (inductive) |
| 5 | **CAGE-CareRF** | 본 연구 모델 (Skip + Gating + Aux + CARE) |
| 6 | CAGE-CareRF w/o CARE | CARE filter ablation |
| 7 | CAGE-CareRF w/o Aux | Aux loss ablation |

YelpChi는 review-level node + 3 relation (R-U-R, R-T-R, R-S-R) — YelpZip의 기본 relation과 동일한 의미 (단 본 데이터는 이미 CARE-GNN repo에서 처리되어 있음).

---

## 산출물

```
yelchi/outputs/
├── metrics_mlp.json
├── metrics_gcn.json
├── metrics_gat.json
├── metrics_graphsage.json
├── metrics_cage_carerf.json           ← FINAL
├── metrics_cage_carerf_no_care.json
└── metrics_cage_carerf_no_aux.json
```

각 JSON 구조:
```json
{
  "dataset": "yelchi",
  "model": "cage_carerf",
  "best_threshold": 0.42,
  "valid_metrics": {"pr_auc": ..., "macro_f1": ..., "roc_auc": ..., ...},
  "test_metrics":  {"pr_auc": ..., "macro_f1": ..., "roc_auc": ..., ...}
}
```

---

## 코드 구조

```
yelchi/
├── data/YelpChi.mat             ← 사용자가 추가
├── configs/default.yaml
├── src/
│   ├── data_loader.py           (.mat → torch + train/valid/test split)
│   ├── models.py                (MLP/GCN/GAT/SAGE/CAGE-CareRF + Focal+Aux loss)
│   ├── train.py                 (학습 + threshold@valid + test 1회 평가)
│   └── metrics.py
└── outputs/

run_all_yelchi.py                ← repo 루트 (7모델 launcher)
```

YelpZip 메인 파이프라인과 **완전히 분리**.
