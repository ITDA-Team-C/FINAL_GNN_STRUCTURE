# Amazon Fraud Detection (CARE-GNN / PC-GNN format)

YelpZip 메인 파이프라인과 독립된, Amazon 데이터셋(.mat 포맷)용 학습 코드.

---

## 데이터 준비

CARE-GNN 또는 PC-GNN repo에서 `Amazon.mat`을 받아 다음 위치에 둡니다:

```
amazon/data/Amazon.mat
```

`Amazon.mat` 필수 키:
- `features` : sparse (N, 25) — node feature
- `label`    : (N,) — {0, 1}, 1 = fraud
- `net_upu`  : sparse (N, N) — user-product-user adjacency
- `net_usu`  : sparse (N, N) — user-star-user adjacency
- `net_uvu`  : sparse (N, N) — user-vote-user adjacency

다운로드 예시:
```bash
# CARE-GNN repo에서 받기
git clone https://github.com/YingtongDou/CARE-GNN.git
unzip CARE-GNN/data/Amazon.zip -d amazon/data/
```

---

## 실행

```bash
# 7개 모델 한 번에
python run_all_amazon.py

# 모델 1개만
python run_all_amazon.py --only cage_carerf

# 특정 mat 경로
python run_all_amazon.py --mat-path /any/path/Amazon.mat

# 짧게 epochs
python run_all_amazon.py --epochs 100
```

또는 모델 1개 직접:
```bash
python -m amazon.src.train --model cage_carerf --mat-path amazon/data/Amazon.mat
```

---

## 학습 모델 7종

| # | 모델 | 설명 |
|---|---|---|
| 1 | MLP | graph-free 텍스트 baseline |
| 2 | GCN | 3 relation의 union edge |
| 3 | GAT | 동일 (multi-head attention) |
| 4 | GraphSAGE | 동일 (inductive aggregation) |
| 5 | **CAGE-CareRF** | 본 연구 모델 (Skip + Gating + Aux + CARE) |
| 6 | CAGE-CareRF w/o CARE | CARE filter 제거 ablation |
| 7 | CAGE-CareRF w/o Aux | Aux loss 제거 ablation |

YelpZip의 6 relation 구조와 달리 Amazon은 3 relation(UPU/USU/UVU)이라 Lean-4/5/6 같은 변종은 적용 불가.

---

## 산출물

```
amazon/outputs/
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
  "dataset": "amazon",
  "model": "cage_carerf",
  "best_threshold": 0.42,
  "valid_metrics": {"pr_auc": ..., "macro_f1": ..., "roc_auc": ..., ...},
  "test_metrics":  {"pr_auc": ..., "macro_f1": ..., "roc_auc": ..., ...}
}
```

---

## 코드 구조

```
amazon/
├── data/Amazon.mat              ← 사용자가 추가
├── configs/default.yaml
├── src/
│   ├── data_loader.py           (.mat → torch + train/valid/test split)
│   ├── models.py                (MLP/GCN/GAT/SAGE/CAGE-CareRF + Focal+Aux loss)
│   ├── train.py                 (학습 루프 + threshold@valid + test 1회 평가)
│   └── metrics.py
└── outputs/                     (실행 시 자동 생성)

run_all_amazon.py                ← repo 루트 (7모델 launcher)
```

YelpZip 메인 파이프라인(`src/`, `configs/`, `data/raw/`)과는 **완전히 분리**되어 있으며, 코드 공유 없음.
