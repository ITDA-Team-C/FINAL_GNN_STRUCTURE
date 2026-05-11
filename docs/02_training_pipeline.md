# 02. 학습 파이프라인

raw CSV에서 시작해 14개 모델(baseline 4 + CAGE-RF 4 + CAGE-CareRF 1 + ablation 5)을 학습·평가하기까지의 **전체 흐름**.

---

## 1. 7단계 파이프라인 개요

```text
[1] load_yelpzip       raw CSV 로딩 + EDA              → data/interim/raw_data.csv
[2] label_convert      -1→1, 1→0 라벨 변환             → data/interim/labeled_data.csv
[3] sampling           hybrid dense + 64/16/20 split   → data/processed/sampled_reviews.csv
[4] feature_engineering TF-IDF/SVD/Scaler train-only fit → data/processed/features.npy (50000, 140)
[5] build_relations    6개 relation edge 생성            → data/processed/edge_index_dict.pt
[6] relation_quality   train labels로 품질 분석          → outputs/metrics/relation_quality.json
[7] train              [optional CARE] → 학습 → threshold@valid → test 평가 → metrics 저장
```

각 단계는 **독립 실행 가능**하며 표준 명령은 `python -m src.<package>.<module>`.

---

## 2. 단계별 명령어 & 입출력

### Step 1: `load_yelpzip`
```bash
python -m src.preprocessing.load_yelpzip
```
- 입력: `data/raw/yelp_zip.csv` (608,458 rows)
- 출력: `data/interim/raw_data.csv`, `raw_eda.txt`
- 자동 처리: `review_id` 컬럼 없으면 `np.arange(len(df))`로 생성

### Step 2: `label_convert`
```bash
python -m src.preprocessing.label_convert
```
- 입력: `data/interim/raw_data.csv`
- 출력: `data/interim/labeled_data.csv`
- 검증: `set(df.label.unique()) == {0, 1}` assert

### Step 3: `sampling` (Graph-Signal Preserving Hybrid Dense Sampling)
```bash
python -m src.preprocessing.sampling
```
- 알고리즘:
  1. `top_products` = 리뷰 수 상위 max(100, N/1000) 개 product
  2. `top_users` = 리뷰 수 상위 max(100, N/1000) 명 user
  3. `top_months` = 리뷰 집중 상위 max(12, N/50000) 개 month
  4. `union = product_mask | user_mask | month_mask`
  5. `union` 크기가 max_nodes(50,000)를 초과하면 그 안에서 무작위 reduce
  6. 부족하면 무작위 보충 (min_nodes=10,000)
  7. Stratified 64/16/20 split, random_state=42
- 출력: `data/processed/sampled_reviews.csv` (split 컬럼 포함), `sampling_stats.txt`
- **현재 결과**: 50,000 sample / fraud_ratio 11.16% / Train 32k · Valid 8k · Test 10k

### Step 4: `feature_engineering` (TRAIN-only fit)
```bash
python -m src.preprocessing.feature_engineering
```
- TF-IDF(50k vocab, ngram=(1,2)) → 32k train docs에서만 fit, 50k 전체에 transform
- TruncatedSVD(128D) → train 행에서만 fit
- numeric 12개 feature (rating_norm, review_length(_log), user_*, product_*, days_since_first_review, month_sin/cos)
- StandardScaler → train에서만 fit
- 출력:
  - `data/processed/features.npy` shape `(50000, 140)`
  - `data/processed/node_samples.csv`
  - `data/processed/feature_meta.json` (`fit_scope: train_only` 박제)

### Step 5: `build_relations`
```bash
python -m src.graph.build_relations
```
- 입력: `node_samples.csv` + `features.npy`
- 6개 relation 호출:
  - `build_rur` (같은 user, top-k=10)
  - `build_rtr` (같은 prod + 같은 month, top-k=10)
  - `build_rsr` (같은 prod + 같은 rating, top-k=10)
  - `build_burst` (|Δdate|≤7d & |Δrating|≤1)
  - `build_semsim` (같은 prod 내 SVD-128 cosine top-5)
  - `build_behavior` (user behavior cosine → user pair → review pair, max 3 reviews/user)
- 모든 edge를 무방향화(`convert_to_undirected`) 후 저장
- 출력: `edge_index_*.pt` 6개 + `edge_index_dict.pt` + `graph_meta.json`
- **현재 결과**: rur 49,754 · rtr 87,228 · rsr 597,432 · burst 33,672 · semsim 330,132 · behavior 550,136

### Step 6: `relation_quality`
```bash
python -m src.graph.relation_quality
```
- relation별 quality 계산 (**train mask label만 사용** → leakage-safe):
  - edge_count, avg_degree, isolated_ratio
  - fraud_fraud_ratio, normal_normal_ratio, fraud_normal_ratio
  - fraud_edge_lift = fraud_fraud_ratio / (train_fraud_ratio²)
- 출력: `outputs/metrics/relation_quality.json`, `outputs/reports/relation_quality.csv`
- **현재 결과 (train fraud_ratio=0.1116)**:
  | rel | edges | iso | FF | FN | lift |
  |---|---|---|---|---|---|
  | rur | 49754 | 0.653 | 0.0157 | 0.0040 | 1.26 |
  | rtr | 87228 | 0.375 | 0.0211 | 0.2040 | 1.69 |
  | rsr | 597432 | 0.070 | 0.0139 | 0.1780 | 1.12 |
  | burst | 33672 | 0.631 | 0.0244 | 0.1892 | **1.96** |
  | semsim | 330132 | 0.026 | 0.0139 | 0.1735 | 1.12 |
  | behavior | 550136 | 0.061 | 0.0091 | 0.1468 | 0.73 |

### Step 7: `train` (학습 + 평가)
```bash
python -m src.training.train --model <name> --config <yaml>
```
- 자동 흐름:
  1. `check_and_preprocess`: 필요한 파일 없으면 Step 1~5 자동 호출 (스킵 가능)
  2. `load_graph_data`: features.npy + edge_index_dict.pt + train/valid/test mask
  3. `[Optional] CARE filter`: `config.care_filter.enabled` 면 `filter_edge_index_dict` 적용, `filter_log.json` 저장
  4. `create_model`: argparse `--model`에 따라 baseline/CAGE-RF/CAGE-CareRF 생성
  5. `calculate_pos_weight + create_loss_fn`: Focal+Aux 또는 WeightedBCE
  6. 학습 루프: epoch마다 train loss + (validation_interval 마다) valid metrics
  7. `find_best_threshold`: valid PR-curve에서 macro_f1 최대값 위치
  8. test 평가 (test set은 **이 단계에서 1회만** 사용)
  9. 저장: `metrics_<version>.json`, `best_model_<model>.pt`, `report_<version>.html`

---

## 3. 14개 모델 실행 명령

### A. Baseline 4종 (200 epoch, **edge = union of 6 relations**)
```bash
python -m src.training.train --model mlp        --config configs/default.yaml
python -m src.training.train --model gcn        --config configs/default.yaml
python -m src.training.train --model gat        --config configs/default.yaml
python -m src.training.train --model graphsage  --config configs/default.yaml
```

### B. CAGE-RF 계열 4종
```bash
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml           # Base (no skip)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml           # Skip (v8)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml       # Refine (v9 Two-Stage)
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml # + CARE filter
```

### C. CAGE-CareRF (최종 제안 모델)
```bash
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf.yaml
```

### D. Ablation 5종
```bash
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_care.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_custom.yaml
```

---

## 4. Loss 구성

```text
loss = FocalLoss(α=0.75, γ=2.0)
     + aux_weight (0.3) × Σ_r BCE(aux_logit_r, y_r) / R         # use_aux_loss=true 일 때만
```

- `WeightedBCELoss`도 선택 가능 (`loss.type: weighted_bce`)
- (선택) gate entropy regularization은 plan에 있지만 현재 train.py에는 미포함 — 본 작업 후속에 옵션 추가 가능
- **클래스 불균형 대응 = Focal + class_weight + threshold tuning** (PC-GNN sampler 없음)

---

## 5. Leakage 차단 메커니즘 (3중 방어)

1. **Feature fit**: `feature_engineering.py`가 `split=='train'` 행에서만 TF-IDF/SVD/Scaler fit. `feature_meta.json`에 `fit_scope: train_only` 박제.
2. **Relation quality**: `relation_quality.py`가 `train_mask`로만 fraud/normal set 구성, valid/test 라벨은 ratio 분모/분자에서 제외.
3. **Threshold 결정**: `threshold.py` + `find_best_threshold`가 **valid mask**에서만 PR-curve 탐색. test에는 valid threshold 그대로 적용.
4. **Test 평가 1회**: `evaluate.py`가 `split == 'test'` mask에서 metrics 계산을 **본 학습 종료 후 단 한 번**만 호출.
5. **CARE filter**: feature cosine만 사용, 라벨 미참조.

---

## 6. CARE filter offline vs inline

```yaml
# configs/cage_carerf.yaml
care_filter:
  enabled: true
  apply: offline   # ← 학습 시작 전에 edge_index_dict를 한 번만 필터링
  top_k:
    rur: 10
    ...
```

- **offline (기본)**: train.py가 학습 루프 시작 전 `filter_edge_index_dict()` 1회 호출 → `edge_index_dict` 갱신. **빠르고 GPU 메모리 부담 적음.** 본 작업에서 사용.
- **inline (실험적)**: `cage_carerf_gnn.py`의 `care_inline=True`로 두면 매 forward마다 필터링. node feature가 epoch마다 변하지 않으므로 의미 없음. 사용하지 않음.

---

## 7. 학습 산출물 (모델 1개당)

```text
outputs/cage_rf_gnn/
├── best_model_<model_name>.pt          # state_dict
├── metrics_<version>.json              # {valid_metrics, test_metrics, best_threshold}
└── report_<version>.html               # HTML 리포트
outputs/metrics/
├── filter_log.json                     # CARE 활성화 시 (마지막 실행 덮어쓰기)
└── relation_quality.json               # 1회 (Step 6 산출)
```

`metrics_*.json` 스키마:
```json
{
  "model": "cage_carerf_gnn",
  "best_threshold": 0.46,
  "valid_metrics": {"pr_auc": ..., "macro_f1": ..., ...},
  "test_metrics":  {"pr_auc": ..., "macro_f1": ..., ...}
}
```

---

## 8. Train 루프 의사코드

```python
config = load_config(yaml)
device = "cuda" if torch.cuda.is_available() else "cpu"

x, y, edge_index_dict, train/valid/test_mask = load_graph_data(config)

if config.care_filter.enabled and apply == "offline":
    edge_index_dict, _ = filter_edge_index_dict(x, edge_index_dict, top_k, min_sim,
                                                 log_path="outputs/metrics/filter_log.json")

model = create_model(model_name, x.shape[1], config).to(device)
loss_fn = create_loss_fn(config, pos_weight, device)          # Focal + Aux 또는 WeightedBCE
optimizer = Adam(model.parameters(), lr=config.training.lr)

best_valid_f1, best_state, patience = 0, None, 0
for epoch in range(num_epochs):
    loss = train_epoch(model, x, y, edge_index_dict, train_mask, optimizer, loss_fn, device,
                       oversample_ratio=config.training.oversample_ratio,
                       hard_mining_ratio=config.training.hard_mining_ratio)
    if epoch % validation_interval == 0:
        valid_metrics = evaluate(..., valid_mask, ...)
        if valid_metrics["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_metrics["macro_f1"]
            best_state = copy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience: break

model.load_state_dict(best_state)
best_threshold, _ = find_best_threshold(y[valid], scores[valid], metric="macro_f1")
test_metrics = calculate_metrics(y[test], scores[test], scores[test] >= best_threshold)
save_json(test_metrics, "outputs/cage_rf_gnn/metrics_<version>.json")
```

---

## 9. 학습 시간 예상 (CPU vs GPU)

| 모델 종류 | CPU (200 epoch) | GPU (200 epoch, 추정) |
|---|---|---|
| baseline (GCN/GAT/SAGE/MLP) | 30~90분 | 2~5분 |
| cage_rf_gnn_cheb (Base/Skip/Refine/+CARE) | 1~3시간 | 5~15분 |
| cage_carerf_gnn (6 branch CHEB + gating) | 1~3시간 | 5~15분 |
| Ablation 5종 | 위와 동일 | 위와 동일 |
| **전체 14개** | **하루 이상** | **2~4시간** |

CHEB K=3 + 6 branch + 50k node는 forward 1회당 GPU에서 ~수십 ms, CPU에서 ~수 초.

---

## 10. 자주 쓰는 검수 (sanity check)

```python
# label은 {0,1}만
assert set(df.label.unique()) <= {0, 1}

# feature fit_scope 박제 확인
import json
m = json.load(open("data/processed/feature_meta.json"))
assert m["fit_scope"] == "train_only"

# threshold 출처 확인
m = json.load(open("outputs/cage_rf_gnn/metrics_cage_carerf_v1.json"))
assert "best_threshold" in m

# CARE filter 적용 흔적
log = json.load(open("outputs/metrics/filter_log.json"))
assert "feature-cosine" in log["meta"]["note"]
```
