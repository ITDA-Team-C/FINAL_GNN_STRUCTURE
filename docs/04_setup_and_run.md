# 04. Setup & Run

GPU/CPU 서버에서 본 repo를 처음 받아 학습까지 가는 단계별 가이드.

---

## 1. Clone

```bash
git clone https://github.com/ITDA-Team-C/FINAL_GNN_STRUCTURE.git
cd FINAL_GNN_STRUCTURE
```

---

## 2. Python 환경

권장: Python 3.11+

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# 또는 Windows: .venv\Scripts\activate
```

---

## 3. 의존성 설치

### 3.1 핵심 패키지
```bash
pip install -r requirements.txt
```

### 3.2 torch-geometric 가속 라이브러리 (GPU 학습 권장)

`torch-geometric` 단독으로는 ChebConv 등 일부 모듈이 느립니다. CUDA/torch 버전에 맞춰 `pyg-lib`, `torch-scatter`, `torch-sparse`를 설치합니다.

GPU (예: CUDA 12.1):
```bash
pip install pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.11.0+cu121.html
```

CPU 전용:
```bash
pip install pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

본인 CUDA 버전 확인:
```bash
nvidia-smi | head -3
```

---

## 4. 데이터 배치

YelpZip 원본 CSV를 다음 경로에 둡니다 (repo에는 포함되지 않음):

```text
data/raw/yelp_zip.csv
```

기대 컬럼:
```
user_id, prod_id, rating, label, date, text, tag
```

- `review_id`는 없어도 됨 (자동 생성)
- 라벨: 사기 = -1, 정상 = 1 (코드에서 자동 변환)
- 인코딩: UTF-8 권장

컬럼명이 다르다면 `src/preprocessing/load_yelpzip.py:33-46`의 `validate_columns()` 또는 column rename 로직을 한 줄 추가하세요.

---

## 5. 전처리 + 그래프 빌드 (1회만)

```bash
python -m src.preprocessing.load_yelpzip       # raw → interim/raw_data.csv + EDA
python -m src.preprocessing.label_convert      # -1→1, 1→0
python -m src.preprocessing.sampling           # hybrid dense → 25k~50k 노드 + 64/16/20 split
python -m src.preprocessing.feature_engineering # TF-IDF/SVD/Scaler train-only fit → (N, 140)
python -m src.graph.build_relations            # 6 relation edge_index_dict.pt
python -m src.graph.relation_quality           # outputs/metrics/relation_quality.json
```

예상 시간 (CPU 기준):
- load_yelpzip: ~5초
- label_convert: ~10초
- sampling: ~30초
- feature_engineering: ~1~2분
- build_relations: ~3~10분 (semsim/behavior cosine이 가장 무거움)
- relation_quality: ~30초

산출물:
```text
data/processed/
├── sampled_reviews.csv
├── sampling_stats.txt
├── features.npy           # (N, 140)
├── node_samples.csv
├── feature_meta.json      # fit_scope: train_only 박제
├── edge_index_{rur,rtr,rsr,burst,semsim,behavior}.pt
├── edge_index_dict.pt
└── graph_meta.json
outputs/
├── metrics/relation_quality.json
└── reports/relation_quality.csv
```

---

## 6. 학습 (14개 모델)

`docs/02_training_pipeline.md` Section 3과 동일.

빠른 검증을 위해 한 모델만 먼저 (CPU에서도 6 epoch면 5분 이내):
```bash
python -c "
import yaml
with open('configs/cage_carerf.yaml', encoding='utf-8') as f: c = yaml.safe_load(f)
c['training']['num_epochs'] = 6
c['training']['validation_interval'] = 2
with open('configs/cage_carerf_smoke.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(c, f, allow_unicode=True)
"
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf_smoke.yaml
```

본 실험 (200 epoch, GPU 추천):
```bash
# A. Baseline 4종
python -m src.training.train --model mlp        --config configs/default.yaml
python -m src.training.train --model gcn        --config configs/default.yaml
python -m src.training.train --model gat        --config configs/default.yaml
python -m src.training.train --model graphsage  --config configs/default.yaml

# B. CAGE-RF 계열
python -m src.training.train --model cage_rf_gnn_cheb --config configs/default.yaml
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v8_skip.yaml
python -m src.training.train --model cage_rf_gnn_cheb --config configs/v9_twostage.yaml
python -m src.training.train --model cage_rf_gnn_cheb --config configs/cage_rf_skip_care.yaml

# C. CAGE-CareRF (최종)
python -m src.training.train --model cage_carerf_gnn --config configs/cage_carerf.yaml

# D. Ablation
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_care.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_skip.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_gating.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_aux.yaml
python -m src.training.train --model cage_carerf_gnn --config configs/ablation_no_custom.yaml
```

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `ImportError: torch_geometric` | PyG 미설치 | §3.2 참조 |
| ChebConv 매우 느림 (CPU에서 epoch 당 수분) | pyg-lib/torch-sparse 미설치 | §3.2 (GPU 가속) |
| `UnicodeDecodeError: 'cp949'` (Windows) | yaml load 인코딩 | `open(yaml_path, encoding='utf-8')` 적용 (코드에는 이미 반영) |
| `KeyError: 'review_id'` 등 컬럼 누락 | yelp_zip.csv 스키마 불일치 | `load_yelpzip.py`에 컬럼 rename 추가 |
| `CUDA out of memory` (50k node) | hidden_dim/num_layers 큼 | config `model.hidden_dim` 64로 낮춤 |
| `df.sample()` 무작위 reduce 경고 | union이 max_nodes 초과 | 정상 동작. 보고서에 명시 |

---

## 8. 재현성 체크리스트

- [ ] `random_state=42` 고정 (`src/utils/seed.py`)
- [ ] `feature_meta.json["fit_scope"] == "train_only"`
- [ ] `outputs/metrics/relation_quality.json["meta"]["note"]`에 train-only 명시
- [ ] threshold는 valid에서만 결정 (`src/training/threshold.py`, `train.py:find_best_threshold`)
- [ ] test set은 1회만 평가 (`train.py` 마지막 단계)
- [ ] `configs/cage_carerf.yaml`의 `sampler.enabled: false`
