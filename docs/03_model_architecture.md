# 03. 모델 구조

본 예선 제안 모델은 **CAGE-CareRF-Lean GNN**이며, 비교 모델 **CAGE-CareRF v1 (with Gating + Custom Relations)** 및 baseline 들의 구조 명세를 함께 기록한다.

## 0. FINAL = Lean 결정 근거 (Ablation 기반)

200-epoch 학습 결과 모든 14개 모델을 비교했을 때, Gated Relation Fusion과 Custom Relations(R-Burst-R / R-SemSim-R / R-Behavior-R)이 PR-AUC와 Macro-F1 모두에서 음(-)의 marginal 효과를 보였다 (`w/o Gating` PR-AUC +0.011, `w/o Custom` PR-AUC +0.007 vs FINAL). 반면 Auxiliary branch loss는 강한 양(+)의 기여(PR-AUC -0.043 if removed)를, CARE filter와 Skip Connection은 미미한 양(+)의 기여를 보였다. 따라서 옵션 A에 따라 **Lean 변종**을 FINAL로 채택하고, 기존 6-relation + Gating 모델은 해석 가능성 비교 모델로 유지한다.

### FINAL (Lean) 구성
- ✅ SkipChebBranch (Skip Connection)
- ✅ CARE Neighbor Filter (offline, label-free)
- ✅ Auxiliary Branch Loss (Focal main + λ·aux)
- ✅ 기본 relation 3개 (R-U-R, R-T-R, R-S-R)
- ❌ Gated Relation Fusion → **Mean Fusion (단순 평균)**
- ❌ Custom Relations 3개 → 보고서에 "신호는 존재하지만 (`fraud_edge_lift` R-Burst-R 1.96), 모델 통합 단계에서 충분히 활용되지 못함"으로 분석

---

## 1. CAGE-CareRF GNN 전체 구조

```text
Input:  x ∈ R^(N × 140)         # N=50,000, 140 = 128 SVD + 12 numeric
        edge_index_dict = {rur, rtr, rsr, burst, semsim, behavior}

      ┌───────────────────────────────────────────────────────────┐
      │  [Offline] CARE Neighbor Filter (label-free, train.py 단계) │
      │  per relation: cosine(x_src, x_dst) top-k 유지              │
      │  before/after edge count → outputs/metrics/filter_log.json  │
      └───────────────────────────────────────────────────────────┘
                              │
                              ▼   (filtered edge_index_dict)
   ┌──────────────────────────────────────────────────────────────┐
   │   Relation-wise Skip GNN Branch (6 branches in parallel)      │
   │   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
   │   │ SkipCheb   │ │ SkipCheb   │ │ SkipCheb   │ │ SkipCheb   │ │
   │   │  (rur)     │ │  (rtr)     │ │  (rsr)     │ │  (burst)   │ │
   │   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ │
   │         │              │              │              │       │
   │   ┌─────┴──────┐  ┌────┴───────┐                             │
   │   │ SkipCheb   │  │ SkipCheb   │                             │
   │   │  (semsim)  │  │  (behavior)│                             │
   │   └─────┬──────┘  └─────┬──────┘                             │
   │         │               │                                    │
   │   각 branch 출력: h_r ∈ R^(N × 128)                            │
   └─────────┴───────┴───────┴───────┴───────┴────────────────────┘
                              │
                              ▼   (stack → (N, 6, 128))
   ┌──────────────────────────────────────────────────────────────┐
   │   Gated Relation Fusion (per-node softmax)                    │
   │   α = softmax_R( MLP(h_r) )      shape (N, 6, 1)              │
   │   h_fused = Σ_r αᵣ · h_r          shape (N, 128)              │
   │   (last_alpha 저장 → 보고서/해석)                                │
   └─────────────────────────────┬────────────────────────────────┘
                                 ▼
                         ┌─────────────┐
                         │ Projection  │  Linear(128→128) + ReLU + Dropout
                         └──────┬──────┘
                                ▼
                         ┌─────────────┐
                         │ Classifier  │  Linear→ReLU→Dropout→Linear(→1)
                         └──────┬──────┘
                                ▼
                          main_logit  ∈ R^N

   Auxiliary branch heads (use_aux_loss=True 일 때만):
       aux_logit_r = Linear_r(h_r) ∈ R^N    for r ∈ {6 relations}

   Loss = FocalLoss(main_logit, y) + 0.3 × (1/6) Σ_r BCE(aux_logit_r, y)
```

핵심 책임 분리:

| 모듈 | 파일 | 책임 |
|---|---|---|
| `CARENeighborFilter` | `src/filtering/care_neighbor_filter.py` | feature cosine top-k 필터링 (train.py가 offline 호출) |
| `SkipChebBranch` | `src/models/skip_cheb_branch.py` | per-relation 임베딩 with residual skip |
| `GatedRelationFusion` | `src/models/gated_relation_fusion.py` | per-node softmax α + entropy regularizer |
| `CAGECareRF_GNN` | `src/models/cage_carerf_gnn.py` | 위 3개 + aux heads + main classifier 통합 |
| `FocalLoss`/`AuxiliaryLoss` | `src/models/losses.py` | imbalance 대응 loss |

---

## 2. CARE Neighbor Filter (offline, label-free)

`src/filtering/care_neighbor_filter.py`

### 의도

같은 relation으로 묶인 이웃이 항상 신호를 주는 것은 아니다. fraud-normal mixed edge는 GNN aggregation을 흐릴 수 있어, CARE-GNN의 camouflage-resistant 관점을 차용한다.

### 알고리즘

```text
Input:  x ∈ R^(N × F), edge_index ∈ Z^(2 × E), top_k, [min_sim]

1. x_norm = L2_normalize(x, dim=1)
2. sim    = (x_norm[src] · x_norm[dst]).sum(dim=1)              # E-dim
3. (optional) drop edges where sim < min_sim
4. group edges by src node (sort stable + run-length)
5. per src: 만일 cnt > top_k 라면 sim 기준 top-k indices만 유지
6. return filtered edge_index
```

### 불변량 (leakage-safety)
- **라벨을 절대 참조하지 않음** — node feature x만 사용
- 모든 노드(train/valid/test) feature를 사용하지만 transductive 가정에서 허용
- log: relation별 before/after edge_count, isolated_ratio_after를 `outputs/metrics/filter_log.json` 저장

### 본 실험 결과 (top_k = {rur:10, rtr:10, rsr:10, burst:10, semsim:5, behavior:5})
| relation | before | after | kept | iso_after |
|---|---|---|---|---|
| rur | 49,754 | 48,524 | 97.5% | 0.653 |
| rtr | 87,228 | 85,752 | 98.3% | 0.375 |
| rsr | 597,432 | 389,214 | 65.2% | 0.070 |
| burst | 33,672 | 33,569 | 99.7% | 0.631 |
| semsim | 330,132 | 230,488 | 69.8% | 0.026 |
| behavior | 550,136 | 234,845 | **42.7%** | 0.061 |

R-Behavior-R이 가장 많이 reduce — fraud_edge_lift=0.73(노이즈 신호와 부합).

---

## 3. SkipChebBranch (relation 1개에 대한 GNN backbone)

`src/models/skip_cheb_branch.py`

### 의도

깊은 GNN에서 over-smoothing 완화 + 1-hop/multi-hop 정보 보존.

### 구조

```text
hidden_dim = 128, num_layers = 3, K (Cheb polynomial order) = 3, dropout = 0.3

Layer 0:
    h_0 = input_projection(x) if input_dim != hidden_dim else x      # shape (N, 128)
    out_0 = ReLU( Dropout( ChebConv(x, edge_index) ) )                # (N, 128)
    h_prev = out_0

Layer l in [1, num_layers-1]:
    conv_l = ChebConv(h_prev, edge_index)                            # (N, 128)
    out_l  = conv_l + h_prev                                          # residual add
    out_l  = ReLU(Dropout(out_l))                                     # (N, 128)
    h_prev = out_l

return h_prev
```

### 왜 residual add?
- Layer 1부터 residual은 conv output과 같은 hidden_dim → 단순 elementwise add
- Layer 0은 dim mismatch 시 `input_projection` 으로 맞춤
- 본 모델은 layer 0에서 conv 결과만 사용하고, layer 1+ 부터 skip 적용 (구현상의 단순화)

### ablation: `use_skip = false`
- `_PlainChebBranch` (skip 없는 일반 ChebConv 스택)을 사용 (`cage_carerf_gnn.py` 안 정의)

---

## 4. GatedRelationFusion (per-node softmax α)

`src/models/gated_relation_fusion.py`

### 의도

6개 relation의 중요도는 node마다 다를 수 있다. 단순 평균이 아니라 learnable gating으로 노드별 기여도를 학습한다.

### 구조

```text
Input:  relation_stack ∈ R^(N × R × H)              # R=6, H=128
        gate_mlp = Linear(H→H) → ReLU → Linear(H→1)

For each (n, r):
    logit_{n,r} = gate_mlp(h_{n,r})                # scalar
α = softmax over R axis: α ∈ R^(N × R × 1)
fused = (α * relation_stack).sum(dim=R)            # (N, H)
return fused, α
```

### `gate_entropy_regularizer` (선택)
```python
H(α_n) = -Σ_r α_{n,r} log α_{n,r}
loss_reg = mean_n H(α_n)
```
α가 한 relation에 과도하게 집중되지 않도록 entropy를 maximize하는 방향으로 사용 가능 (현재 train.py에는 미포함, 후속).

### ablation: `use_gating = false`
- fusion 대신 `torch.cat(embeddings, dim=1)` → shape `(N, R·H)` → projection input 차원이 `R·H`로 자동 조정

---

## 5. CAGECareRF_GNN (전체 모델 클래스)

`src/models/cage_carerf_gnn.py`

### 초기화 시그니처

```python
class CAGECareRF_GNN(nn.Module):
    def __init__(
        self,
        input_dim,           # 140
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        K=3,
        active_relations=None,    # None → 6개 모두. ablation_no_custom에서 ["rur","rtr","rsr"]
        use_skip=True,             # SkipChebBranch vs _PlainChebBranch
        use_gating=True,           # GatedRelationFusion vs concat fallback
        use_aux_loss=True,         # aux_heads 생성 여부
        care_inline=False,         # 보통 offline 사용
        care_top_k=10,
        care_min_sim=None,
    )
```

### Forward

```python
def forward(self, x, edge_index_dict):
    embeddings = []
    for rel in self.active_relations:
        ei = edge_index_dict[rel]
        if self.care_inline:                                   # 보통 False
            ei = filter_edges_by_feature_similarity(x, ei, top_k=self.care_top_k)
        h = self.branches[rel](x, ei)                          # SkipChebBranch
        embeddings.append(h)

    relation_stack = torch.stack(embeddings, dim=1)            # (N, R, H)

    if self.use_gating:
        fused, alpha = self.fusion(relation_stack)             # (N, H), (N, R, 1)
        self.last_alpha = alpha                                # 보고서/대시보드 후속용
    else:
        fused = torch.cat(embeddings, dim=1)                   # (N, R·H)

    h_proj = self.projection(fused)                            # (N, 128)
    logit  = self.classifier(h_proj).squeeze(-1)                # (N,)

    aux_logits = None
    if self.use_aux_loss:
        aux_logits = {rel: aux_heads[rel](h_rel).squeeze(-1)
                      for rel, h_rel in zip(active_relations, embeddings)}

    return logit, aux_logits                                   # tuple (필수)
```

### `get_relation_contribution()`
- 학습 후 평균 α를 (R,) ndarray로 반환 — 보고서의 "relation contribution 표" 용

---

## 6. Loss 구성

`src/models/losses.py`

### FocalLoss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):   # config: 0.75 / 2.0
        ...
    def forward(self, logits, targets):
        p  = sigmoid(logits)
        pt = where(targets == 1, p, 1 - p)
        α  = where(targets == 1, alpha, 1 - alpha)
        ce = BCE_with_logits(logits, targets, reduction="none")
        return (α * (1 - pt)**γ * ce).mean()
```

### AuxiliaryLoss
```python
class AuxiliaryLoss(nn.Module):
    def __init__(self, main_loss_fn, aux_weight=0.3):
        ...
    def forward(self, main_logit, targets, aux_logits_dict=None):
        main = main_loss_fn(main_logit, targets)
        if not aux_logits_dict: return main
        aux  = mean_r BCE_with_logits(aux_logit_r, targets.float())
        return main + aux_weight * aux
```

### `class_weight: auto`
- `train.py.calculate_pos_weight`가 train_mask에서 `neg/pos` 비율 자동 계산. WeightedBCE 사용 시 적용.

---

## 7. CAGE-RF vs CAGE-CareRF 차이

| 항목 | CAGE-RF (`cage_rf_gnn_cheb.py`) | **CAGE-CareRF** (`cage_carerf_gnn.py`) |
|---|---|---|
| Skip connection | optional (`use_skip_connection` flag) — branch 내부 inline 구현 | **분리 모듈 `SkipChebBranch` 사용** (필수, ablation으로 off) |
| Gating fusion | inline `RelationGate` 클래스 | **분리 모듈 `GatedRelationFusion`** |
| CARE filter | ❌ | **✅ (offline, train.py가 적용)** |
| Aux heads | 항상 생성 | `use_aux_loss` flag로 on/off |
| Two-Pass | `use_two_stage` flag로 inline (v9) | (현재 미포함, plan §6.2 후속) |

---

## 8. Baseline 모델

### `baseline_mlp.py` (graph-free 비교용)
- 단순 MLP(`140 → 256 → 256 → 1`)
- node feature만 사용, edge_index_dict 무시

### `baseline_gcn.py` (union of 6 relations)
- `union_edge_index()` 헬퍼: 6개 edge_index를 cat → unique
- `GCNConv` × 3 layers, hidden_dim=128
- 첫 forward에서 union 계산 후 `_cached_edge_index`에 저장 (한 번만 계산)

### `baseline_gat.py` (union)
- `GATConv` × 3 layers, num_heads=8, head_dim=hidden_dim/num_heads
- multi-head attention으로 이웃 가중

### `baseline_graphsage.py` (union)
- `SAGEConv` × 3 layers, hidden_dim=128
- inductive aggregation (mean)

세 GNN baseline 모두 **classifier head = `Linear(128→128)→ReLU→Dropout→Linear(128→1)`** 로 통일.

---

## 9. PC-GNN sampler (예선 메인 제외)

플랜의 핵심 변경:

> 본 예선 모델에서는 PC-GNN의 label-balanced sampler를 메인 파이프라인에 포함하지 않는다. 대회 규정에서 이미 YelpZip 원본으로부터 subgraph를 먼저 샘플링한 뒤 train/valid/test를 분할하도록 요구하고 있어, 추가 sampler를 메인에 두면 샘플링 중복 또는 leakage 오해를 줄 수 있다. 클래스 불균형 문제는 Focal Loss + class weight + threshold tuning으로 완화하고, PC-GNN은 향후 보완 계획 또는 선택적 실험으로만 남긴다.

- `configs/cage_carerf.yaml` 의 `sampler.enabled: false` + `note` 명시
- `src/sampling/` 디렉토리는 비어 있음 (향후 보완 시 `pc_sampler.py` 추가)

---

## 10. 하이퍼파라미터 요약

```yaml
# configs/cage_carerf.yaml
model:
  hidden_dim: 128
  num_layers: 3
  K: 3
  dropout: 0.3
  use_skip: true
  use_gating: true
  use_aux_loss: true
  use_two_pass: false

training:
  batch_size: 1024              # full-batch GNN이므로 실제 사용은 평가에만
  learning_rate: 0.001
  num_epochs: 200
  early_stopping_patience: 20
  validation_interval: 5
  device: cuda

loss:
  type: focal
  focal_alpha: 0.75
  focal_gamma: 2.0
  aux_weight: 0.3
  gate_entropy_weight: 0.01     # (현재 train.py 미사용, 후속 옵션)
  class_weight: auto

care_filter:
  enabled: true
  apply: offline
  top_k: {rur:10, rtr:10, rsr:10, burst:10, semsim:5, behavior:5}
  min_sim: null
```

---

## 11. 학습 가능 파라미터 수 (대략)

- Per branch: ChebConv 3개 ≈ 3 × (140×128 + 128×128×K(3) + 128) ≈ 200k params
- 6 branches: ≈ 1.2M params
- GatedRelationFusion gate_mlp: 128×128 + 128×1 ≈ 16.5k
- projection + classifier: 128×128 + 128×1 ≈ 16.5k
- aux_heads (6): 128 × 6 ≈ 0.8k
- **총 ≈ 1.3M params** (140D input, hidden_dim 128 기준)

---

## 12. 해석 가능성 — Gating α

학습 후 `model.get_relation_contribution()` 호출 시 (R,) 벡터 반환.

보고서 예시 표 (가상):

| Relation | Mean α |
|---|---|
| R-Burst-R | 0.31 |
| R-T-R | 0.22 |
| R-SemSim-R | 0.18 |
| R-U-R | 0.12 |
| R-S-R | 0.10 |
| R-Behavior-R | 0.07 |

→ "단기 평판 폭격(Burst) + 시간 집중(R-T-R)이 fraud 판단의 주요 신호"라는 식의 해석.

본 실험에서 fraud_edge_lift 결과(rtr 1.69, burst 1.96, behavior 0.73)와 일관되는지 사후 검증 가능.

---

## 13. 그림 정리 — 한 페이지 요약

```text
        Reviews x (50000, 140)
              │
              ▼
   ┌──────────────────────┐
   │  edge_index_dict     │  rur, rtr, rsr, burst, semsim, behavior
   └──────────┬───────────┘
              │
   [offline] ▼
   CARE filter (cosine top-k, label-free)
              │
              ▼
   ┌─────────────────────────────────────────────┐
   │  SkipCheb-rur   SkipCheb-rtr   ...           │  6 branches in parallel
   │       │              │                       │  each: ChebConv × 3 + residual
   │       └──────┬───────┘                       │
   │              ▼                                │  shape (N, 6, 128)
   │   GatedRelationFusion (softmax α)            │  → h_fused (N, 128), α (N, 6, 1)
   └──────────────────────────────────────────────┘
              │
              ▼
       Projection + Classifier → main_logit
       Aux heads (per relation) → aux_logits dict

   Loss = Focal(main_logit, y) + 0.3 × mean_r BCE(aux_logit_r, y)
   threshold @ valid PR-curve  →  Test 1회 평가
   metrics: PR-AUC, Macro-F1, ROC-AUC, P, R, Acc
```
