import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed, save_json

set_seed(42)

CONFIG = {
    "processed_dir": "data/processed",
    "interim_dir": "data/interim",
    "input_file": "sampled_reviews.csv",
    # Text encoder, override at runtime via env TEXT_ENCODER:
    #   "sbert"        — Option A: frozen SBERT -> train-only SVD 128
    #   "tfidf"        — original TF-IDF -> train-only SVD 128
    #   "concat"       — TF-IDF->SVD128 ⊕ SBERT->SVD128, per-modality train-only
    #                    z-score, then a joint train-only SVD back to 128 (so the
    #                    downstream dim stays apples-to-apples with the two above).
    #   "sbert_proj"   — NEW: features.npy view = SBERT->SVD128 (same as 'sbert')
    #                    PLUS text_raw.npy = frozen SBERT 384D z-scored. The model
    #                    learns an end-to-end nn.Linear(384->128) projection on the
    #                    raw view; relation building / CARE use the fixed SVD-128
    #                    view unchanged.
    #   "concat_proj"  — NEW: features.npy view = concat→joint SVD-128 (same as
    #                    'concat') PLUS text_raw.npy = [SBERT 384 ⊕ TFIDF SVD128]
    #                    each train-only z-scored. The model learns an end-to-end
    #                    nn.Linear(512->128) projection on the raw view; relations
    #                    / CARE see the fixed view unchanged.
    "text_encoder": os.environ.get("TEXT_ENCODER", "sbert"),
    "sbert_model": os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2"),
    "sbert_batch_size": 256,
    "tfidf_max_features": 50000,
    "tfidf_min_df": 3,
    "tfidf_max_df": 0.9,
    "tfidf_ngram": (1, 2),
    "svd_components": 128,
    "random_state": 42,
}


# Train-only fit / valid·test transform-only.
# Numeric features computed from full df (sampled subgraph) — they are derived from
# observable graph signals, not labels, so they are leakage-safe.


def _train_mask(df):
    return df["split"].values == "train"


def extract_text_embedding(df):
    """Dispatch on CONFIG['text_encoder'].

    All branches end with a train-only-fit TruncatedSVD -> 128 dims for the
    features.npy view, so the downstream pipeline (build_relations[:, :128],
    CARE cosine, model input_dim) is byte-for-byte unchanged.

    The two *_proj variants additionally return a `text_raw` array which is
    saved to text_raw.npy and consumed by the model wrapper to learn an
    end-to-end nn.Linear projection.

    Returns: (text_embeddings_128, vectorizer_or_None, svd, text_raw_or_None)
    """
    encoder = CONFIG["text_encoder"].lower()
    if encoder == "tfidf":
        emb, vec, svd = _extract_tfidf_svd(df)
        return emb, vec, svd, None
    if encoder == "sbert":
        emb, vec, svd = _extract_sbert_svd(df)
        return emb, vec, svd, None
    if encoder == "concat":
        emb, vec, svd = _extract_concat_svd(df)
        return emb, vec, svd, None
    if encoder == "sbert_proj":
        return _extract_sbert_proj(df)
    if encoder == "concat_proj":
        return _extract_concat_proj(df)
    raise ValueError(
        f"Unknown text_encoder: {encoder!r} "
        f"(use 'sbert', 'tfidf', 'concat', 'sbert_proj' or 'concat_proj')"
    )


def _extract_sbert_proj(df):
    """Frozen SBERT raw + trainable Linear projection (learned in the model).

    features.npy view:  SBERT(frozen) -> train-only SVD-128  (== Variant B)
    text_raw.npy view:  SBERT(frozen) raw 384D, train-only z-scored

    The model wrapper applies nn.Linear(384 -> 128) end-to-end on text_raw and
    overwrites the first 128 dims of features.npy at forward time. The SVD-128
    view persists only for relation building (semsim cosine) and CARE filter.

    Leakage-safe: SBERT is pretrained and never fit; SVD and StandardScaler
    are fit on split=='train' rows only.
    """
    print(f"[Feature] 텍스트 임베딩 — SBERT_PROJ "
          f"(frozen SBERT '{CONFIG['sbert_model']}' + trainable Linear)...")

    train_mask = _train_mask(df)
    sbert_raw = _encode_sbert(df)  # (N, 384), frozen pretrained

    n_comp = min(CONFIG["svd_components"], sbert_raw.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd.fit(sbert_raw[train_mask])
    text_embeddings = svd.transform(sbert_raw)
    print(f"  features.npy view: SBERT→SVD{n_comp}, "
          f"EVR={svd.explained_variance_ratio_.sum():.4f}")

    sc_raw = StandardScaler().fit(sbert_raw[train_mask])
    text_raw = sc_raw.transform(sbert_raw).astype(np.float32)
    print(f"  text_raw.npy view: raw SBERT z-scored {text_raw.shape}")

    return text_embeddings, None, svd, text_raw


def _extract_concat_proj(df):
    """Concat(SBERT raw, TF-IDF SVD-128) + trainable Linear projection.

    features.npy view:  TF-IDF→SVD128 ⊕ SBERT→SVD128 → joint SVD-128 (== Variant C)
    text_raw.npy view:  [frozen SBERT 384D (z), TF-IDF SVD-128 (z)] = 512D

    The model wrapper applies nn.Linear(512 -> 128) end-to-end on text_raw, so
    the projection can learn to weight SBERT semantic vs TF-IDF lexical signal.
    Relation building / CARE see the fixed joint-SVD view unchanged.
    """
    print(f"[Feature] 텍스트 임베딩 — CONCAT_PROJ "
          f"(frozen SBERT '{CONFIG['sbert_model']}' ⊕ TF-IDF + trainable Linear)...")

    train_mask = _train_mask(df)
    n_comp = CONFIG["svd_components"]

    # --- TF-IDF block ---
    texts = df["text"].fillna("").values
    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"],
        ngram_range=CONFIG["tfidf_ngram"],
        lowercase=True,
        token_pattern=r"\b\w+\b",
    )
    vectorizer.fit(texts[train_mask])
    tfidf_matrix = vectorizer.transform(texts)
    svd_tfidf = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd_tfidf.fit(tfidf_matrix[train_mask])
    emb_tfidf = svd_tfidf.transform(tfidf_matrix)
    print(f"  TF-IDF block: vocab={len(vectorizer.vocabulary_)}, "
          f"SVD{n_comp} EVR={svd_tfidf.explained_variance_ratio_.sum():.4f}")

    # --- SBERT block (raw + reduced) ---
    sbert_raw = _encode_sbert(df)  # (N, 384), frozen pretrained
    nb = min(n_comp, sbert_raw.shape[1] - 1)
    svd_sbert = TruncatedSVD(n_components=nb, random_state=CONFIG["random_state"])
    svd_sbert.fit(sbert_raw[train_mask])
    emb_sbert = svd_sbert.transform(sbert_raw)
    print(f"  SBERT block: dim={sbert_raw.shape[1]}, "
          f"SVD{nb} EVR={svd_sbert.explained_variance_ratio_.sum():.4f}")

    # --- features.npy view (== Variant C: joint SVD back to n_comp) ---
    sc_a = StandardScaler().fit(emb_tfidf[train_mask])
    sc_b = StandardScaler().fit(emb_sbert[train_mask])
    fused = np.concatenate(
        [sc_a.transform(emb_tfidf), sc_b.transform(emb_sbert)], axis=1
    )
    svd = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd.fit(fused[train_mask])
    text_embeddings = svd.transform(fused)
    print(f"  features.npy view: joint SVD{n_comp}, "
          f"EVR={svd.explained_variance_ratio_.sum():.4f}")

    # --- text_raw.npy view ([SBERT raw 384 (z), TF-IDF SVD-128 (z)]) ---
    sc_raw_sbert = StandardScaler().fit(sbert_raw[train_mask])
    sc_raw_tfidf = StandardScaler().fit(emb_tfidf[train_mask])
    text_raw = np.concatenate(
        [sc_raw_sbert.transform(sbert_raw), sc_raw_tfidf.transform(emb_tfidf)],
        axis=1,
    ).astype(np.float32)
    print(f"  text_raw.npy view: [SBERT 384 (z), TF-IDF {n_comp} (z)] "
          f"= {text_raw.shape}")

    return text_embeddings, vectorizer, svd, text_raw


def _extract_tfidf_svd(df):
    print("[Feature] 텍스트 임베딩 — TF-IDF (train-fit, valid/test transform-only)...")

    texts = df["text"].fillna("").values
    train_mask = _train_mask(df)

    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"],
        ngram_range=CONFIG["tfidf_ngram"],
        lowercase=True,
        token_pattern=r"\b\w+\b",
    )
    vectorizer.fit(texts[train_mask])
    print(f"  TF-IDF fit on {int(train_mask.sum())} train docs, vocab={len(vectorizer.vocabulary_)}")

    tfidf_matrix = vectorizer.transform(texts)
    print(f"  TF-IDF transform 전체: {tfidf_matrix.shape}")

    svd = TruncatedSVD(n_components=CONFIG["svd_components"], random_state=CONFIG["random_state"])
    svd.fit(tfidf_matrix[train_mask])
    text_embeddings = svd.transform(tfidf_matrix)
    print(f"  SVD fit on train rows. Final emb: {text_embeddings.shape}")
    print(f"  Explained variance (train-fit basis): {svd.explained_variance_ratio_.sum():.4f}")

    return text_embeddings, vectorizer, svd


def _row_ids(df):
    """Stable per-row identity for cache invalidation."""
    col = "review_id" if "review_id" in df.columns else None
    ids = df[col].astype(str).values if col else np.arange(len(df)).astype(str)
    return ids


def _sbert_cache_path(model_name, df):
    os.makedirs(CONFIG["interim_dir"], exist_ok=True)
    ids = _row_ids(df)
    key = hashlib.sha1(
        (model_name + "|" + str(len(ids)) + "|" + ",".join(ids)).encode("utf-8")
    ).hexdigest()[:16]
    safe_model = model_name.replace("/", "_")
    return os.path.join(CONFIG["interim_dir"], f"sbert_emb_{safe_model}_{key}.npy")


def _encode_sbert(df):
    """Frozen pretrained SBERT encoding of EVERY row (no fit, no finetune).

    Deterministic + cached to data/interim so the 75x multi-seed runs reuse
    one encoding pass. Leakage-safe: the encoder is pretrained and is never
    fit/finetuned on any split; only the downstream SVD is train-only fit.
    """
    model_name = CONFIG["sbert_model"]
    cache_path = _sbert_cache_path(model_name, df)
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        print(f"  SBERT cache hit: {cache_path} {emb.shape}")
        return emb

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for text_encoder='sbert'. "
            "Install: pip install sentence-transformers"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading SBERT '{model_name}' on {device} (frozen, no fit)...")
    model = SentenceTransformer(model_name, device=device)

    texts = df["text"].fillna("").astype(str).tolist()
    with torch.no_grad():
        emb = model.encode(
            texts,
            batch_size=CONFIG["sbert_batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache_path, emb)
    print(f"  SBERT encoded {emb.shape}, cached -> {cache_path}")
    return emb


def _extract_sbert_svd(df):
    print(f"[Feature] 텍스트 임베딩 — SBERT '{CONFIG['sbert_model']}' "
          f"(frozen) -> SVD (train-only fit)...")

    train_mask = _train_mask(df)
    emb = _encode_sbert(df)  # (N, D), pretrained frozen — NOT fit on any split

    n_comp = min(CONFIG["svd_components"], emb.shape[1] - 1)
    if n_comp != CONFIG["svd_components"]:
        print(f"  [Warn] SBERT dim {emb.shape[1]} < svd_components; using {n_comp}")
    svd = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd.fit(emb[train_mask])
    text_embeddings = svd.transform(emb)
    print(f"  SVD fit on {int(train_mask.sum())} train rows. Final emb: {text_embeddings.shape}")
    print(f"  Explained variance (train-fit basis): {svd.explained_variance_ratio_.sum():.4f}")

    return text_embeddings, None, svd


def _extract_concat_svd(df):
    """TF-IDF ⊕ SBERT fusion, ending in a train-only SVD back to 128 dims.

    Each modality is first reduced by its own train-only SVD(128), then
    z-scored on train rows (so the joint SVD is not dominated by TF-IDF's
    larger singular-value scale), concatenated to 256, and finally compressed
    by one more train-only SVD to 128. Net effect: same downstream
    text_embedding_dim as the 'tfidf' / 'sbert' single-encoder runs, so the
    model/relations/CARE pipeline is byte-for-byte unchanged and the encoder
    comparison stays apples-to-apples. Leakage-safe: every fit is train-only;
    SBERT itself is frozen pretrained (never fit).
    """
    print("[Feature] 텍스트 임베딩 — CONCAT(TF-IDF ⊕ SBERT) "
          "-> 공동 SVD (train-only fit)...")

    train_mask = _train_mask(df)
    n_comp = CONFIG["svd_components"]

    # --- branch A: TF-IDF -> train-only SVD(n_comp) ---
    texts = df["text"].fillna("").values
    vectorizer = TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        min_df=CONFIG["tfidf_min_df"],
        max_df=CONFIG["tfidf_max_df"],
        ngram_range=CONFIG["tfidf_ngram"],
        lowercase=True,
        token_pattern=r"\b\w+\b",
    )
    vectorizer.fit(texts[train_mask])
    tfidf_matrix = vectorizer.transform(texts)
    svd_tfidf = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd_tfidf.fit(tfidf_matrix[train_mask])
    emb_tfidf = svd_tfidf.transform(tfidf_matrix)
    print(f"  TF-IDF block: vocab={len(vectorizer.vocabulary_)}, "
          f"SVD{n_comp} EVR={svd_tfidf.explained_variance_ratio_.sum():.4f}")

    # --- branch B: frozen SBERT -> train-only SVD(n_comp) ---
    sbert_raw = _encode_sbert(df)  # pretrained frozen, NOT fit on any split
    nb = min(n_comp, sbert_raw.shape[1] - 1)
    svd_sbert = TruncatedSVD(n_components=nb, random_state=CONFIG["random_state"])
    svd_sbert.fit(sbert_raw[train_mask])
    emb_sbert = svd_sbert.transform(sbert_raw)
    print(f"  SBERT block: dim={sbert_raw.shape[1]}, "
          f"SVD{nb} EVR={svd_sbert.explained_variance_ratio_.sum():.4f}")

    # --- per-modality train-only z-score, then concat to (N, 2*n_comp) ---
    sc_a = StandardScaler().fit(emb_tfidf[train_mask])
    sc_b = StandardScaler().fit(emb_sbert[train_mask])
    fused = np.concatenate(
        [sc_a.transform(emb_tfidf), sc_b.transform(emb_sbert)], axis=1
    )
    print(f"  Fused (pre-joint-SVD): {fused.shape}")

    # --- joint train-only SVD back to n_comp (apples-to-apples dim) ---
    svd = TruncatedSVD(n_components=n_comp, random_state=CONFIG["random_state"])
    svd.fit(fused[train_mask])
    text_embeddings = svd.transform(fused)
    print(f"  Joint SVD fit on {int(train_mask.sum())} train rows. "
          f"Final emb: {text_embeddings.shape}, "
          f"EVR={svd.explained_variance_ratio_.sum():.4f}")

    return text_embeddings, vectorizer, svd


def extract_numeric_features(df):
    print("\n[Feature] 정형 Feature 추출...")

    features = {}
    features["rating_norm"] = (df["rating"] - 3.0) / 2.0
    features["review_length"] = df["text"].fillna("").str.len()
    features["review_length_log"] = np.log1p(features["review_length"])

    user_stats = df.groupby("user_id").agg({
        "review_id": "count",
        "rating": ["mean", "std"],
    }).fillna(0)
    user_stats.columns = ["user_review_count", "user_avg_rating", "user_rating_std"]

    features["user_review_count_log"] = np.log1p(df["user_id"].map(user_stats["user_review_count"]))
    features["user_avg_rating"] = df["user_id"].map(user_stats["user_avg_rating"])
    features["user_rating_std"] = df["user_id"].map(user_stats["user_rating_std"])

    product_stats = df.groupby("prod_id").agg({
        "review_id": "count",
        "rating": ["mean", "std"],
    }).fillna(0)
    product_stats.columns = ["product_review_count", "product_avg_rating", "product_rating_std"]

    features["product_review_count_log"] = np.log1p(df["prod_id"].map(product_stats["product_review_count"]))
    features["product_avg_rating"] = df["prod_id"].map(product_stats["product_avg_rating"])
    features["product_rating_std"] = df["prod_id"].map(product_stats["product_rating_std"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    min_date = df["date"].min()
    features["days_since_first_review"] = (df["date"] - min_date).dt.days

    month_num = df["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    features["month_cos"] = np.cos(2 * np.pi * month_num / 12)

    feature_df = pd.DataFrame(features).fillna(0)
    print(f"  Numeric features: {feature_df.shape[1]} cols → {list(feature_df.columns)}")
    return feature_df


def normalize_features(text_embeddings, numeric_features, train_mask):
    print("\n[Feature] 정규화 (StandardScaler train-fit)...")

    text_scaler = StandardScaler()
    text_scaler.fit(text_embeddings[train_mask])
    text_embeddings_normalized = text_scaler.transform(text_embeddings)

    numeric_scaler = StandardScaler()
    numeric_scaler.fit(numeric_features[train_mask])
    numeric_features_normalized = numeric_scaler.transform(numeric_features)

    print(f"  텍스트 임베딩: {text_embeddings_normalized.shape}")
    print(f"  정형 Feature: {numeric_features_normalized.shape}")
    return text_embeddings_normalized, numeric_features_normalized, text_scaler, numeric_scaler


def concatenate_features(text_emb, numeric_feat):
    combined = np.concatenate([text_emb, numeric_feat], axis=1)
    print(f"\n[Feature] 최종 Feature: {combined.shape}")
    return combined


def save_features(df, combined_features, text_raw=None):
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    x_path = os.path.join(CONFIG["processed_dir"], "features.npy")
    np.save(x_path, combined_features)
    print(f"[Save] {x_path}")

    samples_path = os.path.join(CONFIG["processed_dir"], "node_samples.csv")
    df.to_csv(samples_path, index=False)
    print(f"[Save] {samples_path}")

    text_raw_path = os.path.join(CONFIG["processed_dir"], "text_raw.npy")
    if text_raw is not None:
        np.save(text_raw_path, text_raw.astype(np.float32))
        print(f"[Save] {text_raw_path}  shape={text_raw.shape}")
    elif os.path.exists(text_raw_path):
        # Avoid stale text_raw.npy from a prior *_proj run polluting a
        # non-proj encoder swap.
        os.remove(text_raw_path)
        print(f"[Clean] removed stale {text_raw_path}")

    encoder = CONFIG["text_encoder"].lower()
    if encoder == "sbert":
        fit_note = (
            f"SBERT('{CONFIG['sbert_model']}') frozen — 어떤 split에도 fit/finetune 안 함 "
            f"(사전학습 임베딩 캐시 동봉). SVD/StandardScaler는 split=='train' 행에서만 fit, "
            f"그 외 transform-only."
        )
    elif encoder == "concat":
        fit_note = (
            f"CONCAT(TF-IDF ⊕ SBERT('{CONFIG['sbert_model']}') frozen). 모달리티별 "
            f"SVD128 후 train-only z-score → concat(256) → 공동 SVD128. SBERT는 "
            f"어떤 split에도 fit/finetune 안 함; TF-IDF/모든 SVD/StandardScaler는 "
            f"split=='train' 행에서만 fit, 그 외 transform-only."
        )
    elif encoder == "sbert_proj":
        fit_note = (
            f"SBERT_PROJ: features.npy = SBERT('{CONFIG['sbert_model']}') frozen "
            f"→ train-only SVD128 (relation building / CARE 용). text_raw.npy = "
            f"frozen SBERT 384D + train-only z-score. 모델에서 nn.Linear(384→128) "
            f"을 end-to-end 학습해 첫 128 dims를 덮어씀. SBERT 본체는 frozen, "
            f"SVD/StandardScaler는 split=='train' 행에서만 fit."
        )
    elif encoder == "concat_proj":
        fit_note = (
            f"CONCAT_PROJ: features.npy = CONCAT(TF-IDF ⊕ SBERT frozen) → joint "
            f"SVD128 (relation building / CARE 용). text_raw.npy = [frozen SBERT "
            f"384D (z), TF-IDF SVD128 (z)] = 512D. 모델에서 nn.Linear(512→128)을 "
            f"end-to-end 학습해 첫 128 dims를 덮어씀. SBERT 본체는 frozen, "
            f"TF-IDF/SVD/StandardScaler는 split=='train' 행에서만 fit."
        )
    else:
        fit_note = "TF-IDF/SVD/StandardScaler 모두 split=='train' 행에서만 fit, 그 외 transform-only"

    meta = {
        "num_nodes": int(combined_features.shape[0]),
        "num_features": int(combined_features.shape[1]),
        "text_embedding_dim": int(CONFIG["svd_components"]),
        "numeric_features_dim": int(combined_features.shape[1] - CONFIG["svd_components"]),
        "text_encoder": encoder,
        "sbert_model": (
            CONFIG["sbert_model"]
            if encoder in ("sbert", "concat", "sbert_proj", "concat_proj")
            else None
        ),
        "fit_scope": "train_only",
        "note": fit_note,
    }
    if text_raw is not None:
        meta["text_raw_dim"] = int(text_raw.shape[1])
        meta["text_proj_dim"] = int(CONFIG["svd_components"])
        meta["trainable_text_projection"] = True
    else:
        meta["trainable_text_projection"] = False
    meta_path = os.path.join(CONFIG["processed_dir"], "feature_meta.json")
    save_json(meta, meta_path)
    print(f"[Save] {meta_path}")


if __name__ == "__main__":
    input_path = os.path.join(CONFIG["processed_dir"], "sampled_reviews.csv")
    df = pd.read_csv(input_path)
    assert "split" in df.columns, "sampled_reviews.csv must contain 'split' column (run sampling.py first)"
    assert (df["split"] == "train").sum() > 0, "No train rows found"

    text_embeddings, vectorizer, svd, text_raw = extract_text_embedding(df)
    numeric_features = extract_numeric_features(df)

    train_mask = _train_mask(df)
    text_norm, numeric_norm, text_scaler, numeric_scaler = normalize_features(
        text_embeddings, numeric_features.values, train_mask
    )

    combined_features = concatenate_features(text_norm, numeric_norm)
    save_features(df, combined_features, text_raw=text_raw)

    print(f"\n[Done] Feature Engineering 완료. shape={combined_features.shape}")
    if text_raw is not None:
        print(f"        + text_raw.shape={text_raw.shape} "
              f"(consumed by TextProjectionWrapper during training)")
