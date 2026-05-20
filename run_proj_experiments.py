"""Run trainable Linear text projection experiments.

For each of TWO new encoder variants:
  - sbert_proj  : frozen SBERT 384D + nn.Linear(384 -> 128) end-to-end
  - concat_proj : [frozen SBERT 384D, TF-IDF SVD-128] + nn.Linear(512 -> 128)

re-build features (TEXT_ENCODER=...), re-build relations (semsim depends on the
text view in features.npy), then run the FINAL CAGE-RF + CARE backbone over
5 seeds. Outputs are tagged per-variant under outputs/proj_experiments/<variant>/
so a later variant does not clobber the earlier one.

Usage:
    python run_proj_experiments.py                 # both variants × 5 seeds
    python run_proj_experiments.py --only sbert_proj
    python run_proj_experiments.py --seeds 7 42 123
    python run_proj_experiments.py --continue-on-error
    python run_proj_experiments.py --dry-run

The output of each (variant, seed) run is copied to:
    outputs/proj_experiments/<variant>/metrics_seed{N}.json
A small aggregate file is written at:
    outputs/proj_experiments/<variant>/summary.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = [7, 42, 123, 2024, 1234]
DEFAULT_VARIANTS = ["sbert_proj", "concat_proj"]
MODEL = "cage_rf_gnn_cheb"
CONFIG = "configs/cage_rf_skip_care.yaml"
# Where train.py writes the FINAL CAGE-RF + CARE run for this (model, config).
# (See train.py output_dir logic; for cage_rf_gnn_cheb + version 'cage_rf_skip_care'.)
TRAIN_OUTPUT_DIR = REPO_ROOT / "outputs" / "benchmark" / "CHEB"
METRICS_TEMPLATE = "metrics_cage_rf_gnn_cheb_cage_rf_skip_care_seed{seed}.json"
EXPERIMENT_ROOT = REPO_ROOT / "outputs" / "proj_experiments"


def run(cmd, env=None, dry_run=False, continue_on_error=False):
    cmd_str = " ".join(str(c) for c in cmd)
    env_extra = (
        " ".join(f"{k}={v}" for k, v in (env or {}).items() if k == "TEXT_ENCODER")
        if env else ""
    )
    print(f"\n$ {env_extra} {cmd_str}".strip())
    if dry_run:
        return 0
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=full_env)
    if result.returncode != 0:
        msg = f"command failed (exit {result.returncode}): {cmd_str}"
        if continue_on_error:
            print(f"[continue-on-error] {msg}")
            return result.returncode
        raise RuntimeError(msg)
    return 0


def prepare_features(variant, dry_run, continue_on_error):
    """Re-run feature_engineering + build_relations for this variant."""
    env = {"TEXT_ENCODER": variant}
    rc = run([sys.executable, "-m", "src.preprocessing.feature_engineering"],
             env=env, dry_run=dry_run, continue_on_error=continue_on_error)
    if rc != 0:
        return rc
    rc = run([sys.executable, "-m", "src.graph.build_relations"],
             dry_run=dry_run, continue_on_error=continue_on_error)
    if rc != 0:
        return rc
    rc = run([sys.executable, "-m", "src.graph.relation_quality"],
             dry_run=dry_run, continue_on_error=continue_on_error)
    return rc


def train_one(variant, seed, dry_run, continue_on_error):
    cmd = [sys.executable, "-m", "src.training.train",
           "--model", MODEL,
           "--config", CONFIG,
           "--seed", str(seed)]
    rc = run(cmd, dry_run=dry_run, continue_on_error=continue_on_error)
    if rc != 0 or dry_run:
        return rc

    src = TRAIN_OUTPUT_DIR / METRICS_TEMPLATE.format(seed=seed)
    if not src.exists():
        msg = f"expected metrics file missing: {src}"
        if continue_on_error:
            print(f"[continue-on-error] {msg}")
            return 1
        raise FileNotFoundError(msg)

    dst_dir = EXPERIMENT_ROOT / variant
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"metrics_seed{seed}.json"
    shutil.copy2(src, dst)
    print(f"[Tag] {src.name} -> {dst}")
    return 0


def summarize(variant, seeds):
    out_dir = EXPERIMENT_ROOT / variant
    if not out_dir.exists():
        return
    rows = []
    for seed in seeds:
        path = out_dir / f"metrics_seed{seed}.json"
        if not path.exists():
            continue
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        return

    def col(key):
        return np.array([r["test_metrics"].get(key) for r in rows
                         if r["test_metrics"].get(key) is not None], dtype=float)

    keys = ["pr_auc", "macro_f1", "g_mean", "roc_auc"]
    summary = {"variant": variant, "n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for k in keys:
        vals = col(k)
        if len(vals) == 0:
            continue
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                      "n": int(len(vals))}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary:{variant}]")
    for k in keys:
        if k in summary:
            print(f"  {k:>10s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f} "
                  f"(n={summary[k]['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=DEFAULT_VARIANTS, default=None,
                    help="run only this variant (default: both)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"seeds to train (default: {DEFAULT_SEEDS})")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going past failures instead of raising")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing them")
    args = ap.parse_args()

    variants = [args.only] if args.only else DEFAULT_VARIANTS
    seeds = args.seeds

    print(f"[Plan] variants={variants}, seeds={seeds}, "
          f"model={MODEL}, config={CONFIG}")
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print("\n" + "=" * 70)
        print(f"[Variant] {variant}")
        print("=" * 70)
        rc = prepare_features(variant, args.dry_run, args.continue_on_error)
        if rc != 0 and not args.continue_on_error:
            return rc

        for seed in seeds:
            print(f"\n--- {variant} | seed={seed} ---")
            train_one(variant, seed, args.dry_run, args.continue_on_error)

        if not args.dry_run:
            summarize(variant, seeds)

    print("\n[Done] All variants finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
