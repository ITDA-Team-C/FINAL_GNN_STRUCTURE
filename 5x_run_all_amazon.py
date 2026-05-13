"""Run 7 Amazon models with 5 random seeds (35 runs total).

Each (model, seed) pair saves to `amazon/outputs/metrics_<model>_seed{N}.json`.
After completion, aggregates mean ± std per model across seeds.

Usage:
    python 5x_run_all_amazon.py
    python 5x_run_all_amazon.py --seeds 42 123
    python 5x_run_all_amazon.py --only cage_carerf
    python 5x_run_all_amazon.py --mat-path amazon/data/Amazon.mat
    python 5x_run_all_amazon.py --epochs 100
    python 5x_run_all_amazon.py --continue-on-error
    python 5x_run_all_amazon.py --dry-run
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, ".")

MODELS = [
    "mlp",
    "gcn",
    "gat",
    "graphsage",
    "cage_carerf",
    "cage_carerf_no_care",
    "cage_carerf_no_aux",
]

DEFAULT_SEEDS = [42, 123, 2024, 7, 1234]


def fmt(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def aggregate(out_dir="amazon/outputs"):
    metric_keys = ["pr_auc", "macro_f1", "roc_auc", "g_mean", "recall_pos", "recall_neg", "accuracy"]
    groups = defaultdict(list)

    for f in sorted(glob.glob(os.path.join(out_dir, "metrics_*_seed*.json"))):
        base = os.path.basename(f)
        stem = base[len("metrics_"):-len(".json")]
        if "_seed" not in stem:
            continue
        model_key = stem.rsplit("_seed", 1)[0]
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        test = d.get("test_metrics", d)
        groups[model_key].append({k: test.get(k) for k in metric_keys})

    if not groups:
        print("\n[Aggregate] No seed-result files found yet.")
        return

    print("\n" + "=" * 110)
    print(f"=== Amazon Multi-seed Aggregation (n={len(groups)} models) ===")
    print("=" * 110)
    header = f"{'Model':35s} {'n':>3s} " + " ".join(f"{k:>16s}" for k in ["pr_auc", "macro_f1", "g_mean"])
    print(header)
    print("-" * 110)

    rows = []
    for model_key, runs in sorted(groups.items()):
        n = len(runs)

        def stat(key):
            vals = [r[key] for r in runs if r[key] is not None]
            if not vals:
                return None, None
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        pr_m, pr_s = stat("pr_auc")
        f1_m, f1_s = stat("macro_f1")
        gm_m, gm_s = stat("g_mean")
        rows.append((model_key, n, pr_m, pr_s, f1_m, f1_s, gm_m, gm_s))
        if pr_m is None:
            continue
        cell = lambda m, s: f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"
        print(f"{model_key:35s} {n:>3d} "
              f"{cell(pr_m, pr_s):>16s} {cell(f1_m, f1_s):>16s} {cell(gm_m, gm_s):>16s}")

    print("-" * 110)
    best_pr = max((r for r in rows if r[2] is not None), key=lambda r: r[2], default=None)
    best_f1 = max((r for r in rows if r[4] is not None), key=lambda r: r[4], default=None)
    best_gm = max((r for r in rows if r[6] is not None), key=lambda r: r[6], default=None)
    if best_pr:
        print(f"  Best PR-AUC : {best_pr[0]:35s} mean={best_pr[2]:.4f} ± {best_pr[3]:.4f}")
    if best_f1:
        print(f"  Best F1     : {best_f1[0]:35s} mean={best_f1[4]:.4f} ± {best_f1[5]:.4f}")
    if best_gm:
        print(f"  Best G-Mean : {best_gm[0]:35s} mean={best_gm[6]:.4f} ± {best_gm[7]:.4f}")
    print("=" * 110)

    summary = {
        "dataset": "amazon",
        "n_models": len(groups),
        "rows": [
            {
                "model": r[0], "n_seeds": r[1],
                "pr_auc_mean": r[2], "pr_auc_std": r[3],
                "macro_f1_mean": r[4], "macro_f1_std": r[5],
                "g_mean_mean": r[6], "g_mean_std": r[7],
            }
            for r in rows
        ],
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary_path = os.path.join(out_dir, "multi_seed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"\n[Save] {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--only", help="comma-sep model names; default = all 7")
    p.add_argument("--skip", help="comma-sep model names to skip")
    p.add_argument("--mat-path", default="amazon/data/Amazon.mat")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    selected = set(MODELS)
    if args.only:
        selected = set(s.strip() for s in args.only.split(","))
    if args.skip:
        selected -= set(s.strip() for s in args.skip.split(","))
    plan = [m for m in MODELS if m in selected]
    seeds = args.seeds
    total_runs = len(plan) * len(seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Amazon: {len(plan)} models × {len(seeds)} seeds = {total_runs} runs on {device} ===")
    print(f"=== mat_path = {args.mat_path}")
    print(f"=== epochs   = {args.epochs}")
    print(f"=== seeds    = {seeds}\n")

    from amazon.src.train import run_single

    t0 = time.time()
    results = []
    idx = 0
    for seed in seeds:
        for model_name in plan:
            idx += 1
            print(f"\n>>> [{idx}/{total_runs}] seed={seed} {model_name}")
            if args.dry_run:
                results.append((model_name, seed, 0, 0.0))
                continue
            start = time.time()
            try:
                run_single(model_name, args.mat_path, device,
                           epochs=args.epochs, seed=seed)
                rc = 0
            except Exception as e:
                print(f"  ERROR: {e}")
                rc = 1
                if not args.continue_on_error:
                    results.append((model_name, seed, rc, time.time() - start))
                    break
            results.append((model_name, seed, rc, time.time() - start))
        else:
            continue
        break

    total = time.time() - t0
    print(f"\n{'=' * 70}\n=== Summary (total {fmt(total)}) ===\n{'=' * 70}")
    for name, seed, rc, dur in results:
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  [{mark}] seed={seed:5d} {name:<28} duration={fmt(dur)}")
    fail = sum(1 for _, _, rc, _ in results if rc != 0)

    if not args.dry_run:
        aggregate()

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
