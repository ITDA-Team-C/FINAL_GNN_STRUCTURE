"""Run all 15 models with 5 random seeds (75 training runs total).

Updated lineup:
- Baselines (6): MLP / GCN / GAT / GraphSAGE / ChebConv / TAGConv
- CAGE-RF family (4): Base / Skip (w/o CARE) / Refine (v9) / + CARE
- CAGE-CareRF Lean (2): Lean-4 / Lean-5
- Ablations (3): w/o Skip / w/o Gating / w/o Aux Loss
- (+1 excluded from report: w/o Custom Relations — rule-violating)
- `w/o CARE` ablation is now folded into `CAGE-RF Skip (w/o CARE)` to avoid duplication.
- Lean-6 removed: was algorithmically identical to `w/o Gating` ablation.


Each (model, seed) pair saves results to a distinct file with `_seed{N}` suffix,
so all 80 outputs coexist in `outputs/`. After completion, summary printed and
aggregated metrics (mean ± std across 5 seeds) printed per model.

Usage:
    python 5x_run_all_models.py                           # run all 80
    python 5x_run_all_models.py --seeds 42 123            # custom seeds
    python 5x_run_all_models.py --only baselines          # only baseline group × 5 seeds
    python 5x_run_all_models.py --skip baselines          # skip baselines
    python 5x_run_all_models.py --continue-on-error
    python 5x_run_all_models.py --dry-run
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# (group_id, model_arg, config_path, label)
PLAN = [
    # A. Baselines (edge = union of 6 relations)
    ("baselines", "mlp",              "configs/default.yaml",                "MLP"),
    ("baselines", "gcn",              "configs/default.yaml",                "GCN"),
    ("baselines", "gat",              "configs/default.yaml",                "GAT"),
    ("baselines", "graphsage",        "configs/default.yaml",                "GraphSAGE"),
    ("baselines", "cheb",             "configs/default.yaml",                "ChebConv (baseline)"),
    ("baselines", "tag",              "configs/default.yaml",                "TAGConv (baseline)"),
    # B. CAGE-RF family (Skip variant also serves as the `w/o CARE` ablation)
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/default.yaml",                "CAGE-RF Base"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/v8_skip.yaml",                "CAGE-RF Skip (w/o CARE)"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/v9_twostage.yaml",            "CAGE-RF Refine (v9)"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/cage_rf_skip_care.yaml",      "CAGE-RF + CARE"),
    # C. CAGE-CareRF Lean variants (Lean-6 removed: identical to ablation_no_gating)
    ("carerf",    "cage_carerf_gnn",  "configs/cage_carerf_lean.yaml",       "CAGE-CareRF Lean-4"),
    ("carerf",    "cage_carerf_gnn",  "configs/cage_carerf_lean_5.yaml",     "CAGE-CareRF Lean-5"),
    # D. Ablations (base = full Lean-6 modules; FINAL = CAGE-RF + CARE)
    #    NOTE: `w/o CARE` is folded into "CAGE-RF Skip (w/o CARE)" above to avoid duplication.
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_skip.yaml",       "w/o Skip"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_gating.yaml",     "w/o Gating"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_aux.yaml",        "w/o Aux Loss"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_custom.yaml",     "w/o Custom Relations"),
]

ALL_GROUPS = ("baselines", "cage_rf", "carerf", "ablation")
DEFAULT_SEEDS = [42, 123, 2024, 7, 1234]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds to use (default: 42 123 2024 7 1234)")
    p.add_argument("--only", help="Comma-sep groups: baselines,cage_rf,carerf,ablation")
    p.add_argument("--skip", help="Comma-sep groups to skip")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going if a single training fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands only, don't execute")
    return p.parse_args()


def fmt_dur(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def aggregate_metrics():
    """Glob all metrics_*_seed*.json, group by (model_name minus _seed), compute mean/std."""
    metric_keys = ["pr_auc", "macro_f1", "roc_auc", "g_mean", "recall_pos", "recall_neg", "accuracy"]
    groups = defaultdict(list)

    patterns = [
        "outputs/cage_rf_gnn/metrics_*_seed*.json",
        "outputs/benchmark/CHEB/metrics_*_seed*.json",
    ]
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            base = os.path.basename(f)
            # metrics_<name>_seed<N>.json -> name
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
    print(f"=== Multi-seed Aggregation (n={len(groups)} models) ===")
    print("=" * 110)
    header = f"{'Model':50s} {'n':>3s} " + " ".join(f"{k:>16s}" for k in ["pr_auc", "macro_f1", "g_mean"])
    print(header)
    print("-" * 110)

    rows = []
    for model_key, runs in sorted(groups.items()):
        n = len(runs)

        def stat(key):
            vals = [r[key] for r in runs if r[key] is not None]
            if not vals:
                return None, None
            import statistics
            return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

        pr_mean, pr_std = stat("pr_auc")
        f1_mean, f1_std = stat("macro_f1")
        gm_mean, gm_std = stat("g_mean")
        rows.append((model_key, n, pr_mean, pr_std, f1_mean, f1_std, gm_mean, gm_std))
        if pr_mean is None:
            continue
        cell = lambda m, s: f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"
        print(f"{model_key:50s} {n:>3d} "
              f"{cell(pr_mean, pr_std):>16s} {cell(f1_mean, f1_std):>16s} {cell(gm_mean, gm_std):>16s}")

    # Best by mean
    best_pr = max((r for r in rows if r[2] is not None), key=lambda r: r[2], default=None)
    best_f1 = max((r for r in rows if r[4] is not None), key=lambda r: r[4], default=None)
    best_gm = max((r for r in rows if r[6] is not None), key=lambda r: r[6], default=None)
    print("-" * 110)
    if best_pr:
        print(f"  Best PR-AUC : {best_pr[0]:50s} mean={best_pr[2]:.4f} ± {best_pr[3]:.4f}")
    if best_f1:
        print(f"  Best F1     : {best_f1[0]:50s} mean={best_f1[4]:.4f} ± {best_f1[5]:.4f}")
    if best_gm:
        print(f"  Best G-Mean : {best_gm[0]:50s} mean={best_gm[6]:.4f} ± {best_gm[7]:.4f}")
    print("=" * 110)

    # Save aggregated CSV-like summary
    summary_path = "outputs/multi_seed_summary.json"
    Path("outputs").mkdir(exist_ok=True)
    summary = {
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
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"\n[Save] {summary_path}")


def main():
    args = parse_args()

    selected = set(ALL_GROUPS)
    if args.only:
        selected = set(s.strip() for s in args.only.split(","))
    if args.skip:
        selected -= set(s.strip() for s in args.skip.split(","))

    plan = [item for item in PLAN if item[0] in selected]
    seeds = args.seeds
    total_runs = len(plan) * len(seeds)
    print(f"=== Running {len(plan)} models × {len(seeds)} seeds = {total_runs} runs ===")
    print(f"=== Groups: {sorted(selected)} ===")
    print(f"=== Seeds:  {seeds} ===\n")

    t0 = time.time()
    results = []   # (label, seed, returncode, dur)
    idx = 0
    for seed in seeds:
        for group, model, cfg, label in plan:
            idx += 1
            cmd = [sys.executable, "-m", "src.training.train",
                   "--model", model, "--config", cfg, "--seed", str(seed)]
            header = f"[{idx}/{total_runs}] seed={seed} ({group}) {label}"
            print("=" * 80)
            print(header)
            print(f"  cmd: {' '.join(cmd)}")
            print("=" * 80, flush=True)

            if args.dry_run:
                results.append((label, seed, 0, 0.0))
                continue

            start = time.time()
            proc = subprocess.run(cmd)
            dur = time.time() - start
            results.append((label, seed, proc.returncode, dur))
            print(f"\n  -> exit={proc.returncode}  duration={fmt_dur(dur)}\n", flush=True)

            if proc.returncode != 0 and not args.continue_on_error:
                print(f"!! Training failed for seed={seed} {label} (exit {proc.returncode}). "
                      f"Use --continue-on-error to skip and keep going.")
                break
        else:
            continue
        break

    total = time.time() - t0
    print("=" * 80)
    print(f"=== Summary (total {fmt_dur(total)}) ===")
    print("=" * 80)
    ok = sum(1 for _, _, rc, _ in results if rc == 0)
    fail = len(results) - ok
    for label, seed, rc, dur in results:
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  [{mark}] seed={seed:5d} {label:<40} duration={fmt_dur(dur)} exit={rc}")
    print()
    print(f"  passed: {ok}    failed: {fail}    not_run: {total_runs - len(results)}")

    if not args.dry_run:
        aggregate_metrics()

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
