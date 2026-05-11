"""Run all 15 models sequentially (4 baseline + 4 CAGE-RF + Lean FINAL + v1 + 5 ablation).

Usage:
    python run_all_models.py                       # run everything
    python run_all_models.py --skip-baselines      # skip A
    python run_all_models.py --only baselines      # only A
    python run_all_models.py --only carerf,ablation # only C+D
    python run_all_models.py --continue-on-error   # don't stop on a single failure
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# (group_id, model_arg, config_path, label)
PLAN = [
    # A. Baselines (edge = union of 6 relations)
    ("baselines", "mlp",              "configs/default.yaml",                "MLP"),
    ("baselines", "gcn",              "configs/default.yaml",                "GCN"),
    ("baselines", "gat",              "configs/default.yaml",                "GAT"),
    ("baselines", "graphsage",        "configs/default.yaml",                "GraphSAGE"),
    # B. CAGE-RF family
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/default.yaml",                "CAGE-RF Base"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/v8_skip.yaml",                "CAGE-RF Skip (v8)"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/v9_twostage.yaml",            "CAGE-RF Refine (v9)"),
    ("cage_rf",   "cage_rf_gnn_cheb", "configs/cage_rf_skip_care.yaml",      "CAGE-RF + CARE"),
    # C. CAGE-CareRF FINAL + v1
    ("carerf",    "cage_carerf_gnn",  "configs/cage_carerf_lean.yaml",       "CAGE-CareRF FINAL (Lean)"),
    ("carerf",    "cage_carerf_gnn",  "configs/cage_carerf.yaml",            "CAGE-CareRF v1 (with Gating/Custom)"),
    # D. Ablations (CAGE-CareRF v1 base)
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_care.yaml",       "w/o CARE filter"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_skip.yaml",       "w/o Skip"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_gating.yaml",     "w/o Gating"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_aux.yaml",        "w/o Aux Loss"),
    ("ablation",  "cage_carerf_gnn",  "configs/ablation_no_custom.yaml",     "w/o Custom Relations"),
]

ALL_GROUPS = ("baselines", "cage_rf", "carerf", "ablation")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-sep groups: baselines,cage_rf,carerf,ablation")
    p.add_argument("--skip", help="comma-sep groups to skip")
    p.add_argument("--continue-on-error", action="store_true",
                   help="keep going if a single training fails")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands only, don't execute")
    return p.parse_args()


def fmt_dur(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = parse_args()

    selected = set(ALL_GROUPS)
    if args.only:
        selected = set(s.strip() for s in args.only.split(","))
    if args.skip:
        selected -= set(s.strip() for s in args.skip.split(","))

    plan = [item for item in PLAN if item[0] in selected]
    print(f"=== Running {len(plan)} / {len(PLAN)} models ===")
    print(f"=== Groups: {sorted(selected)} ===\n")

    t0 = time.time()
    results = []   # (label, returncode, dur_seconds)
    for idx, (group, model, cfg, label) in enumerate(plan, 1):
        cmd = [sys.executable, "-m", "src.training.train", "--model", model, "--config", cfg]
        header = f"[{idx}/{len(plan)}] ({group}) {label}"
        print("=" * 80)
        print(header)
        print(f"  cmd: {' '.join(cmd)}")
        print("=" * 80, flush=True)

        if args.dry_run:
            results.append((label, 0, 0.0))
            continue

        start = time.time()
        proc = subprocess.run(cmd)
        dur = time.time() - start
        results.append((label, proc.returncode, dur))
        print(f"\n  -> exit={proc.returncode}  duration={fmt_dur(dur)}\n", flush=True)

        if proc.returncode != 0 and not args.continue_on_error:
            print(f"!! Training failed for {label} (exit {proc.returncode}). "
                  f"Use --continue-on-error to skip and keep going.")
            break

    total = time.time() - t0
    print("=" * 80)
    print(f"=== Summary (total {fmt_dur(total)}) ===")
    print("=" * 80)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    fail = len(results) - ok
    for label, rc, dur in results:
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  [{mark}] {label:<40} duration={fmt_dur(dur)} exit={rc}")
    print()
    print(f"  passed: {ok}    failed: {fail}    not_run: {len(plan)-len(results)}")

    print("\n=== Metrics output files (check existence) ===")
    metric_files = sorted(Path("outputs/cage_rf_gnn").glob("metrics_*.json")) \
                 + sorted(Path("outputs/benchmark/cheb").glob("metrics_*.json"))
    for f in metric_files:
        print(f"  {f}")

    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
