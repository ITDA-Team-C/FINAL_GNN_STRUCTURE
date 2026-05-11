"""Run 7 models on YelpChi dataset (.mat format).

Usage:
    # Place YelpChi.mat in yelchi/data/ first
    python run_all_yelchi.py
    python run_all_yelchi.py --only cage_carerf
    python run_all_yelchi.py --mat-path /path/to/YelpChi.mat
    python run_all_yelchi.py --epochs 100
    python run_all_yelchi.py --continue-on-error
"""
import argparse
import sys
import time
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


def fmt(seconds):
    h, r = divmod(int(seconds), 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-sep model names; default = all 7")
    p.add_argument("--skip", help="comma-sep model names to skip")
    p.add_argument("--mat-path", default="yelchi/data/YelpChi.mat")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YelpChi: running {len(plan)}/{len(MODELS)} models on {device} ===")
    print(f"=== mat_path = {args.mat_path}")
    print(f"=== epochs   = {args.epochs}\n")

    from yelchi.src.train import run_single

    t0 = time.time()
    results = []
    for i, model_name in enumerate(plan, 1):
        print(f"\n>>> [{i}/{len(plan)}] {model_name}")
        if args.dry_run:
            results.append((model_name, 0, 0.0))
            continue
        start = time.time()
        try:
            run_single(model_name, args.mat_path, device, epochs=args.epochs)
            rc = 0
        except Exception as e:
            print(f"  ERROR: {e}")
            rc = 1
            if not args.continue_on_error:
                results.append((model_name, rc, time.time() - start))
                break
        results.append((model_name, rc, time.time() - start))

    total = time.time() - t0
    print(f"\n{'=' * 70}\n=== Summary (total {fmt(total)}) ===\n{'=' * 70}")
    for name, rc, dur in results:
        mark = "OK " if rc == 0 else "FAIL"
        print(f"  [{mark}] {name:<28} duration={fmt(dur)}")
    fail = sum(1 for _, rc, _ in results if rc != 0)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
