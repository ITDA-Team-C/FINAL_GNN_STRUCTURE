#!/usr/bin/env python
"""
Baseline 모델 4개 일괄 학습 스크립트
MLP, GCN, GraphSAGE, GAT 순서대로 학습
"""
import subprocess
import sys
import time

baselines = ["mlp", "gcn", "graphsage", "gat"]

print("\n" + "=" * 70)
print("🚀 Baseline Models Training Pipeline (MLP, GCN, GraphSAGE, GAT)")
print("=" * 70)

start_time = time.time()
results = []

for i, model in enumerate(baselines, 1):
    print(f"\n[{i}/4] {model.upper()} Training with configs/default.yaml")
    print("-" * 70)

    cmd = [
        sys.executable, "-m", "src.training.train",
        "--model", model,
        "--config", "configs/default.yaml",
        "--skip-preprocessing",  # Use preprocess.py 결과 재사용
        "--skip-graph"           # 동일한 그래프 재사용
    ]

    model_start = time.time()
    result = subprocess.run(cmd)
    model_time = time.time() - model_start

    if result.returncode != 0:
        print(f"❌ {model} training failed with code {result.returncode}")
        results.append((model, "FAILED", model_time))
    else:
        print(f"✅ {model} training completed in {model_time:.1f}s")
        results.append((model, "SUCCESS", model_time))

print("\n" + "=" * 70)
print("📋 Training Summary")
print("=" * 70)
for model, status, elapsed in results:
    status_icon = "✅" if status == "SUCCESS" else "❌"
    print(f"{status_icon} {model:12s} | {status:8s} | {elapsed:7.1f}s")

total_time = time.time() - start_time
print("-" * 70)
print(f"Total time: {total_time/60:.1f} minutes")
print("=" * 70)
