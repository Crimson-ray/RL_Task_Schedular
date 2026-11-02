#!/usr/bin/env python3
"""
Benchmark every scheduler (D3QN + 5 baselines) on the preprocessed real traces.

Uses the project's own simulator and BenchmarkEngine unchanged; only the task
streams come from Alibaba / Google instead of the synthetic generator.
The trained model never saw these traces, so this is a generalization test.

Usage:
    python datasets/run_benchmark_on_traces.py                 # every dataset with processed windows
    python datasets/run_benchmark_on_traces.py --dataset alibaba
"""
import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np

from src.benchmarks.benchmark_engine import BenchmarkEngine
from src.config import ClusterConfig, WorkloadConfig
from src.scheduler.heuristics import (
    EDFScheduler,
    FIFOScheduler,
    RoundRobinScheduler,
    SJFScheduler,
    TetrisBestFitScheduler,
)
from src.scheduler.intelligent_scheduler import IntelligentD3QNScheduler
from src.workload.trace_generator import TraceGenerator

DATASETS = {
    "alibaba": os.path.join("datasets", "alibaba_2018", "processed", "alibaba_window_*.csv"),
    "google_borg": os.path.join("datasets", "google_borg_2011", "processed", "google_borg_window_*.csv"),
}
RESULTS_DIR = os.path.join("datasets", "results")


def build_schedulers(cluster_config):
    return [
        IntelligentD3QNScheduler(checkpoint_path="checkpoints/best_d3qn_scheduler.pt", cluster_config=cluster_config),
        FIFOScheduler(),
        SJFScheduler(),
        EDFScheduler(),
        TetrisBestFitScheduler(),
        RoundRobinScheduler(),
    ]


def run_dataset(name, pattern, engine, schedulers):
    windows = sorted(glob.glob(pattern))
    if not windows:
        print(f"[{name}] no processed windows found ({pattern}); skipping.")
        return None

    print(f"\n[{name}] {len(windows)} window(s)")
    raw = {s.name: [] for s in schedulers}
    for path in windows:
        tasks = TraceGenerator.load_from_csv(path)
        print(f"  {os.path.basename(path)}: {len(tasks)} tasks")
        for sched in schedulers:
            raw[sched.name].append(engine.evaluate_scheduler(sched, tasks).to_dict())

    summary = {
        s: {k: round(float(np.mean([r[k] for r in runs])), 2) for k in runs[0]}
        for s, runs in raw.items()
    }
    engine.print_comparison_table(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{name}_benchmark_results.json")
    with open(out, "w") as f:
        json.dump({"dataset": name, "windows": windows, "summary": summary, "raw_runs": raw}, f, indent=2)
    print(f"Saved {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark schedulers on real-trace windows")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    args = parser.parse_args()

    cluster_config = ClusterConfig()
    engine = BenchmarkEngine(cluster_config, WorkloadConfig())
    schedulers = build_schedulers(cluster_config)

    names = [args.dataset] if args.dataset else sorted(DATASETS)
    for name in names:
        run_dataset(name, DATASETS[name], engine, schedulers)


if __name__ == "__main__":
    main()
