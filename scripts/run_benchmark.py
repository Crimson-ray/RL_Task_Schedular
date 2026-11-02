#!/usr/bin/env python3
"""
CLI Script to execute production benchmarks comparing the Intelligent Scheduler
against traditional algorithms (FIFO, SJF, EDF, Tetris BestFit, Round Robin).
Exports benchmark_results.json and generates benchmark_comparison.png.
"""
import sys
import os
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.benchmark_engine import BenchmarkEngine
from src.scheduler.heuristics import (
    FIFOScheduler,
    SJFScheduler,
    EDFScheduler,
    TetrisBestFitScheduler,
    RoundRobinScheduler
)
from src.scheduler.intelligent_scheduler import IntelligentD3QNScheduler
from src.config import ClusterConfig, WorkloadConfig, RLConfig

def main():
    print("=========================================================================")
    print("       PRODUCTION INTELLIGENT TASK SCHEDULER BENCHMARK SUITE             ")
    print("=========================================================================")

    cluster_config = ClusterConfig()
    workload_config = WorkloadConfig(num_tasks=400)

    checkpoint_path = "checkpoints/best_d3qn_scheduler.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Please run scripts/tune.py first.")
        sys.exit(1)

    print(f"Loading trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    rl_config = RLConfig(
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 2)
    )

    intelligent_scheduler = IntelligentD3QNScheduler(
        checkpoint_path=checkpoint_path,
        cluster_config=cluster_config,
        rl_config=rl_config
    )

    schedulers = [
        intelligent_scheduler,
        FIFOScheduler(),
        SJFScheduler(),
        EDFScheduler(),
        TetrisBestFitScheduler(),
        RoundRobinScheduler()
    ]

    engine = BenchmarkEngine(cluster_config, workload_config)
    results = engine.run_suite(
        schedulers=schedulers,
        num_tasks=400,
        num_runs=3,
        base_seed=1337
    )

    # Print ASCII table
    engine.print_comparison_table(results["summary"])

    # Export benchmark_results.json to root workspace
    output_json = "benchmark_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results exported to: {os.path.abspath(output_json)}")

    # Generate comparison plot
    output_plot = "benchmark_comparison.png"
    engine.generate_visual_plot(results["summary"], output_path=output_plot)

if __name__ == "__main__":
    main()
