"""
Benchmarking Engine for NeuraSched.
Runs controlled head-to-head simulations of Intelligent-D3QN against
FIFO, SJF, EDF, Tetris Best-Fit, and Round-Robin.
"""
import time
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import List, Dict, Any, Optional

from src.environment.simulator import ClusterSimulator, SimulationMetrics
from src.scheduler.base import BaseScheduler
from src.config import ClusterConfig, WorkloadConfig
from src.workload.trace_generator import TraceGenerator
from src.workload.task import Task

class BenchmarkEngine:
    def __init__(
        self,
        cluster_config: Optional[ClusterConfig] = None,
        workload_config: Optional[WorkloadConfig] = None
    ):
        self.cluster_config = cluster_config or ClusterConfig()
        self.workload_config = workload_config or WorkloadConfig()
        self.trace_gen = TraceGenerator(self.workload_config)

    def evaluate_scheduler(
        self,
        scheduler: BaseScheduler,
        task_trace: List[Task]
    ) -> SimulationMetrics:
        """Run single scheduler on deterministic trace with high-precision timing."""
        sim = ClusterSimulator(self.cluster_config)
        state, mask, current_task = sim.reset(task_trace)
        scheduler.reset()

        decision_latencies: List[float] = []

        while current_task is not None:
            t0 = time.perf_counter()
            chosen_node = scheduler.select_node(
                cluster=sim.cluster,
                task=current_task,
                action_mask=mask,
                current_time=sim.clock
            )
            lat = time.perf_counter() - t0
            decision_latencies.append(lat)

            next_state, next_mask, _, done, info = sim.step(chosen_node)

            if done:
                break
            state = next_state
            mask = next_mask
            current_task = info.get("task")

        return sim.get_metrics(decision_latencies)

    def run_suite(
        self,
        schedulers: List[BaseScheduler],
        num_tasks: int = 400,
        num_runs: int = 3,
        base_seed: int = 1337
    ) -> Dict[str, Any]:
        """
        Execute full benchmark across multiple randomized runs for statistical significance.
        """
        print(f"\n=========================================================================")
        print(f"  RUNNING BENCHMARK SUITE ({len(schedulers)} Schedulers, {num_tasks} Tasks, {num_runs} Runs)")
        print(f"=========================================================================\n")

        raw_results: Dict[str, List[Dict[str, Any]]] = {s.name: [] for s in schedulers}

        for run_idx in range(num_runs):
            seed = base_seed + (run_idx * 137)
            print(f">>> Benchmark Run {run_idx + 1}/{num_runs} (Seed: {seed})...")
            tasks = self.trace_gen.generate_trace(num_tasks=num_tasks, seed=seed)

            for scheduler in schedulers:
                metrics = self.evaluate_scheduler(scheduler, tasks)
                raw_results[scheduler.name].append(metrics.to_dict())

        # Aggregate averages across runs
        aggregated: Dict[str, Dict[str, float]] = {}
        for s_name, runs in raw_results.items():
            keys = runs[0].keys()
            aggregated[s_name] = {}
            for k in keys:
                if isinstance(runs[0][k], (int, float)):
                    aggregated[s_name][k] = round(float(np.mean([r[k] for r in runs])), 2)

        return {
            "summary": aggregated,
            "raw_runs": raw_results
        }

    def print_comparison_table(self, summary: Dict[str, Dict[str, float]]):
        """Format and print benchmark results as an ASCII table."""
        headers = [
            "Scheduler",
            "Makespan (s)",
            "Avg Turnaround (s)",
            "P99 Turnaround (s)",
            "Avg Wait (s)",
            "CPU Util (%)",
            "RAM Util (%)",
            "Stranded (%)",
            "SLA Viol (%)"
        ]

        rows = []
        for name, m in summary.items():
            rows.append([
                name,
                f"{m['makespan']:.1f}",
                f"{m['avg_turnaround_time']:.1f}",
                f"{m['p99_turnaround_time']:.1f}",
                f"{m['avg_wait_time']:.1f}",
                f"{m['avg_cpu_utilization']:.1f}%",
                f"{m['avg_ram_utilization']:.1f}%",
                f"{m['stranded_resource_ratio']:.1f}%",
                f"{m['sla_violation_rate']:.1f}%"
            ])

        table_str = tabulate(rows, headers=headers, tablefmt="fancy_grid")
        print("\n" + table_str + "\n")
        return table_str

    def generate_visual_plot(self, summary: Dict[str, Dict[str, float]], output_path: str = "benchmark_comparison.png"):
        """Generate 4-panel comparison figure."""
        schedulers = list(summary.keys())
        colors = ["#2ecc71" if "Intelligent" in s else "#3498db" for s in schedulers]

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Intelligent Task Scheduler vs. Classical Baselines", fontsize=16, fontweight="bold")

        # 1. Makespan
        makespans = [summary[s]["makespan"] for s in schedulers]
        axs[0, 0].bar(schedulers, makespans, color=colors, edgecolor="black", alpha=0.85)
        axs[0, 0].set_title("Total Makespan (Lower is Better)", fontweight="bold")
        axs[0, 0].set_ylabel("Seconds")
        axs[0, 0].tick_params(axis="x", rotation=25)
        axs[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # 2. SLA Violation Rate
        sla_viols = [summary[s]["sla_violation_rate"] for s in schedulers]
        axs[0, 1].bar(schedulers, sla_viols, color=colors, edgecolor="black", alpha=0.85)
        axs[0, 1].set_title("SLA Violation Rate (%) (Lower is Better)", fontweight="bold")
        axs[0, 1].set_ylabel("Violation %")
        axs[0, 1].tick_params(axis="x", rotation=25)
        axs[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

        # 3. Resource Utilization (CPU & RAM)
        x = np.arange(len(schedulers))
        width = 0.35
        cpu_utils = [summary[s]["avg_cpu_utilization"] for s in schedulers]
        ram_utils = [summary[s]["avg_ram_utilization"] for s in schedulers]
        axs[1, 0].bar(x - width/2, cpu_utils, width, label="CPU Util (%)", color="#34495e", edgecolor="black")
        axs[1, 0].bar(x + width/2, ram_utils, width, label="RAM Util (%)", color="#e67e22", edgecolor="black")
        axs[1, 0].set_title("Cluster Resource Utilization (Higher is Better)", fontweight="bold")
        axs[1, 0].set_ylabel("Utilization %")
        axs[1, 0].set_xticks(x)
        axs[1, 0].set_xticklabels(schedulers, rotation=25)
        axs[1, 0].legend()
        axs[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

        # 4. Average Task Turnaround Time
        turnarounds = [summary[s]["avg_turnaround_time"] for s in schedulers]
        axs[1, 1].bar(schedulers, turnarounds, color=colors, edgecolor="black", alpha=0.85)
        axs[1, 1].set_title("Average Task Turnaround Time (Lower is Better)", fontweight="bold")
        axs[1, 1].set_ylabel("Seconds")
        axs[1, 1].tick_params(axis="x", rotation=25)
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        svg_path = output_path.replace(".png", ".svg")
        plt.savefig(svg_path, format="svg")
        plt.close()
        print(f"Saved benchmark visualizations to: {output_path} and {svg_path}")
