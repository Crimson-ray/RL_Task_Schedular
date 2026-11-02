"""
Industrial Cluster Trace Generator (Google Borg & Alibaba inspired).
Generates realistic multi-resource workloads with bursty arrivals,
heavy-tailed durations, heterogeneous resource profiles, and SLA deadlines.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from src.workload.task import Task
from src.config import WorkloadConfig

class TraceGenerator:
    def __init__(self, config: Optional[WorkloadConfig] = None):
        self.config = config or WorkloadConfig()

    def generate_trace(self, num_tasks: Optional[int] = None, seed: Optional[int] = None) -> List[Task]:
        """
        Generate a list of simulated cluster tasks.
        Uses seed for strict deterministic reproducibility across benchmarks.
        """
        if seed is not None:
            np.random.seed(seed)

        n = num_tasks if num_tasks is not None else self.config.num_tasks
        tasks: List[Task] = []

        current_time = 0.0
        in_burst = False
        burst_remaining = 0

        # Pre-sample priorities: 0 (Batch), 1 (Standard), 2 (Latency-Critical)
        priorities = np.random.choice(
            [0, 1, 2],
            size=n,
            p=self.config.priority_distribution
        )

        for i in range(n):
            # Bursty arrival model
            if burst_remaining <= 0:
                in_burst = (np.random.random() < self.config.burst_probability)
                burst_remaining = np.random.randint(5, 20) if in_burst else np.random.randint(10, 40)

            eff_rate = self.config.arrival_rate * (self.config.burst_multiplier if in_burst else 1.0)
            inter_arrival = np.random.exponential(1.0 / eff_rate)
            current_time += inter_arrival

            # Heavy-tailed execution duration (Pareto distribution bounded)
            raw_duration = (np.random.pareto(self.config.pareto_shape) + 1.0) * self.config.min_duration
            duration = float(np.clip(raw_duration, self.config.min_duration, self.config.max_duration))

            # Task profile: Balanced, Compute-heavy, or Memory-heavy
            profile_type = np.random.choice(["balanced", "compute_heavy", "memory_heavy"], p=[0.4, 0.3, 0.3])
            
            if profile_type == "compute_heavy":
                cpu_req = float(np.random.uniform(8.0, self.config.cpu_range[1]))
                ram_req = float(np.random.uniform(self.config.ram_range[0], 16.0))
            elif profile_type == "memory_heavy":
                cpu_req = float(np.random.uniform(self.config.cpu_range[0], 8.0))
                ram_req = float(np.random.uniform(16.0, self.config.ram_range[1]))
            else:
                cpu_req = float(np.random.uniform(self.config.cpu_range[0], self.config.cpu_range[1]))
                ram_req = float(np.random.uniform(self.config.ram_range[0], self.config.ram_range[1]))

            # Round to clean fractional increments (0.5 cores, 1.0 GB)
            cpu_req = round(cpu_req * 2) / 2.0
            ram_req = round(ram_req)

            priority = int(priorities[i])
            slack_factor = self.config.sla_slack_factors[priority]
            deadline = current_time + (duration * slack_factor)

            tasks.append(Task(
                task_id=i,
                arrival_time=round(current_time, 3),
                cpu_req=cpu_req,
                ram_req=ram_req,
                duration=round(duration, 2),
                priority=priority,
                deadline=round(deadline, 2)
            ))

        return tasks

    @staticmethod
    def save_to_csv(tasks: List[Task], filepath: str) -> None:
        """Export generated workload to CSV file."""
        records = [{
            "task_id": t.task_id,
            "arrival_time": t.arrival_time,
            "cpu_req": t.cpu_req,
            "ram_req": t.ram_req,
            "duration": t.duration,
            "priority": t.priority,
            "deadline": t.deadline
        } for t in tasks]
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)

    @staticmethod
    def load_from_csv(filepath: str) -> List[Task]:
        """Load tasks from CSV file."""
        df = pd.read_csv(filepath)
        tasks = []
        for _, row in df.iterrows():
            tasks.append(Task(
                task_id=int(row["task_id"]),
                arrival_time=float(row["arrival_time"]),
                cpu_req=float(row["cpu_req"]),
                ram_req=float(row["ram_req"]),
                duration=float(row["duration"]),
                priority=int(row["priority"]),
                deadline=float(row["deadline"])
            ))
        return tasks
