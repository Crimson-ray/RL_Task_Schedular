"""
Classical and Industrial Heuristic Schedulers.
Serves as competitive baselines against the Intelligent RL Scheduler.
"""
import numpy as np
from typing import Optional
from src.scheduler.base import BaseScheduler
from src.environment.cluster import Cluster
from src.workload.task import Task

class FIFOScheduler(BaseScheduler):
    """
    First-In First-Out (FIFO) Baseline:
    Schedules task onto the lowest index feasible node, or Defers if none feasible.
    """
    def __init__(self):
        super().__init__("FIFO")

    def select_node(self, cluster: Cluster, task: Task, action_mask: np.ndarray, current_time: float) -> int:
        valid_indices = np.where(action_mask[:cluster.num_nodes])[0]
        if len(valid_indices) == 0:
            return cluster.num_nodes # Defer (Action 6)
        return int(valid_indices[0])

class SJFScheduler(BaseScheduler):
    """
    Shortest Job First / Least Work Backlog Baseline:
    Selects the feasible node with the minimum remaining execution backlog.
    """
    def __init__(self):
        super().__init__("SJF-Backlog")

    def select_node(self, cluster: Cluster, task: Task, action_mask: np.ndarray, current_time: float) -> int:
        valid_indices = np.where(action_mask[:cluster.num_nodes])[0]
        if len(valid_indices) == 0:
            return cluster.num_nodes # Defer (Action 6)

        best_node = valid_indices[0]
        min_backlog = float("inf")

        for node_idx in valid_indices:
            node = cluster.nodes[node_idx]
            backlog = sum(max(0.0, (t.finish_time or current_time) - current_time) for t in node.running_tasks)
            if backlog < min_backlog:
                min_backlog = backlog
                best_node = node_idx

        return int(best_node)

class EDFScheduler(BaseScheduler):
    """
    Earliest Deadline First (EDF) Baseline:
    Prefers feasible nodes that guarantee maximum safety margin before deadline.
    """
    def __init__(self):
        super().__init__("EDF")

    def select_node(self, cluster: Cluster, task: Task, action_mask: np.ndarray, current_time: float) -> int:
        valid_indices = np.where(action_mask[:cluster.num_nodes])[0]
        if len(valid_indices) == 0:
            return cluster.num_nodes # Defer (Action 6)

        best_node = valid_indices[0]
        max_slack = -float("inf")

        for node_idx in valid_indices:
            node = cluster.nodes[node_idx]
            projected_finish = current_time + task.duration
            slack = task.deadline - projected_finish
            load_factor = (node.cpu_utilization + node.ram_utilization) / 2.0
            score = slack - (load_factor * 10.0)

            if score > max_slack:
                max_slack = score
                best_node = node_idx

        return int(best_node)

class TetrisBestFitScheduler(BaseScheduler):
    """
    Tetris / Borg Multi-Resource Vector Packing Heuristic:
    Maximizes cosine alignment between task resource demand vector
    and node residual capacity vector to prevent resource stranding.
    """
    def __init__(self):
        super().__init__("Tetris-BestFit")

    def select_node(self, cluster: Cluster, task: Task, action_mask: np.ndarray, current_time: float) -> int:
        valid_indices = np.where(action_mask[:cluster.num_nodes])[0]
        if len(valid_indices) == 0:
            return cluster.num_nodes # Defer (Action 6)

        best_node = valid_indices[0]
        best_alignment = -float("inf")

        for node_idx in valid_indices:
            alignment = cluster.calculate_packing_alignment(node_idx, task)
            if alignment > best_alignment:
                best_alignment = alignment
                best_node = node_idx

        return int(best_node)

class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin Baseline:
    Distributes tasks cyclically across feasible nodes.
    """
    def __init__(self):
        super().__init__("Round-Robin")
        self.last_idx = -1

    def select_node(self, cluster: Cluster, task: Task, action_mask: np.ndarray, current_time: float) -> int:
        valid_indices = np.where(action_mask[:cluster.num_nodes])[0]
        if len(valid_indices) == 0:
            return cluster.num_nodes # Defer (Action 6)

        num_nodes = cluster.num_nodes
        for step in range(1, num_nodes + 1):
            candidate = (self.last_idx + step) % num_nodes
            if action_mask[candidate]:
                self.last_idx = candidate
                return candidate

        self.last_idx = valid_indices[0]
        return int(valid_indices[0])

    def reset(self):
        self.last_idx = -1
