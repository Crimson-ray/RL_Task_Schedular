"""
Discrete-Event Cluster Simulation Environment for NeuraSched.
Provides Gym-like step interface and deterministic multi-run evaluation.
Reward formula matches Page 2 architectural specification:
r_t = r_sla + r_pack + r_balance + r_speed - r_defer
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from src.environment.cluster import Cluster
from src.workload.task import Task
from src.config import ClusterConfig, RewardConfig

@dataclass
class SimulationMetrics:
    makespan: float
    avg_turnaround_time: float
    p95_turnaround_time: float
    p99_turnaround_time: float
    avg_wait_time: float
    avg_cpu_utilization: float
    avg_ram_utilization: float
    stranded_resource_ratio: float
    sla_violation_rate: float
    sla_violations_count: int
    critical_sla_violation_rate: float
    total_tasks: int
    mean_decision_latency_us: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "makespan": round(self.makespan, 2),
            "avg_turnaround_time": round(self.avg_turnaround_time, 2),
            "p95_turnaround_time": round(self.p95_turnaround_time, 2),
            "p99_turnaround_time": round(self.p99_turnaround_time, 2),
            "avg_wait_time": round(self.avg_wait_time, 2),
            "avg_cpu_utilization": round(self.avg_cpu_utilization * 100, 2),
            "avg_ram_utilization": round(self.avg_ram_utilization * 100, 2),
            "stranded_resource_ratio": round(self.stranded_resource_ratio * 100, 2),
            "sla_violation_rate": round(self.sla_violation_rate * 100, 2),
            "sla_violations_count": self.sla_violations_count,
            "critical_sla_violation_rate": round(self.critical_sla_violation_rate * 100, 2),
            "total_tasks": self.total_tasks,
            "mean_decision_latency_us": round(self.mean_decision_latency_us, 2)
        }

class ClusterSimulator:
    def __init__(
        self,
        cluster_config: Optional[ClusterConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        self.cluster = Cluster(cluster_config)
        self.reward_config = reward_config or RewardConfig()

        self.clock: float = 0.0
        self.pending_tasks: List[Task] = []
        self.active_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []

        self.time_samples: List[float] = []
        self.cpu_util_samples: List[float] = []
        self.ram_util_samples: List[float] = []
        self.stranded_samples: List[float] = []

        self._current_task_idx: int = 0
        self._task_stream: List[Task] = []
        self._task_defer_count: int = 0

    def reset(self, task_stream: List[Task]) -> Tuple[np.ndarray, np.ndarray, Optional[Task]]:
        self.cluster.reset()
        self.clock = 0.0
        self.pending_tasks = []
        self.active_tasks = []
        self.completed_tasks = []

        self.time_samples = [0.0]
        self.cpu_util_samples = [0.0]
        self.ram_util_samples = [0.0]
        self.stranded_samples = [0.0]

        self._task_stream = [t.clone() for t in task_stream]
        self._current_task_idx = 0
        self._task_defer_count = 0

        if not self._task_stream:
            return np.zeros(self.cluster.state_dim(), dtype=np.float32), np.zeros(self.cluster.action_dim, dtype=bool), None

        first_task = self._task_stream[0]
        self.clock = first_task.arrival_time
        mask = self.cluster.get_action_mask(first_task)
        state = self.cluster.get_state_vector(first_task, len(self._task_stream) - 1, self.clock)
        return state, mask, first_task

    def _sample_metrics(self):
        self.time_samples.append(self.clock)
        self.cpu_util_samples.append(self.cluster.cluster_cpu_utilization)
        self.ram_util_samples.append(self.cluster.cluster_ram_utilization)
        self.stranded_samples.append(self.cluster.cluster_stranded_ratio)

    def _advance_until_feasible(self, task: Task) -> bool:
        while not np.any(self.cluster.get_action_mask(task)[:self.cluster.num_nodes]):
            earliest = self.cluster.earliest_completion_time()
            if earliest is None:
                return False
            self.clock = max(self.clock, earliest)
            freed = self.cluster.release_completed(self.clock)
            self.completed_tasks.extend(freed)
            self._sample_metrics()
        return True

    def calculate_reward(self, node_id: int, task: Task) -> float:
        """
        NeuraSched shaped reward:
        r_t = r_sla + r_pack + r_balance + r_speed - r_defer
        """
        node = self.cluster.nodes[node_id]

        # 1. r_sla (SLA compliance / deadline margin)
        effective_duration = task.duration / node.processing_speed
        finish_est = self.clock + effective_duration
        if finish_est <= task.deadline:
            slack_margin = min(task.deadline - finish_est, task.duration) / max(1.0, task.duration)
            r_sla = 2.0 + (slack_margin * 2.0)
        else:
            tardiness = (finish_est - task.deadline) / max(1.0, task.duration)
            penalty_mult = 4.0 if task.priority == 2 else 2.0
            r_sla = -penalty_mult * tardiness

        # 2. r_pack (Cosine function alignment)
        r_pack = self.cluster.calculate_packing_alignment(node_id, task)

        # 3. r_balance (Imbalance penalty)
        r_balance = - (node.cpu_utilization ** 2 + node.ram_utilization ** 2) * 0.15

        # 4. r_speed (Speed efficiency: placing latency-sensitive jobs on faster hardware)
        if task.priority == 2:
            r_speed = 3.0 if node.processing_speed > 1.0 else -1.5
        elif task.priority == 0:
            r_speed = -0.5 if node.processing_speed > 1.0 else 0.5
        else:
            r_speed = 0.5 if node.processing_speed > 1.0 else 0.0

        total_r = (
            self.reward_config.weight_sla * r_sla
            + self.reward_config.weight_packing * r_pack
            + self.reward_config.weight_balance * r_balance
            + self.reward_config.weight_speed * r_speed
        )
        return float(total_r)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:
        """
        NeuraSched Gym-like step method:
        Actions 0-5: Allocate to ComputeNode 0-5
        Action 6: Defer / Hold in queue (advances time to next completion event)
        """
        if self._current_task_idx >= len(self._task_stream):
            return np.zeros(self.cluster.state_dim(), dtype=np.float32), np.zeros(self.cluster.action_dim, dtype=bool), 0.0, True, {}

        current_task = self._task_stream[self._current_task_idx]

        # Fast forward if clock is behind arrival time
        if self.clock < current_task.arrival_time:
            self.clock = current_task.arrival_time
            freed = self.cluster.release_completed(self.clock)
            self.completed_tasks.extend(freed)
            self._sample_metrics()

        mask = self.cluster.get_action_mask(current_task, allow_defer=(self._task_defer_count < 2))

        # Handle Action 6: Defer / Hold in queue
        if action == self.cluster.num_nodes:
            self._task_defer_count += 1
            feasible_now = np.any(mask[:self.cluster.num_nodes])
            earliest = self.cluster.earliest_completion_time()
            if earliest is not None:
                time_waited = max(0.0, earliest - self.clock)
                self.clock = max(self.clock, earliest)
                freed = self.cluster.release_completed(self.clock)
                self.completed_tasks.extend(freed)
                self._sample_metrics()
            else:
                time_waited = 0.5
                self.clock += 0.5

            if not feasible_now:
                # Physically necessary deferral: valid and correct
                reward = 0.5
            else:
                # Unnecessary deferral when idle nodes exist: heavily penalize delay
                reward = - (self.reward_config.weight_defer + 5.0 * (1.0 + time_waited) + (2.0 if current_task.priority == 2 else 0.0))

            # If deferred twice or no node feasible, advance until at least one node is feasible and allocate!
            mask = self.cluster.get_action_mask(current_task, allow_defer=(self._task_defer_count < 2))
            if self._task_defer_count >= 2 or not np.any(mask[:self.cluster.num_nodes]):
                self._advance_until_feasible(current_task)
                mask = self.cluster.get_action_mask(current_task, allow_defer=False)
                valid_nodes = np.where(mask[:self.cluster.num_nodes])[0]
                action = int(valid_nodes[0]) if len(valid_nodes) > 0 else 0
                self.cluster.nodes[action].allocate(current_task, self.clock)
                self._sample_metrics()
                reward += self.calculate_reward(action, current_task)
                self._task_defer_count = 0
                self._current_task_idx += 1
                done = (self._current_task_idx >= len(self._task_stream))
                if not done:
                    next_task = self._task_stream[self._current_task_idx]
                    if self.clock < next_task.arrival_time:
                        self.clock = next_task.arrival_time
                        freed = self.cluster.release_completed(self.clock)
                        self.completed_tasks.extend(freed)
                    next_mask = self.cluster.get_action_mask(next_task)
                    remaining_queue = len(self._task_stream) - self._current_task_idx
                    next_state = self.cluster.get_state_vector(next_task, remaining_queue, self.clock)
                else:
                    self.drain_cluster()
                    next_state = np.zeros(self.cluster.state_dim(), dtype=np.float32)
                    next_mask = np.zeros(self.cluster.action_dim, dtype=bool)
                return next_state, next_mask, reward, done, {"task": current_task}

            # Still in deferral: task stays current_task
            remaining_queue = len(self._task_stream) - self._current_task_idx
            next_state = self.cluster.get_state_vector(current_task, remaining_queue, self.clock)
            return next_state, mask, reward, False, {"task": current_task}

        # Node allocation (Actions 0..5)
        self._task_defer_count = 0
        if not mask[action]:
            self._advance_until_feasible(current_task)
            mask = self.cluster.get_action_mask(current_task, allow_defer=False)
            if not mask[action]:
                valid_nodes = np.where(mask[:self.cluster.num_nodes])[0]
                action = int(valid_nodes[0]) if len(valid_nodes) > 0 else 0

        # Allocate on chosen node
        self.cluster.nodes[action].allocate(current_task, self.clock)
        self._sample_metrics()

        reward = self.calculate_reward(action, current_task)

        # Move to next task
        self._current_task_idx += 1
        done = (self._current_task_idx >= len(self._task_stream))

        if not done:
            next_task = self._task_stream[self._current_task_idx]
            if self.clock < next_task.arrival_time:
                self.clock = next_task.arrival_time
                freed = self.cluster.release_completed(self.clock)
                self.completed_tasks.extend(freed)

            if not np.any(self.cluster.get_action_mask(next_task)[:self.cluster.num_nodes]):
                self._advance_until_feasible(next_task)

            next_mask = self.cluster.get_action_mask(next_task)
            remaining_queue = len(self._task_stream) - self._current_task_idx
            next_state = self.cluster.get_state_vector(next_task, remaining_queue, self.clock)
        else:
            self.drain_cluster()
            next_state = np.zeros(self.cluster.state_dim(), dtype=np.float32)
            next_mask = np.zeros(self.cluster.action_dim, dtype=bool)

        return next_state, next_mask, reward, done, {"task": current_task}

    def drain_cluster(self):
        while True:
            earliest = self.cluster.earliest_completion_time()
            if earliest is None:
                break
            self.clock = earliest
            freed = self.cluster.release_completed(self.clock)
            self.completed_tasks.extend(freed)
            self._sample_metrics()

    def get_metrics(self, decision_latencies: List[float]) -> SimulationMetrics:
        self.drain_cluster()
        all_tasks = self.completed_tasks

        if not all_tasks:
            return SimulationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        turnarounds = [t.turnaround_time for t in all_tasks]
        waits = [t.wait_time for t in all_tasks]
        sla_viols = [t for t in all_tasks if t.is_sla_violated]
        crit_tasks = [t for t in all_tasks if t.priority == 2]
        crit_sla_viols = [t for t in crit_tasks if t.is_sla_violated]

        if len(self.time_samples) > 1:
            dt = np.diff(self.time_samples)
            total_time = self.time_samples[-1] - self.time_samples[0]
            if total_time > 0:
                avg_cpu = float(np.sum(np.array(self.cpu_util_samples[:-1]) * dt) / total_time)
                avg_ram = float(np.sum(np.array(self.ram_util_samples[:-1]) * dt) / total_time)
                avg_stranded = float(np.sum(np.array(self.stranded_samples[:-1]) * dt) / total_time)
            else:
                avg_cpu = float(np.mean(self.cpu_util_samples))
                avg_ram = float(np.mean(self.ram_util_samples))
                avg_stranded = float(np.mean(self.stranded_samples))
        else:
            avg_cpu = 0.0
            avg_ram = 0.0
            avg_stranded = 0.0

        makespan = max(t.finish_time for t in all_tasks if t.finish_time is not None) - min(t.arrival_time for t in all_tasks)
        mean_latency = float(np.mean(decision_latencies) * 1e6) if decision_latencies else 0.0

        return SimulationMetrics(
            makespan=makespan,
            avg_turnaround_time=float(np.mean(turnarounds)),
            p95_turnaround_time=float(np.percentile(turnarounds, 95)),
            p99_turnaround_time=float(np.percentile(turnarounds, 99)),
            avg_wait_time=float(np.mean(waits)),
            avg_cpu_utilization=avg_cpu,
            avg_ram_utilization=avg_ram,
            stranded_resource_ratio=avg_stranded,
            sla_violation_rate=len(sla_viols) / len(all_tasks),
            sla_violations_count=len(sla_viols),
            critical_sla_violation_rate=(len(crit_sla_viols) / len(crit_tasks)) if crit_tasks else 0.0,
            total_tasks=len(all_tasks),
            mean_decision_latency_us=mean_latency
        )
