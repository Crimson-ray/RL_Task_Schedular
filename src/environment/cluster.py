"""
Cluster Management Layer for NeuraSched.
Coordinates 6 heterogeneous compute nodes, maintains cluster-wide statistics,
constructs the 45-dimensional state vector S_t, and generates the 7-action feasibility mask.
"""
import numpy as np
from typing import List, Optional, Tuple
from src.environment.node import ComputeNode
from src.workload.task import Task
from src.config import ClusterConfig

class Cluster:
    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig()
        self.nodes: List[ComputeNode] = []
        self._init_nodes()

    def _init_nodes(self):
        self.nodes.clear()
        for idx, spec in enumerate(self.config.heterogeneous_specs):
            if len(spec) == 4:
                cpu, ram, ntype, speed = spec
            else:
                cpu, ram, ntype = spec
                speed = 1.0
            self.nodes.append(ComputeNode(
                node_id=idx,
                cpu_capacity=cpu,
                ram_capacity=ram,
                node_type=ntype,
                processing_speed=speed
            ))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def action_dim(self) -> int:
        """6 compute node placements + 1 Defer/Hold action = 7 actions."""
        return self.num_nodes + 1

    @property
    def total_cpu_capacity(self) -> float:
        return sum(n.cpu_capacity for n in self.nodes)

    @property
    def total_ram_capacity(self) -> float:
        return sum(n.ram_capacity for n in self.nodes)

    @property
    def total_allocated_cpu(self) -> float:
        return sum(n.allocated_cpu for n in self.nodes)

    @property
    def total_allocated_ram(self) -> float:
        return sum(n.allocated_ram for n in self.nodes)

    @property
    def cluster_cpu_utilization(self) -> float:
        total = self.total_cpu_capacity
        return (self.total_allocated_cpu / total) if total > 0 else 0.0

    @property
    def cluster_ram_utilization(self) -> float:
        total = self.total_ram_capacity
        return (self.total_allocated_ram / total) if total > 0 else 0.0

    @property
    def cluster_load_variance(self) -> float:
        utils = [n.cpu_utilization for n in self.nodes]
        return float(np.var(utils))

    @property
    def cluster_stranded_ratio(self) -> float:
        return float(np.mean([n.stranded_resource_ratio for n in self.nodes]))

    def get_action_mask(self, task: Task, allow_defer: bool = True) -> np.ndarray:
        """
        Boolean feasibility mask of shape (7,) for NeuraSched:
        Actions 0-5: Node placement feasibility (True if node has capacity).
        Action 6: Defer/hold in queue (True if no node is feasible, or strategic deferral is valid).
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        for i, node in enumerate(self.nodes):
            if node.can_accommodate(task.cpu_req, task.ram_req):
                mask[i] = True

        has_running = any(len(n.running_tasks) > 0 for n in self.nodes)
        has_feasible_node = np.any(mask[:self.num_nodes])

        if not has_feasible_node:
            mask[self.num_nodes] = True
        elif allow_defer and has_running:
            mask[self.num_nodes] = True
        else:
            mask[self.num_nodes] = False

        return mask

    def release_completed(self, current_time: float) -> List[Task]:
        all_completed = []
        for node in self.nodes:
            all_completed.extend(node.release_completed(current_time))
        return all_completed

    def earliest_completion_time(self) -> Optional[float]:
        times = [n.earliest_completion_time() for n in self.nodes]
        valid_times = [t for t in times if t is not None]
        return min(valid_times) if valid_times else None

    def calculate_packing_alignment(self, node_id: int, task: Task) -> float:
        """Cosine function multi-resource packing alignment: r_pack."""
        node = self.nodes[node_id]
        demand = np.array([task.cpu_req, task.ram_req], dtype=np.float32)
        residual = np.array([node.available_cpu, node.available_ram], dtype=np.float32)

        norm_d = np.linalg.norm(demand)
        norm_r = np.linalg.norm(residual)
        if norm_d < 1e-6 or norm_r < 1e-6:
            return 0.0
        val = np.dot(demand, residual) / (norm_d * norm_r)
        return float(np.clip(val, 0.0, 1.0))

    def get_state_vector(self, task: Task, queue_len: int, current_time: float) -> np.ndarray:
        """
        NeuraSched State Encoder: S_t = [f_task, f_nodes, f_cluster] in R^45.
        - f_task: 5 dim [Normalized CPU, RAM, duration, slack, priority]
        - f_nodes: 36 dim (6 nodes * 6 features per node) [Free CPU/RAM ratio, speed]
        - f_cluster: 4 dim [Cluster CPU/RAM util, queue backlog, load variance]
        """
        # 1. Task features (5 dim)
        slack = max(0.0, task.deadline - current_time) / max(1.0, task.duration)
        f_task = np.array([
            task.cpu_req / 32.0,
            task.ram_req / 64.0,
            task.duration / 120.0,
            min(5.0, slack) / 5.0,
            task.priority / 2.0
        ], dtype=np.float32)

        # 2. Node features (6 nodes * 6 features = 36 dim)
        f_nodes = np.concatenate([n.to_feature_vector() for n in self.nodes])

        # 3. Cluster global features (4 dim)
        f_cluster = np.array([
            self.cluster_cpu_utilization,
            self.cluster_ram_utilization,
            min(1.0, queue_len / 50.0),
            self.cluster_load_variance
        ], dtype=np.float32)

        state = np.concatenate([f_task, f_nodes, f_cluster]).astype(np.float32)
        assert len(state) == 45, f"Expected state dimension 45, got {len(state)}"
        return state

    def state_dim(self) -> int:
        dummy_task = Task(0, 0.0, 1.0, 1.0, 1.0, 0, 10.0)
        return len(self.get_state_vector(dummy_task, 0, 0.0))

    def reset(self):
        for node in self.nodes:
            node.reset()
