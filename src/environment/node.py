"""
Compute Node representation in a heterogeneous cluster.
Dynamically tracks CPU and memory allocations, node processing speed,
running tasks, and resource alignment for NeuraSched.
"""
import numpy as np
from typing import List, Optional
from src.workload.task import Task

class ComputeNode:
    def __init__(
        self,
        node_id: int,
        cpu_capacity: float,
        ram_capacity: float,
        node_type: str = "general-purpose",
        processing_speed: float = 1.0
    ):
        self.node_id = node_id
        self.cpu_capacity = float(cpu_capacity)
        self.ram_capacity = float(ram_capacity)
        self.node_type = node_type
        self.processing_speed = float(processing_speed)

        self.allocated_cpu: float = 0.0
        self.allocated_ram: float = 0.0
        self.running_tasks: List[Task] = []
        self.total_tasks_processed: int = 0

    @property
    def available_cpu(self) -> float:
        return max(0.0, self.cpu_capacity - self.allocated_cpu)

    @property
    def available_ram(self) -> float:
        return max(0.0, self.ram_capacity - self.allocated_ram)

    @property
    def cpu_utilization(self) -> float:
        return (self.allocated_cpu / self.cpu_capacity) if self.cpu_capacity > 0 else 0.0

    @property
    def ram_utilization(self) -> float:
        return (self.allocated_ram / self.ram_capacity) if self.ram_capacity > 0 else 0.0

    @property
    def stranded_resource_ratio(self) -> float:
        return abs(self.cpu_utilization - self.ram_utilization)

    def can_accommodate(self, cpu_req: float, ram_req: float) -> bool:
        return (self.available_cpu >= cpu_req - 1e-6) and (self.available_ram >= ram_req - 1e-6)

    def allocate(self, task: Task, current_time: float) -> bool:
        if not self.can_accommodate(task.cpu_req, task.ram_req):
            return False

        task.start_time = current_time
        # Effective duration reflects node processing speed:
        effective_duration = task.duration / self.processing_speed
        task.finish_time = current_time + effective_duration
        task.assigned_node_id = self.node_id

        self.allocated_cpu += task.cpu_req
        self.allocated_ram += task.ram_req
        self.running_tasks.append(task)
        self.total_tasks_processed += 1
        return True

    def release_completed(self, current_time: float) -> List[Task]:
        still_running: List[Task] = []
        completed: List[Task] = []

        for task in self.running_tasks:
            if task.finish_time is not None and task.finish_time <= current_time + 1e-6:
                self.allocated_cpu = max(0.0, self.allocated_cpu - task.cpu_req)
                self.allocated_ram = max(0.0, self.allocated_ram - task.ram_req)
                completed.append(task)
            else:
                still_running.append(task)

        self.running_tasks = still_running
        return completed

    def earliest_completion_time(self) -> Optional[float]:
        if not self.running_tasks:
            return None
        return min(t.finish_time for t in self.running_tasks if t.finish_time is not None)

    def reset(self):
        self.allocated_cpu = 0.0
        self.allocated_ram = 0.0
        self.running_tasks.clear()
        self.total_tasks_processed = 0

    def to_feature_vector(self) -> np.ndarray:
        """
        NeuraSched per-node feature vector (6 features per node):
        1. Free CPU ratio: available_cpu / capacity
        2. Free RAM ratio: available_ram / capacity
        3. Node CPU utilization
        4. Node RAM utilization
        5. CPU-RAM alignment (free CPU/RAM ratio)
        6. Normalized node processing speed
        Total for 6 nodes = 36 dimensions.
        """
        free_ratio = min(5.0, self.available_cpu / max(1.0, self.available_ram)) / 5.0
        return np.array([
            self.available_cpu / self.cpu_capacity,
            self.available_ram / self.ram_capacity,
            self.cpu_utilization,
            self.ram_utilization,
            free_ratio,
            self.processing_speed / 1.5
        ], dtype=np.float32)
