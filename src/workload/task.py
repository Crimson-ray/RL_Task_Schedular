"""
Task definition for distributed cluster scheduling.
Tracks resource requirements, lifecycle timings, priority tiers, and SLA metrics.
"""
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class Task:
    task_id: Union[int, str]
    arrival_time: float
    cpu_req: float         # CPU cores requested
    ram_req: float         # RAM in GB requested
    duration: float        # Execution time in seconds
    priority: int          # 0: Batch (low), 1: Standard (medium), 2: Latency-Critical (high)
    deadline: float        # Absolute SLA deadline timestamp
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    assigned_node_id: Optional[int] = None

    @property
    def is_completed(self) -> bool:
        return self.finish_time is not None

    @property
    def wait_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return max(0.0, self.start_time - self.arrival_time)

    @property
    def turnaround_time(self) -> float:
        if self.finish_time is None:
            return 0.0
        return max(0.0, self.finish_time - self.arrival_time)

    @property
    def tardiness(self) -> float:
        """Tardiness: max(0, finish_time - deadline). Zero if met."""
        if self.finish_time is None:
            return 0.0
        return max(0.0, self.finish_time - self.deadline)

    @property
    def is_sla_violated(self) -> bool:
        if self.finish_time is None:
            return False
        return self.finish_time > self.deadline

    def reset(self):
        """Reset execution state for replay under different schedulers."""
        self.start_time = None
        self.finish_time = None
        self.assigned_node_id = None

    def clone(self) -> 'Task':
        """Deep clone of task for independent simulation runs."""
        return Task(
            task_id=self.task_id,
            arrival_time=self.arrival_time,
            cpu_req=self.cpu_req,
            ram_req=self.ram_req,
            duration=self.duration,
            priority=self.priority,
            deadline=self.deadline,
        )
