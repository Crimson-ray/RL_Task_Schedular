"""
Abstract Base Scheduler Interface.
All scheduling algorithms (heuristics and deep RL models) inherit from this class.
"""
from abc import ABC, abstractmethod
import numpy as np
from src.environment.cluster import Cluster
from src.workload.task import Task

class BaseScheduler(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_node(
        self,
        cluster: Cluster,
        task: Task,
        action_mask: np.ndarray,
        current_time: float
    ) -> int:
        """
        Choose node index to place incoming task.
        Must return an integer in [0, cluster.num_nodes - 1].
        If action_mask has valid nodes, should strictly return a valid index.
        """
        pass

    def reset(self):
        """Reset internal state if any."""
        pass
