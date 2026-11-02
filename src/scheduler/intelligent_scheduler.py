"""
Production-grade Intelligent Task Scheduler using Dueling Double Deep Q-Networks.
Combines dynamic action masking, multi-resource vector alignment, and sub-millisecond inference.
"""
import time
import os
import torch
import numpy as np
from typing import Optional
from src.scheduler.base import BaseScheduler
from src.environment.cluster import Cluster
from src.workload.task import Task
from src.models.dueling_dqn import DuelingDQN
from src.config import ClusterConfig, RLConfig

class IntelligentD3QNScheduler(BaseScheduler):
    def __init__(
        self,
        model: Optional[DuelingDQN] = None,
        checkpoint_path: Optional[str] = None,
        cluster_config: Optional[ClusterConfig] = None,
        rl_config: Optional[RLConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__("Intelligent-D3QN")
        self.cluster_config = cluster_config or ClusterConfig()
        self.rl_config = rl_config or RLConfig()

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(self.rl_config.device)

        dummy_cluster = Cluster(self.cluster_config)
        self.state_dim = dummy_cluster.state_dim()
        self.action_dim = dummy_cluster.action_dim

        hidden_dim = self.rl_config.hidden_dim
        num_layers = self.rl_config.num_layers

        saved_checkpoint = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            saved_checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "hidden_dim" in saved_checkpoint:
                hidden_dim = saved_checkpoint["hidden_dim"]
            if "num_layers" in saved_checkpoint:
                num_layers = saved_checkpoint["num_layers"]

        if model is not None:
            self.policy_net = model.to(self.device)
        else:
            self.policy_net = DuelingDQN(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)

        if saved_checkpoint is not None:
            self.policy_net.load_state_dict(saved_checkpoint["model_state_dict"])

        self.policy_net.eval()

    def select_node(
        self,
        cluster: Cluster,
        task: Task,
        action_mask: np.ndarray,
        current_time: float
    ) -> int:
        """
        Sub-millisecond inference to place task on the optimal node.
        """
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            return 0
        if len(valid_indices) == 1:
            return int(valid_indices[0])

        state_vec = cluster.get_state_vector(task, queue_len=len(cluster.nodes), current_time=current_time)
        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor, mask_tensor)
            action = int(torch.argmax(q_values, dim=-1).item())

        return action

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.policy_net.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.rl_config.hidden_dim,
            "num_layers": self.rl_config.num_layers
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()
