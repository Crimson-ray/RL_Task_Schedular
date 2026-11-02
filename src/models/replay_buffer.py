"""
Prioritized Experience Replay (PER) implementation using a Sum-Tree.
Enables prioritized transition sampling with importance sampling weights
to correct distribution shift.
"""
import numpy as np
import torch
from typing import Tuple, Optional

class SumTree:
    """Binary Sum-Tree where leaves store transition priorities."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data_pointer = 0
        self.size = 0

    def add(self, priority: float):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_idx: int, priority: float):
        priority = float(np.clip(priority, 1e-4, 100.0))
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def total_priority(self) -> float:
        tot = float(self.tree[0])
        return max(1e-4, tot)

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """Traverse tree to find leaf node corresponding to sampled value."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), data_idx

class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 10000
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Memory pre-allocation
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool
    ):
        idx = self.tree.data_pointer
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.next_masks[idx] = next_mask
        self.dones[idx] = 1.0 if done else 0.0

        # Assign max priority for fresh transitions
        priority = (self.max_priority + 1e-5) ** self.alpha
        self.tree.add(priority)

    def sample(self, batch_size: int, device: str = "cpu"):
        assert self.tree.size >= batch_size, "Buffer does not have enough transitions yet."

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1

        total_p = max(1e-4, self.tree.total_priority())
        segment = total_p / batch_size
        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            a = max(0.0, segment * i)
            b = max(a + 1e-5, min(total_p, segment * (i + 1)))
            val = float(np.random.uniform(a, b))
            tree_idx, p, data_idx = self.tree.get_leaf(val)
            tree_indices[i] = tree_idx
            priorities[i] = max(1e-5, p)
            indices[i] = data_idx

        # Compute importance-sampling weights defensively
        probs = np.clip(priorities / total_p, 1e-7, 1.0)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights /= max(1e-5, float(weights.max()))

        batch = {
            "states": torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.int64, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            "next_masks": torch.as_tensor(self.next_masks[indices], dtype=torch.bool, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
            "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
            "tree_indices": tree_indices
        }
        return batch

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        td_clean = np.nan_to_num(td_errors, nan=1.0, posinf=100.0, neginf=1.0)
        td_clean = np.clip(np.abs(td_clean), 1e-4, 100.0)
        for idx, td in zip(tree_indices, td_clean):
            p = float(td ** self.alpha)
            self.max_priority = max(self.max_priority, float(td))
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.size
