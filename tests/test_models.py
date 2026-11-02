"""
Unit tests for Dueling DQN and Prioritized Replay Buffer.
"""
import torch
import numpy as np
from src.models.dueling_dqn import DuelingDQN
from src.models.replay_buffer import PrioritizedReplayBuffer

def test_dueling_dqn_action_masking():
    state_dim = 20
    action_dim = 4
    model = DuelingDQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=32, num_layers=2)

    state = torch.randn(1, state_dim)
    # Mask where only action 2 is feasible
    mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)

    q_vals = model(state, mask)
    assert q_vals.shape == (1, action_dim)
    # Infeasible actions must have logits near -1e9
    assert q_vals[0, 0].item() <= -1e8
    assert q_vals[0, 1].item() <= -1e8
    assert q_vals[0, 3].item() <= -1e8
    assert q_vals[0, 2].item() > -1e5

    # Argmax should strictly pick action 2
    action = model.select_action(state, mask, epsilon=0.0)
    assert action == 2

def test_prioritized_replay_buffer():
    capacity = 100
    state_dim = 10
    action_dim = 4
    buf = PrioritizedReplayBuffer(capacity=capacity, state_dim=state_dim, action_dim=action_dim)

    # Push transitions
    for i in range(20):
        s = np.ones(state_dim, dtype=np.float32) * i
        a = i % action_dim
        r = float(i)
        ns = s + 1.0
        m = np.ones(action_dim, dtype=bool)
        done = False
        buf.push(s, a, r, ns, m, done)

    assert len(buf) == 20

    # Sample batch
    batch = buf.sample(batch_size=8)
    assert batch["states"].shape == (8, state_dim)
    assert batch["actions"].shape == (8,)
    assert batch["weights"].shape == (8,)

    # Update priorities
    td_errors = np.ones(8, dtype=np.float32) * 2.0
    buf.update_priorities(batch["tree_indices"], td_errors)
