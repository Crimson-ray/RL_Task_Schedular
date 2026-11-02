"""
Dueling Deep Q-Network with Dynamic Action Masking.
Decouples state-value estimation V(s) from action advantages A(s, a).
Guarantees zero-allocation to infeasible nodes via dynamic action masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DuelingDQN(nn.Module):
    def __init__(
        self,
        state_dim: int = 45,
        action_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super(DuelingDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # NeuraSched Backbone: Linear(45->128) -> LayerNorm(128) + LeakyReLU -> Linear(128->128) -> LayerNorm(128) + LeakyReLU
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Value Stream [how good is our state]: Linear(128->64) + LeakyReLU -> Linear(64->1) -> Scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1)
        )

        # Advantage Stream [what is the better action]: Linear(128->64) + LeakyReLU -> Linear(64->7) -> A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        state: Tensor of shape (batch_size, state_dim) or (state_dim,)
        mask: Optional boolean Tensor of shape (batch_size, action_dim) or (action_dim,)
              where True = Feasible, False = Infeasible
        Returns:
        Q-values of shape (batch_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)

        features = self.feature_extractor(state)
        values = self.value_stream(features)         # (batch_size, 1)
        advantages = self.advantage_stream(features) # (batch_size, action_dim)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))

        # Dynamic Action Masking: mask infeasible actions with -1e9
        if mask is not None:
            # Ensure mask is boolean
            bool_mask = mask.bool()
            # If a row has all False (no node feasible), keep original to avoid all -inf
            has_valid = bool_mask.any(dim=-1, keepdim=True)
            safe_mask = torch.where(has_valid, bool_mask, torch.ones_like(bool_mask))
            q_values = torch.where(safe_mask, q_values, torch.tensor(-1e9, device=q_values.device, dtype=q_values.dtype))

        return q_values

    def select_action(self, state: torch.Tensor, mask: Optional[torch.Tensor] = None, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy exploration over feasible actions.
        """
        self.eval()
        with torch.no_grad():
            if mask is not None and not mask.any():
                return 0

            # Exploration
            if torch.rand(1).item() < epsilon:
                if mask is not None:
                    valid_actions = torch.where(mask)[0].cpu().numpy()
                    if len(valid_actions) > 0:
                        import numpy as np
                        return int(np.random.choice(valid_actions))
                import numpy as np
                return int(np.random.randint(0, self.action_dim))

            # Exploitation: ArgMax Q over masked actions
            q_vals = self.forward(state, mask)
            action = int(torch.argmax(q_vals, dim=-1).item())
            return action
