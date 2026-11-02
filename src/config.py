"""
Global Configuration for NeuraSched (Intelligent Task Scheduler).
Production-grade cluster simulation, Google Borg & Alibaba workload profiles,
and Deep RL hyperparameter specifications matching the architectural blueprint.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ClusterNodeConfig:
    node_id: int
    cpu_cores: float
    ram_gb: float
    node_type: str
    processing_speed: float = 1.0

@dataclass
class ClusterConfig:
    num_nodes: int = 6
    # (CPU cores, RAM GB, node_type, processing_speed)
    heterogeneous_specs: List[Tuple[float, float, str, float]] = field(default_factory=lambda: [
        (64.0, 128.0, "general-purpose", 1.0),
        (64.0, 128.0, "general-purpose", 1.0),
        (96.0, 64.0, "compute-heavy", 1.25),   # Faster processing speed
        (96.0, 64.0, "compute-heavy", 1.25),   # Faster processing speed
        (32.0, 256.0, "memory-heavy", 1.0),
        (32.0, 256.0, "memory-heavy", 1.0),
    ])

@dataclass
class WorkloadConfig:
    num_tasks: int = 400
    arrival_rate: float = 5.5         # Tasks per second (realistic Poisson intensity)
    burst_probability: float = 0.20   # Probability of entering a burst wave
    burst_multiplier: float = 2.5     # Multiplier for burst arrival rate
    pareto_shape: float = 1.35        # Heavy-tail shape for duration (Borg/Alibaba trace)
    min_duration: float = 2.0         # Seconds
    max_duration: float = 60.0        # Seconds
    cpu_range: Tuple[float, float] = (1.0, 24.0)
    ram_range: Tuple[float, float] = (2.0, 48.0)
    priority_distribution: Tuple[float, float, float] = (0.3, 0.5, 0.2) # Batch, Standard, Latency-Critical
    sla_slack_factors: Tuple[float, float, float] = (3.0, 1.8, 1.2)   # Slack multiplier per tier

@dataclass
class RLConfig:
    learning_rate: float = 3e-4       # Matches NeuraSched training spec
    gamma: float = 0.99               # Discount factor (future reward vs immediate)
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay_steps: int = 2000
    buffer_capacity: int = 50000
    batch_size: int = 64              # Matches NeuraSched training spec
    target_update_freq: int = 50
    hidden_dim: int = 128             # Matches NeuraSched backbone spec
    num_layers: int = 2
    max_grad_norm: float = 0.5        # Matches NeuraSched gradient clipping spec
    use_prioritized_replay: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    device: str = "cpu"

@dataclass
class RewardConfig:
    # r_t = r_sla + r_pack + r_balance + r_speed - r_defer
    weight_sla: float = 4.5           # r_sla (Heavy penalty for deadline breach, reward for early finish)
    weight_packing: float = 2.0       # r_pack (cosine multi-resource packing function)
    weight_balance: float = 0.5       # r_balance (imbalance penalty)
    weight_speed: float = 2.0         # r_speed (speed efficiency: latency sensitive jobs on faster nodes)
    weight_defer: float = 2.0         # r_defer (penalty for holding in queue to avoid idle delay)
