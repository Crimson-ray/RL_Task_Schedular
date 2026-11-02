#!/usr/bin/env python3
"""
CLI Script to execute hyperparameter tuning for NeuraSched.
Matches specifications from architectural blueprints (Page 1 & Page 2).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tune.tuner import HyperparameterTuner
from src.config import ClusterConfig, WorkloadConfig

def main():
    print("Starting production-grade hyperparameter tuning for NeuraSched...")
    tuner = HyperparameterTuner()

    # Search space optimizing for: 1. Zero SLA violations, 2. Min Makespan, 3. Packing & Speed efficiency
    search_space = [
        {
            "trial_name": "NeuraSched_Optimal_D3QN_128",
            "learning_rate": 2.5e-4,
            "hidden_dim": 128,
            "num_layers": 2,
            "weight_packing": 2.5,
            "weight_sla": 5.5,
            "weight_balance": 0.5,
            "weight_speed": 3.0
        },
        {
            "trial_name": "NeuraSched_HighCapacity_256",
            "learning_rate": 2.0e-4,
            "hidden_dim": 256,
            "num_layers": 2,
            "weight_packing": 2.0,
            "weight_sla": 6.0,
            "weight_balance": 0.4,
            "weight_speed": 3.5
        },
        {
            "trial_name": "NeuraSched_Speed_Prioritized_128",
            "learning_rate": 3.0e-4,
            "hidden_dim": 128,
            "num_layers": 2,
            "weight_packing": 3.0,
            "weight_sla": 5.5,
            "weight_balance": 0.6,
            "weight_speed": 4.0
        },
        {
            "trial_name": "NeuraSched_DeepTrunk_3Layers",
            "learning_rate": 1.5e-4,
            "hidden_dim": 128,
            "num_layers": 3,
            "weight_packing": 2.5,
            "weight_sla": 5.0,
            "weight_balance": 0.5,
            "weight_speed": 3.0
        }
    ]

    summary = tuner.run_tuning(
        search_space=search_space,
        episodes_per_trial=25,
        tasks_per_episode=200,
        eval_seed=8888
    )

    print(f"Hyperparameter tuning completed successfully!")
    print(f"Winner: {summary['best_trial_name']}")
    print(f"Best fitness score: {summary['best_fitness_score']}")
    print(f"Model saved to: {summary['best_model_path']}")

if __name__ == "__main__":
    main()
