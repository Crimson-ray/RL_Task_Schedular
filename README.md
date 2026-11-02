# Intelligent Task Scheduler (Production-Grade)

An autonomous, deep reinforcement learning cluster task scheduler designed for heterogeneous cloud and edge computing environments (Google Borg / Alibaba inspired workloads).

Powered by **Dueling Double Deep Q-Networks (D3QN) with Dynamic Action Masking, Multi-Resource Anti-Stranding Packing, and Prioritized Experience Replay (PER)**.

---

## Key Performance Results

| Metric | Intelligent D3QN | FIFO | SJF-Backlog | EDF | Tetris BestFit | Round-Robin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Makespan (s)** | **128.97** | 129.67 | 130.46 | 129.94 | 129.96 | 129.36 |
| **Avg Turnaround Time (s)** | **6.97** | 7.39 | 7.57 | 7.29 | 7.74 | 7.44 |
| **P99 Turnaround Time (s)** | **51.08** | 51.96 | 51.95 | 51.31 | 51.15 | 51.04 |
| **Cluster CPU Util (%)** | **45.12%** | 44.90% | 44.63% | 44.67% | 44.70% | 44.95% |
| **Cluster RAM Util (%)** | **37.20%** | 36.99% | 36.77% | 36.85% | 36.84% | 37.07% |
| **Stranded Resource Ratio (%)** | **16.58%** | 16.97% | 19.07% | 19.90% | 19.98% | 19.30% |
| **SLA Violation Rate (%)** | **13.58%** | 18.92% | 21.42% | 19.00% | 23.58% | 20.08% |
| **Critical SLA Violation (%)** | **37.31%** | 42.73% | 45.02% | 45.67% | 52.34% | 46.06% |
| **Decision Latency (μs)** | **53.7 μs** | 0.65 μs | 2.61 μs | 2.07 μs | 14.17 μs | 1.05 μs |

*Evaluated over 400 bursty heterogeneous tasks across 3 seed-locked runs (1,200 total scheduling decisions).*

---

## Project Structure

```
DL_PROJECT/
├── architecture_spec.md            # Complete first-principles architecture specification
├── benchmark_comparison.png        # 4-panel visual benchmarking plot
├── benchmark_results.json          # Complete benchmark metric output
├── checkpoints/
│   ├── best_d3qn_scheduler.pt      # Optimized model weights binary
│   └── tuning_summary.json         # Complete hyperparameter tuning search history
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── src/
│   ├── config.py                   # Cluster, Workload, RL, and Reward configurations
│   ├── environment/
│   │   ├── node.py                 # Heterogeneous compute node dynamic tracking
│   │   ├── cluster.py              # Multi-node coordinator, action masks, state vectors
│   │   └── simulator.py            # Discrete-event cluster simulator (Gym-like step + batch)
│   ├── workload/
│   │   ├── task.py                 # Task lifecycle, resource requirements, SLAs
│   │   └── trace_generator.py      # Borg/Alibaba heavy-tailed trace generator & CSV loader
│   ├── models/
│   │   ├── dueling_dqn.py          # Dueling DQN with dynamic action masking
│   │   └── replay_buffer.py        # Prioritized Experience Replay (Sum-Tree + IS weights)
│   ├── scheduler/
│   │   ├── base.py                 # Abstract Base Scheduler interface
│   │   ├── intelligent_scheduler.py# High-performance sub-millisecond inference scheduler
│   │   ├── heuristics.py           # FIFO, SJF, EDF, Tetris Best-Fit, Round-Robin
│   │   └── trainer.py              # Double Q-learning training engine
│   ├── tune/
│   │   └── tuner.py                # Hyperparameter exploration and fitness evaluator
│   └── benchmarks/
│       └── benchmark_engine.py     # Statistical benchmark suite & visualization generator
├── scripts/
│   ├── tune.py                     # CLI: Hyperparameter optimization loop
│   └── run_benchmark.py            # CLI: Production benchmark execution
└── tests/
    ├── test_environment.py         # Unit tests: Cluster, nodes, allocations
    ├── test_workload.py            # Unit tests: Trace generator, Pareto distributions
    ├── test_models.py              # Unit tests: Dueling DQN, action masking, PER buffer
    └── test_scheduler.py           # Integration tests: End-to-end simulations
```

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/
```

### 3. Run Hyperparameter Tuning
```bash
python scripts/tune.py
```

### 4. Run Benchmark Suite
```bash
python scripts/run_benchmark.py
```
This generates `benchmark_results.json` and `benchmark_comparison.png` in the root workspace.
