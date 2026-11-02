# Real cluster trace datasets

This folder adds two public production traces to the project and converts them into task streams the NeuraSched simulator can run.

- Alibaba Cluster Trace v2018
- Google Borg cluster trace 2011

Nothing outside `datasets/` was changed. The existing simulator, schedulers, trained model and benchmark engine are imported and used as they are.

```
datasets/
├── README.md
├── common/
│   └── fit_to_cluster.py          shared steps: scale resources and time, windows, load, deadlines, outputs
├── alibaba_2018/
│   ├── download.py                streams batch_task.csv from Alibaba's public bucket
│   ├── preprocess.py              Alibaba-specific cleaning, then the shared steps
│   ├── raw/batch_task.csv         first 1,000,000 rows (56 MB)
│   └── processed/                 3 benchmark windows, a small scenario, summary.json
├── google_borg_2011/
│   ├── download.py                fetches task_events part files from Google's public bucket
│   ├── preprocess.py              Google-specific cleaning, then the shared steps
│   ├── raw/task_events/           parts 0-2 (about 4 MB each, about 4.7 hours of trace)
│   └── processed/                 3 benchmark windows, a small scenario, summary.json
├── run_benchmark_on_traces.py     runs D3QN and the 5 baselines on the processed windows
└── results/                       benchmark JSON per dataset
```

## How to run

Run everything from the project root with the project's Python 3.9 virtual environment.

```bash
# 1. Download
.venv/bin/python datasets/alibaba_2018/download.py --max-rows 1000000   # 0 = whole table (~14M rows)
.venv/bin/python datasets/google_borg_2011/download.py --num-parts 3    # up to 500 parts

# 2. Preprocess
.venv/bin/python datasets/alibaba_2018/preprocess.py
.venv/bin/python datasets/google_borg_2011/preprocess.py

# 3. Benchmark every scheduler on the real traces
.venv/bin/python datasets/run_benchmark_on_traces.py

# Step through one small real scenario with the existing manual test script
.venv/bin/python scripts/manual_test.py datasets/alibaba_2018/processed/alibaba_scenario.json
```

Both preprocessors accept these options:

| Option | Meaning |
|---|---|
| `--window-size` | tasks per window (default 400) |
| `--num-windows` | number of windows (default 3) |
| `--target-load` | offered load per window (default: same as the synthetic benchmark) |
| `--stride` | how thinly each window is sampled; see step 4 below |
| `--seed` | random seed for picking windows (default 42) |

The Alibaba preprocessor also takes `--resource-mode {rank,quantile}`. The Google preprocessor also takes `--priority-source {scheduling_class,priority}`.

## Output format

Each `processed/*_window_k.csv` has exactly the columns read by `TraceGenerator.load_from_csv`:

`task_id, arrival_time, cpu_req, ram_req, duration, priority, deadline`

Each `*_scenario.json` uses the same format as `test_example_*.json`.

`summary.json` records every number produced along the way:

- rows kept after each cleaning step;
- scale factors;
- target load;
- per-window statistics.

## Preprocessing

### Alibaba Cluster Trace v2018 (`batch_task.csv`)

Source: <https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018>. It covers 8 days of about 4,000 machines.

| Step | What is done | Rows left (1M-row sample) |
|---|---|---|
| Load | 9 columns: task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem | 1,000,000 |
| Status | keep `Terminated` (finished normally) | 983,715 |
| Validity | drop missing values or values <= 0, and rows where end <= start | 929,644 |
| Duplicates | drop exact duplicate rows | 929,644 |

The kept rows are then converted into tasks:

- **Arrival** = `start_time`. The batch table has no separate submit time.
- **Duration** = `end - start`. The raw median is 9 s.
- **Demand:**
  - CPU = `plan_cpu / 100` cores per instance × `instance_num`;
  - RAM = `plan_mem` (% of a machine) × `instance_num`.
- **Priority:** the batch trace has none, so tiers are assigned by duration to match the synthetic mix:
  - shortest 20% (≤ 3 s) → Latency-Critical;
  - next 50% (≤ 30 s) → Standard;
  - longest 30% → Batch.

### Google Borg 2011 (`task_events`)

Source: <https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md>. It covers 29 days of about 12,500 machines. The 2019 Borg trace is published only in BigQuery, so the 2011 trace is used because it can be downloaded directly. It comes from the same Borg scheduler and uses the same task-event model.

| Step | What is done | Rows left (3 parts) |
|---|---|---|
| Load | 13 columns; times in microseconds | 688,564 events |
| Events | keep SUBMIT (0), SCHEDULE (1) and FINISH (4); drop time 0 (tasks already running when the trace started) | 356,286 events |
| Join | per (job ID, task index): first FINISH, the last SCHEDULE before it, the last SUBMIT before that. Tasks that were evicted, failed, killed or never finished are dropped. | 69,693 tasks |
| Validity | CPU and memory requests > 0, finish > schedule | 69,280 tasks |

The kept tasks are then converted:

- **Arrival** = SUBMIT time.
- **Duration** = FINISH − SCHEDULE, converted to seconds. The raw median is 356 s.
- **Demand** = the normalized CPU and memory request on the SUBMIT event.
- **Priority tier** from Google's *scheduling class*, its latency-sensitivity label:
  - 0 → Batch;
  - 1 → Standard;
  - 2–3 → Latency-Critical.

  Google's 0–11 *priority* field is available with `--priority-source priority`. About 92% of tasks have priority 2, so that mapping makes almost everything Standard.

### Shared steps (`common/fit_to_cluster.py`)

The real clusters have thousands of machines, while the simulator has 6 nodes (384 cores, 896 GB). Every trace therefore goes through the same fitting steps:

1. **Resource scaling.**
   - **Google:** linear scaling, so the 95th-percentile request equals the synthetic maximum (24 cores, 48 GB). The relative sizes stay exact. No task needed shrinking.
   - **Alibaba:** *rank* (quantile) mapping. Task footprints range from 1 to over 10,000 instances, so linear scaling left the median task at 0.5 cores. Rank mapping keeps each task's relative size and its CPU/RAM pairing, and takes its size from the synthetic workload's distribution.
2. **Feasibility:** any task that fits no single node is shrunk proportionally to fit. This affected 0 tasks with the defaults.
3. **Time scaling:** one factor for all tasks, so the 95th-percentile duration becomes 60 s, then clipped to 2–120 s. Relative job lengths and the heavy tail are kept.
   - Alibaba: factor 0.158.
   - Google: factor 0.0176.
4. **Windows:** 3 windows of 400 tasks.
   - Each window samples 400 tasks, kept in time order, from a random block of 4,000 consecutive arrivals.
   - Plain consecutive windows were often one big job, because Google submits hundreds of tasks at once.
   - Use `--stride 1` for plain consecutive windows.
5. **Load matching:** the gaps between arrivals are stretched or compressed, keeping their real pattern, so each window has the same dominant-resource offered load as the synthetic benchmark (0.967). The results are then comparable with `benchmark_results.json`.
6. **Deadlines:** deadline = arrival + duration × SLA slack (3.0 Batch, 1.8 Standard, 1.2 Critical), the same rule as the synthetic generator.

## Results

All 6 schedulers ran on the same 3 windows per dataset (1,200 decisions each) using the existing simulator. The trained model has never seen these traces, so these numbers test how well it generalizes.

### Alibaba 2018

| Scheduler | Makespan (s) | Avg turnaround (s) | P99 turnaround (s) | Avg wait (s) | CPU util | RAM util | Stranded | SLA viol. |
|---|---|---|---|---|---|---|---|---|
| **D3QN** | **305.6** | **33.5** | **141.3** | **20.6** | **44.0%** | **37.8%** | 30.5% | **59.3%** |
| FIFO | 328.8 | 39.4 | 168.9 | 25.9 | 41.2% | 35.4% | 27.8% | 63.2% |
| SJF-Backlog | 306.0 | 34.9 | 147.2 | 21.8 | 43.8% | 37.7% | 31.5% | 60.6% |
| EDF | 313.3 | 35.4 | 153.6 | 22.2 | 43.0% | 36.9% | 31.2% | 60.2% |
| Tetris-BestFit | 328.6 | 39.5 | 161.2 | 25.8 | 41.4% | 35.6% | **27.1%** | 63.5% |
| Round-Robin | 321.4 | 37.2 | 157.6 | 24.1 | 42.2% | 36.2% | 29.1% | 61.4% |

### Google Borg 2011

| Scheduler | Makespan (s) | Avg turnaround (s) | P99 turnaround (s) | Avg wait (s) | CPU util | RAM util | Stranded | SLA viol. |
|---|---|---|---|---|---|---|---|---|
| D3QN | 852.9 | 151.9 | 472.6 | 126.4 | 38.2% | 46.8% | 37.4% | 68.5% |
| **FIFO** | **845.2** | **141.2** | 465.7 | **115.7** | **39.7%** | **48.2%** | 37.0% | **63.3%** |
| SJF-Backlog | 856.0 | 154.4 | 472.6 | 128.9 | 37.9% | 46.5% | 36.6% | 68.8% |
| EDF | 852.6 | 151.9 | 471.0 | 126.4 | 38.6% | 47.1% | 37.1% | 68.9% |
| Tetris-BestFit | 849.4 | 150.8 | **463.6** | 125.3 | 38.9% | 47.4% | **34.7%** | 68.8% |
| Round-Robin | 854.5 | 153.8 | 475.1 | 128.3 | 38.4% | 46.9% | 36.9% | 69.1% |

### What the results show

- **Alibaba:** D3QN is best on 7 of 8 quality metrics, ahead of every baseline. Only stranding is lower for Tetris and FIFO.
- **Google:** D3QN does not win. FIFO is best on most metrics, and D3QN is in the middle of the pack.
  - These windows are memory-bound: RAM utilization is above CPU utilization, the opposite of the training data.
  - Their long jobs build up queues of over 100 s.
  - The model was trained only on synthetic, CPU-dominated traces and does not transfer well here.
  - Retraining or fine-tuning on preprocessed Google windows would be the next step.

## Limitations

- **Samples, not full traces.** Only the first 1M of about 14M Alibaba rows and 3 of 500 Google parts are used. The download scripts can fetch more.
- **Time is compressed.** Resource sizes are fitted to a 6-node cluster (linear scaling for Google, rank mapping for Alibaba). The patterns in arrivals, durations and priorities are real, but absolute sizes and times are not.
- **Alibaba arrival times and priorities are approximations.** Arrival uses start time, because the trace has no submit time, and priority tiers are derived from duration.
- **Google's mix is mostly batch work** (about 90% scheduling class 0). That is how the real trace looks, but it means few Latency-Critical tasks.
- **The SLA deadlines are synthetic.** Neither trace publishes deadlines, so the project's slack rule is used.
