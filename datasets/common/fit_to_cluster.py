"""
Shared preprocessing for real cluster traces (Alibaba 2018, Google Borg 2011).

Every dataset-specific preprocessor first produces a "clean" DataFrame with
these columns, in the trace's own units converted to seconds:

    arrival_time (s), duration (s), cpu_raw, ram_raw, priority (0/1/2)

The functions here then fit that clean trace to the NeuraSched simulator
(6 nodes, 384 cores, 896 GB) and write it in the exact schema read by
src.workload.trace_generator.TraceGenerator.load_from_csv:

    task_id, arrival_time, cpu_req, ram_req, duration, priority, deadline

Pipeline (each step is recorded in the summary report):
    1. Resource scaling   map the trace's CPU / RAM demand onto the simulator scale
    2. Feasibility        shrink any task that no single node could ever host
    3. Duration scaling   compress time so durations sit in the simulator's range
    4. Windowing          cut N-task windows (episodes) from the trace
    5. Load matching      rescale inter-arrival gaps to a target offered load
    6. Deadlines          deadline = arrival + duration * SLA slack(priority)
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ClusterConfig, WorkloadConfig
from src.workload.trace_generator import TraceGenerator

OUTPUT_COLUMNS = ["task_id", "arrival_time", "cpu_req", "ram_req", "duration", "priority", "deadline"]
PRIORITY_NAMES = {0: "Batch", 1: "Standard", 2: "Latency-Critical"}


# =========================================================================
# 1. Resource scaling
# =========================================================================
def scale_resources(
    df: pd.DataFrame,
    mode: str = "quantile",
    cpu_target: Optional[float] = None,
    ram_target: Optional[float] = None,
    quantile: float = 0.95,
    cpu_unit: float = 1.0,
    ram_unit: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    mode="quantile": scale so the trace's `quantile` CPU / RAM demand equals the
                     largest request of the synthetic workload (24 cores, 48 GB by
                     default). Keeps the shape of each distribution.
    mode="rank":     quantile matching. Each task keeps its percentile rank in the
                     trace (so bigger tasks stay bigger and CPU/RAM stay paired) and
                     takes the value at that percentile of the synthetic workload's
                     distribution. Used when a trace spans orders of magnitude
                     (Alibaba: 1 to 10,000+ instances per task), where any linear
                     scaling crushes almost every task to the minimum size.
    mode="absolute": multiply raw values by cpu_unit / ram_unit (e.g. cores per
                     normalized unit of the reference machine).
    """
    wl = WorkloadConfig()
    cpu_target = cpu_target if cpu_target is not None else wl.cpu_range[1]
    ram_target = ram_target if ram_target is not None else wl.ram_range[1]

    out = df.copy()
    if mode == "quantile":
        cpu_q = float(out["cpu_raw"].quantile(quantile))
        ram_q = float(out["ram_raw"].quantile(quantile))
        cpu_factor = cpu_target / cpu_q if cpu_q > 0 else 1.0
        ram_factor = ram_target / ram_q if ram_q > 0 else 1.0
    elif mode == "rank":
        ref = TraceGenerator(wl).generate_trace(num_tasks=20000, seed=0)
        ref_cpu = np.array([t.cpu_req for t in ref])
        ref_ram = np.array([t.ram_req for t in ref])
        out["cpu_req"] = np.quantile(ref_cpu, out["cpu_raw"].rank(pct=True, method="average").clip(0, 1))
        out["ram_req"] = np.quantile(ref_ram, out["ram_raw"].rank(pct=True, method="average").clip(0, 1))
        out["cpu_req"] = np.maximum(0.5, np.round(out["cpu_req"] * 2.0) / 2.0)
        out["ram_req"] = np.maximum(1.0, np.round(out["ram_req"]))
        return out, {"mode": mode, "reference": "synthetic WorkloadConfig, 20000 tasks, seed 0"}
    elif mode == "absolute":
        cpu_factor, ram_factor = cpu_unit, ram_unit
    else:
        raise ValueError(f"Unknown resource mode: {mode}")

    # Same rounding as the synthetic generator: 0.5 cores, 1 GB, with a floor
    out["cpu_req"] = np.maximum(0.5, np.round(out["cpu_raw"] * cpu_factor * 2.0) / 2.0)
    out["ram_req"] = np.maximum(1.0, np.round(out["ram_raw"] * ram_factor))
    return out, {"mode": mode, "cpu_factor": cpu_factor, "ram_factor": ram_factor, "quantile": quantile}


# =========================================================================
# 2. Feasibility
# =========================================================================
def make_feasible(df: pd.DataFrame, cluster_config: Optional[ClusterConfig] = None) -> Tuple[pd.DataFrame, int]:
    """
    Shrink (proportionally, keeping the CPU:RAM ratio) any task that does not fit
    on any single node, so it fits the node that needs the least shrinking.
    Returns the adjusted frame and how many tasks were changed.
    """
    specs = (cluster_config or ClusterConfig()).heterogeneous_specs
    caps = np.array([(s[0], s[1]) for s in specs], dtype=float)

    cpu = df["cpu_req"].to_numpy(dtype=float)
    ram = df["ram_req"].to_numpy(dtype=float)
    fits = np.zeros(len(df), dtype=bool)
    for c, r in caps:
        fits |= (cpu <= c) & (ram <= r)

    adjusted = 0
    for i in np.where(~fits)[0]:
        # largest shrink factor that fits each node; pick the node needing least shrinking
        best = max(min(c / cpu[i], r / ram[i]) for c, r in caps)
        cpu[i] = max(0.5, np.floor(cpu[i] * best * 2.0) / 2.0)
        ram[i] = max(1.0, np.floor(ram[i] * best))
        adjusted += 1

    out = df.copy()
    out["cpu_req"] = cpu
    out["ram_req"] = ram
    return out, adjusted


# =========================================================================
# 3. Duration scaling
# =========================================================================
def scale_durations(
    df: pd.DataFrame,
    target_p95: Optional[float] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Real traces run from seconds to days. Compress time with one factor so the
    95th-percentile duration equals the simulator's max synthetic duration
    (60 s), then clip to [min, 2 * max]. One factor for all tasks keeps the
    relative lengths (and heavy tail) of the real jobs.
    """
    wl = WorkloadConfig()
    target_p95 = target_p95 if target_p95 is not None else wl.max_duration
    min_duration = min_duration if min_duration is not None else wl.min_duration
    max_duration = max_duration if max_duration is not None else 2.0 * wl.max_duration

    p95 = float(df["duration"].quantile(0.95))
    factor = target_p95 / p95 if p95 > 0 else 1.0
    out = df.copy()
    out["duration"] = np.clip(out["duration"] * factor, min_duration, max_duration).round(2)
    out["arrival_time"] = out["arrival_time"] * factor
    return out, {"time_factor": factor, "raw_p95_duration_s": p95}


# =========================================================================
# 4 + 5. Windowing and load matching
# =========================================================================
def offered_load(df: pd.DataFrame, cluster_config: Optional[ClusterConfig] = None) -> float:
    """Dominant-resource offered load: demand-seconds / (span * capacity)."""
    specs = (cluster_config or ClusterConfig()).heterogeneous_specs
    total_cpu = sum(s[0] for s in specs)
    total_ram = sum(s[1] for s in specs)
    span = max(1e-6, float(df["arrival_time"].max() - df["arrival_time"].min()))
    cpu_load = float((df["cpu_req"] * df["duration"]).sum()) / (span * total_cpu)
    ram_load = float((df["ram_req"] * df["duration"]).sum()) / (span * total_ram)
    return max(cpu_load, ram_load)


def synthetic_reference_load(num_tasks: int = 400, seeds: Tuple[int, ...] = (1337, 1474, 1611)) -> float:
    """Offered load of the project's own synthetic benchmark traces (same seeds as the benchmark)."""
    gen = TraceGenerator(WorkloadConfig())
    loads = []
    for seed in seeds:
        tasks = gen.generate_trace(num_tasks=num_tasks, seed=seed)
        frame = pd.DataFrame(
            [{"arrival_time": t.arrival_time, "cpu_req": t.cpu_req, "ram_req": t.ram_req, "duration": t.duration} for t in tasks]
        )
        loads.append(offered_load(frame))
    return float(np.mean(loads))


def cut_windows(df: pd.DataFrame, window_size: int, num_windows: int, seed: int = 42, stride: int = 10) -> List[pd.DataFrame]:
    """
    Cut `num_windows` non-overlapping windows. Each window starts from a random block
    of `window_size * stride` consecutive arrivals and keeps `window_size` of them,
    sampled uniformly and kept in time order.

    Real traces submit big jobs as hundreds of tasks at once, so a plain run of
    consecutive arrivals is often a single job. Thinning a longer block keeps the
    real arrival pattern (bursts, gaps) while mixing many jobs. stride=1 gives
    plain consecutive windows.
    """
    df = df.sort_values("arrival_time", kind="mergesort").reset_index(drop=True)
    block = window_size * max(1, stride)
    n_slots = len(df) // block
    if n_slots == 0:
        if stride > 1:
            return cut_windows(df, window_size, num_windows, seed=seed, stride=max(1, len(df) // window_size))
        raise ValueError(f"Only {len(df)} clean tasks; need at least {window_size} for one window.")
    rng = np.random.default_rng(seed)
    slots = sorted(rng.choice(n_slots, size=min(num_windows, n_slots), replace=False))
    windows = []
    for s in slots:
        rows = np.sort(rng.choice(block, size=window_size, replace=False)) + s * block
        windows.append(df.iloc[rows].copy())
    return windows


def match_load(window: pd.DataFrame, target_load: float) -> Tuple[pd.DataFrame, float]:
    """
    Rescale inter-arrival gaps (keeping their real pattern, including bursts) so the
    window's offered load equals target_load, and shift it to start at t=0.
    """
    out = window.sort_values("arrival_time", kind="mergesort").copy()
    out["arrival_time"] = out["arrival_time"] - out["arrival_time"].min()
    current = offered_load(out)
    factor = current / target_load if target_load > 0 and current > 0 else 1.0
    out["arrival_time"] = (out["arrival_time"] * factor).round(3)
    return out, factor


# =========================================================================
# 6. Deadlines + output
# =========================================================================
def add_deadlines(df: pd.DataFrame) -> pd.DataFrame:
    slack = WorkloadConfig().sla_slack_factors
    out = df.copy()
    out["deadline"] = (out["arrival_time"] + out["duration"] * out["priority"].map(lambda p: slack[int(p)])).round(2)
    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["task_id"] = np.arange(len(out))
    out["priority"] = out["priority"].astype(int)
    return out[OUTPUT_COLUMNS]


def describe(df: pd.DataFrame) -> Dict[str, object]:
    pri = df["priority"].value_counts(normalize=True).to_dict()
    return {
        "tasks": int(len(df)),
        "span_s": round(float(df["arrival_time"].max() - df["arrival_time"].min()), 2),
        "offered_load": round(offered_load(df), 3),
        "cpu_req": {k: round(float(v), 2) for k, v in df["cpu_req"].describe(percentiles=[0.5, 0.95]).items() if k != "count"},
        "ram_req": {k: round(float(v), 2) for k, v in df["ram_req"].describe(percentiles=[0.5, 0.95]).items() if k != "count"},
        "duration": {k: round(float(v), 2) for k, v in df["duration"].describe(percentiles=[0.5, 0.95]).items() if k != "count"},
        "priority_share": {PRIORITY_NAMES[int(k)]: round(float(v), 3) for k, v in sorted(pri.items())},
    }


def write_outputs(
    windows: List[pd.DataFrame],
    out_dir: str,
    prefix: str,
    summary: Dict[str, object],
    scenario_tasks: int = 8,
    description: str = "",
) -> List[str]:
    """
    Writes:
      processed/<prefix>_window_<k>.csv     benchmark-ready traces (TraceGenerator.load_from_csv format)
      processed/<prefix>_scenario.json      small scenario in test_example_*.json format (manual_test.py / app)
      processed/summary.json                every preprocessing statistic
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, w in enumerate(windows, 1):
        path = os.path.join(out_dir, f"{prefix}_window_{k}.csv")
        w.to_csv(path, index=False)
        paths.append(path)

    first = windows[0].head(scenario_tasks)
    first = first.assign(arrival_time=(first["arrival_time"] - first["arrival_time"].min()).round(2))
    first = add_deadlines(first.drop(columns=["deadline"]))
    scenario = {
        "description": description or f"First {scenario_tasks} tasks of {prefix} window 1 (real trace, preprocessed).",
        "tasks": [
            {
                "task_id": int(r.task_id),
                "arrival_time": float(r.arrival_time),
                "cpu_req": float(r.cpu_req),
                "ram_req": float(r.ram_req),
                "duration": float(r.duration),
                "priority": int(r.priority),
                "deadline": float(r.deadline),
            }
            for r in first.itertuples()
        ],
    }
    with open(os.path.join(out_dir, f"{prefix}_scenario.json"), "w") as f:
        json.dump(scenario, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return paths


def fit_clean_trace(
    clean: pd.DataFrame,
    out_dir: str,
    prefix: str,
    cleaning_report: Dict[str, object],
    resource_mode: str = "quantile",
    cpu_unit: float = 1.0,
    ram_unit: float = 1.0,
    window_size: int = 400,
    num_windows: int = 3,
    target_load: Optional[float] = None,
    seed: int = 42,
    stride: int = 10,
    description: str = "",
) -> Dict[str, object]:
    """Run steps 1-6 on a clean trace and write every output. Returns the summary."""
    scaled, res_info = scale_resources(clean, mode=resource_mode, cpu_unit=cpu_unit, ram_unit=ram_unit)
    feasible, n_adjusted = make_feasible(scaled)
    timed, time_info = scale_durations(feasible)

    if target_load is None:
        target_load = synthetic_reference_load(num_tasks=window_size)

    windows = []
    window_info = []
    for w in cut_windows(timed, window_size, num_windows, seed=seed, stride=stride):
        raw_load = offered_load(w.assign(arrival_time=w["arrival_time"] - w["arrival_time"].min()))
        matched, gap_factor = match_load(w, target_load)
        final = finalize(add_deadlines(matched))
        windows.append(final)
        window_info.append({"raw_offered_load": round(raw_load, 3), "gap_factor": round(gap_factor, 5), **describe(final)})

    summary = {
        "dataset": prefix,
        "cleaning": cleaning_report,
        "resource_scaling": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res_info.items()},
        "tasks_shrunk_to_fit_a_node": n_adjusted,
        "duration_scaling": {k: round(v, 6) for k, v in time_info.items()},
        "target_offered_load": round(target_load, 3),
        "window_sampling": f"{window_size} tasks sampled in time order from blocks of {window_size * stride} consecutive arrivals",
        "full_clean_trace_after_scaling": describe(timed.assign(deadline=0.0)),
        "windows": window_info,
    }
    summary["output_files"] = write_outputs(windows, out_dir, prefix, summary, description=description)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nCleaning")
    for k, v in summary["cleaning"].items():
        print(f"  {k:<40} {v}")
    print(f"\nResource scaling      {summary['resource_scaling']}")
    print(f"Tasks shrunk to fit   {summary['tasks_shrunk_to_fit_a_node']}")
    print(f"Duration scaling      {summary['duration_scaling']}")
    print(f"Target offered load   {summary['target_offered_load']}")
    print("\nWindows")
    for i, w in enumerate(summary["windows"], 1):
        print(
            f"  window {i}: {w['tasks']} tasks, span {w['span_s']}s, load {w['offered_load']}, "
            f"median cpu {w['cpu_req']['50%']} / ram {w['ram_req']['50%']} GB / dur {w['duration']['50%']}s, "
            f"priorities {w['priority_share']}"
        )
    print("\nWrote:")
    for p in summary["output_files"]:
        print(f"  {p}")
