#!/usr/bin/env python3
"""
Preprocess Alibaba Cluster Trace v2018 (batch_task.csv) into NeuraSched traces.

Cleaning (dataset specific):
  - keep status == "Terminated" (finished normally)
  - drop rows with missing / non-positive start, end, plan_cpu, plan_mem, instance_num
  - drop rows with end_time <= start_time and exact duplicates
  - arrival_time = start_time (batch_task has no separate submit time)
  - duration     = end_time - start_time (seconds)
  - CPU demand   = plan_cpu / 100 cores per instance  x  instance_num
  - RAM demand   = plan_mem (% of a machine)          x  instance_num
    (a task's instances run together, so the task's total footprint is used)
  - priority: the batch trace has no priority field, so tiers are assigned by
    duration, matching the synthetic mix: shortest 20% -> Latency-Critical,
    next 50% -> Standard, longest 30% -> Batch

Resource scaling defaults to "rank" (quantile matching): task footprints range from
1 to 10,000+ instances, so a linear scale would make almost every task 0.5 cores.
Rank mapping keeps each task's relative size and CPU/RAM pairing. Use
--resource-mode quantile for plain linear scaling instead.

Then the shared pipeline (datasets/common/fit_to_cluster.py) scales resources and
time to the simulator, cuts windows, matches the offered load and adds deadlines.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from datasets.common.fit_to_cluster import fit_clean_trace, print_summary

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "raw", "batch_task.csv")
OUT_DIR = os.path.join(HERE, "processed")


def clean_alibaba(raw_path: str = RAW_PATH):
    df = pd.read_csv(raw_path)
    report = {"raw_rows": int(len(df))}

    df = df[df["status"] == "Terminated"]
    report["after_status_terminated"] = int(len(df))

    numeric = ["instance_num", "start_time", "end_time", "plan_cpu", "plan_mem"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric)
    df = df[(df["start_time"] > 0) & (df["end_time"] > df["start_time"])
            & (df["plan_cpu"] > 0) & (df["plan_mem"] > 0) & (df["instance_num"] >= 1)]
    report["after_valid_times_and_resources"] = int(len(df))

    df = df.drop_duplicates()
    report["after_dedup"] = int(len(df))

    clean = pd.DataFrame({
        "arrival_time": df["start_time"].astype(float),
        "duration": (df["end_time"] - df["start_time"]).astype(float),
        "cpu_raw": df["plan_cpu"] / 100.0 * df["instance_num"],
        "ram_raw": df["plan_mem"] * df["instance_num"],
    }).reset_index(drop=True)

    q20, q70 = clean["duration"].quantile([0.2, 0.7])
    clean["priority"] = np.where(clean["duration"] <= q20, 2, np.where(clean["duration"] <= q70, 1, 0))
    report["priority_rule"] = f"duration <= {q20:.0f}s -> Critical, <= {q70:.0f}s -> Standard, else Batch"
    report["clean_tasks"] = int(len(clean))
    report["raw_duration_s_median"] = round(float(clean["duration"].median()), 1)
    report["raw_instances_median"] = float(df["instance_num"].median())
    return clean, report


def main():
    parser = argparse.ArgumentParser(description="Preprocess Alibaba v2018 batch tasks for NeuraSched")
    parser.add_argument("--raw", default=RAW_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--target-load", type=float, default=None,
                        help="Offered load per window (default: same as the synthetic benchmark).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resource-mode", choices=["rank", "quantile"], default="rank")
    parser.add_argument("--stride", type=int, default=10,
                        help="Each window samples window-size tasks from window-size*stride consecutive arrivals (1 = consecutive).")
    args = parser.parse_args()

    if not os.path.isfile(args.raw):
        sys.exit(f"Raw file not found: {args.raw}\nRun: python datasets/alibaba_2018/download.py")

    clean, report = clean_alibaba(args.raw)
    summary = fit_clean_trace(
        clean, args.out, "alibaba", report, resource_mode=args.resource_mode,
        window_size=args.window_size, num_windows=args.num_windows,
        target_load=args.target_load, seed=args.seed, stride=args.stride,
        description="Alibaba Cluster Trace v2018 batch tasks: first 8 tasks of window 1 after preprocessing.",
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
