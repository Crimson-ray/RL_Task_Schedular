#!/usr/bin/env python3
"""
Preprocess Google Borg cluster trace 2011 (task_events) into NeuraSched traces.

Cleaning (dataset specific):
  - keep only SUBMIT (0), SCHEDULE (1) and FINISH (4) events
  - drop events at time 0 (tasks already running before the trace started)
  - per task (job ID, task index): first FINISH, the last SCHEDULE before it,
    and the last SUBMIT before that schedule. Tasks that were evicted, failed,
    killed or never finished inside the downloaded parts are dropped.
  - arrival_time = SUBMIT time, duration = FINISH - SCHEDULE (microseconds -> seconds)
  - CPU / RAM demand = the request on the SUBMIT event (normalized to the largest
    Google machine); tasks without a positive request are dropped
  - priority tier (--priority-source):
        scheduling_class (default): Google's latency-sensitivity label
            0 -> Batch, 1 -> Standard, 2-3 -> Latency-Critical
        priority: Google's 0-11 priority
            0-1 (free) -> Batch, 2-8 (middle) -> Standard, 9-11 (production) -> Latency-Critical
        In the 2011 trace ~90% of tasks have priority 2, so the priority mapping
        makes almost every task Standard; scheduling class gives a real mix.

Then the shared pipeline (datasets/common/fit_to_cluster.py) scales resources and
time to the simulator, cuts windows, matches the offered load and adds deadlines.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from datasets.common.fit_to_cluster import fit_clean_trace, print_summary

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw", "task_events")
OUT_DIR = os.path.join(HERE, "processed")

COLUMNS = ["time", "missing_info", "job_id", "task_index", "machine_id", "event_type", "user",
           "scheduling_class", "priority", "cpu_request", "memory_request", "disk_request",
           "different_machines"]
SUBMIT, SCHEDULE, FINISH = 0, 1, 4


def map_priority(p: pd.Series) -> np.ndarray:
    return np.where(p >= 9, 2, np.where(p >= 2, 1, 0))


def map_scheduling_class(c: pd.Series) -> np.ndarray:
    return np.where(c >= 2, 2, np.where(c == 1, 1, 0))


def clean_google(raw_dir: str = RAW_DIR, priority_source: str = "scheduling_class"):
    parts = sorted(glob.glob(os.path.join(raw_dir, "part-*.csv.gz")))
    if not parts:
        raise FileNotFoundError(raw_dir)
    usecols = ["time", "job_id", "task_index", "event_type", "scheduling_class", "priority",
               "cpu_request", "memory_request"]
    df = pd.concat(
        [pd.read_csv(p, header=None, names=COLUMNS, usecols=usecols) for p in parts],
        ignore_index=True,
    )
    report = {"part_files": len(parts), "raw_events": int(len(df))}

    df = df[df["event_type"].isin([SUBMIT, SCHEDULE, FINISH]) & (df["time"] > 0)]
    report["submit_schedule_finish_events_after_time_0"] = int(len(df))
    df = df.sort_values("time", kind="mergesort")
    key = ["job_id", "task_index"]

    finish = df[df["event_type"] == FINISH].groupby(key, as_index=False)["time"].first()
    finish = finish.rename(columns={"time": "finish"})

    sched = df[df["event_type"] == SCHEDULE].merge(finish, on=key)
    sched = sched[sched["time"] < sched["finish"]].groupby(key, as_index=False).agg(
        schedule=("time", "last"), finish=("finish", "first"))

    submit = df[df["event_type"] == SUBMIT].merge(sched, on=key)
    submit = submit[submit["time"] <= submit["schedule"]].groupby(key, as_index=False).last()
    report["tasks_with_submit_schedule_finish"] = int(len(submit))

    submit = submit.dropna(subset=["cpu_request", "memory_request", "priority", "scheduling_class"])
    submit = submit[(submit["cpu_request"] > 0) & (submit["memory_request"] > 0)
                    & (submit["finish"] > submit["schedule"])]
    report["after_valid_requests"] = int(len(submit))

    clean = pd.DataFrame({
        "arrival_time": submit["time"] / 1e6,
        "duration": (submit["finish"] - submit["schedule"]) / 1e6,
        "cpu_raw": submit["cpu_request"].astype(float),
        "ram_raw": submit["memory_request"].astype(float),
        "priority": (map_scheduling_class(submit["scheduling_class"]) if priority_source == "scheduling_class"
                     else map_priority(submit["priority"])),
    }).reset_index(drop=True)

    report["priority_source"] = priority_source
    report["google_priority_counts"] = {int(k): int(v) for k, v in submit["priority"].value_counts().sort_index().items()}
    report["google_scheduling_class_counts"] = {
        int(k): int(v) for k, v in submit["scheduling_class"].value_counts().sort_index().items()}
    report["clean_tasks"] = int(len(clean))
    report["raw_duration_s_median"] = round(float(clean["duration"].median()), 1)
    report["raw_wait_s_median"] = round(float(((submit["schedule"] - submit["time"]) / 1e6).median()), 2)
    return clean, report


def main():
    parser = argparse.ArgumentParser(description="Preprocess Google Borg 2011 task_events for NeuraSched")
    parser.add_argument("--raw", default=RAW_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--target-load", type=float, default=None,
                        help="Offered load per window (default: same as the synthetic benchmark).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=10,
                        help="Each window samples window-size tasks from window-size*stride consecutive arrivals (1 = consecutive).")
    parser.add_argument("--priority-source", choices=["scheduling_class", "priority"], default="scheduling_class")
    args = parser.parse_args()

    try:
        clean, report = clean_google(args.raw, args.priority_source)
    except FileNotFoundError:
        sys.exit(f"No part files in {args.raw}\nRun: python datasets/google_borg_2011/download.py")

    summary = fit_clean_trace(
        clean, args.out, "google_borg", report,
        window_size=args.window_size, num_windows=args.num_windows,
        target_load=args.target_load, seed=args.seed, stride=args.stride,
        description="Google Borg cluster trace 2011: first 8 tasks of window 1 after preprocessing.",
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
