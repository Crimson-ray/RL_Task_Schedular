#!/usr/bin/env python3
"""
Download part files of the Google cluster-usage trace (clusterdata-2011-2).

Source:  https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md
Bucket:  https://storage.googleapis.com/clusterdata-2011-2/
Table:   task_events, 500 gzip parts of ~4 MB each, ~1.5 hours of trace per part.

The 2019 Borg trace is only published in BigQuery, so the 2011 trace (same
Borg scheduler, same task-event model) is used here because it can be
downloaded directly.

task_events columns (no header):
    time (us), missing info, job ID, task index, machine ID, event type, user,
    scheduling class, priority, CPU request, memory request, disk space request,
    different machines restriction
"""
import argparse
import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/clusterdata-2011-2"
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "task_events")


def download_parts(num_parts: int = 3, start: int = 0, out_dir: str = RAW_DIR):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(start, start + num_parts):
        name = f"part-{i:05d}-of-00500.csv.gz"
        path = os.path.join(out_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"  exists  {name}")
        else:
            print(f"  fetch   {name}")
            urllib.request.urlretrieve(f"{BASE_URL}/task_events/{name}", path)
        paths.append(path)
    schema = os.path.join(os.path.dirname(out_dir), "schema.csv")
    if not os.path.isfile(schema):
        urllib.request.urlretrieve(f"{BASE_URL}/schema.csv", schema)
    print(f"Saved {len(paths)} part file(s) to {out_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Download Google clusterdata-2011-2 task_events parts")
    parser.add_argument("--num-parts", type=int, default=3, help="How many consecutive parts to fetch (~4 MB each).")
    parser.add_argument("--start", type=int, default=0, help="Index of the first part (0-499).")
    args = parser.parse_args()
    download_parts(args.num_parts, args.start)


if __name__ == "__main__":
    main()
