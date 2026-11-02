#!/usr/bin/env python3
"""
Download the Alibaba Cluster Trace v2018 batch task table.

Source:  https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018
File:    batch_task.tar.gz (130 MB compressed, 800 MB / ~14M rows uncompressed)

By default the archive is streamed and only the first --max-rows rows are kept,
so nothing large is stored. Use --max-rows 0 to keep the whole table.

Columns of batch_task.csv (no header in the original file):
    task_name, instance_num, job_name, task_type, status,
    start_time, end_time, plan_cpu, plan_mem
"""
import argparse
import csv
import io
import os
import tarfile
import urllib.request

BASE_URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
COLUMNS = ["task_name", "instance_num", "job_name", "task_type", "status",
           "start_time", "end_time", "plan_cpu", "plan_mem"]
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def download_batch_task(max_rows: int = 1_000_000, out_path: str = os.path.join(RAW_DIR, "batch_task.csv")) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    url = f"{BASE_URL}/batch_task.tar.gz"
    print(f"Streaming {url}")
    written = 0
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.name.endswith("batch_task.csv"):
                    continue
                source = archive.extractfile(member)
                with open(out_path, "w", newline="") as out:
                    writer = csv.writer(out)
                    writer.writerow(COLUMNS)
                    # Streamed tar members are not seekable on Python 3.9, so split lines by hand
                    pending = b""
                    done = False
                    while not done:
                        chunk = source.read(1 << 20)
                        if not chunk:
                            break
                        lines = (pending + chunk).split(b"\n")
                        pending = lines.pop()
                        for row in csv.reader(io.StringIO(b"\n".join(lines).decode("utf-8"))):
                            writer.writerow(row)
                            written += 1
                            if written % 250_000 == 0:
                                print(f"  {written:,} rows")
                            if max_rows and written >= max_rows:
                                done = True
                                break
                    if pending and not done:
                        writer.writerow(next(csv.reader([pending.decode("utf-8")])))
                        written += 1
                break
    print(f"Saved {written:,} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Alibaba v2018 batch_task.csv")
    parser.add_argument("--max-rows", type=int, default=1_000_000,
                        help="Rows to keep from the start of the table (0 = all ~14M rows).")
    args = parser.parse_args()
    download_batch_task(max_rows=args.max_rows)


if __name__ == "__main__":
    main()
