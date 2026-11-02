#!/usr/bin/env python3
"""
Manual Testing Script for NeuraSched (Intelligent Task Scheduler).
Loads custom task streams (3-8 tasks) from JSON files (test_example_i.json),
dynamically simulates task arrivals, node state updates, action masking,
neural Q-network inference, and prints a step-by-step trace and final summary.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
from tabulate import tabulate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.cluster import Cluster
from src.workload.task import Task
from src.scheduler.intelligent_scheduler import IntelligentD3QNScheduler
from src.config import ClusterConfig, RLConfig

def run_manual_test(json_filepath: str, scheduler: IntelligentD3QNScheduler):
    print("\n" + "=" * 90)
    print(f"  RUNNING MANUAL TEST: {os.path.basename(json_filepath)}")
    print("=" * 90)

    if not os.path.exists(json_filepath):
        print(f"Error: File not found: {json_filepath}")
        return

    with open(json_filepath, "r") as f:
        data = json.load(f)

    description = data.get("description", "No description provided.")
    print(f"\nScenario Description:\n  {description}\n")

    # Setup Cluster
    cluster = Cluster(scheduler.cluster_config)
    sim_clock = 0.0

    # Optional: Pre-allocate cluster load if specified in JSON
    if "pre_allocate_cluster" in data:
        print("--- PRE-LOADED CLUSTER INITIAL STATE ---")
        for pre in data["pre_allocate_cluster"]:
            nid = pre["node_id"]
            c_used = pre.get("cpu_used", 0.0)
            r_used = pre.get("ram_used", 0.0)
            node = cluster.nodes[nid]
            node.allocated_cpu = min(node.cpu_capacity, float(c_used))
            node.allocated_ram = min(node.ram_capacity, float(r_used))
            # Dummy task holding resources
            hold_dur = pre.get("duration", 25.0)
            dummy = Task(-1, 0.0, node.allocated_cpu, node.allocated_ram, hold_dur, 1, 60.0)
            dummy.finish_time = hold_dur
            node.running_tasks.append(dummy)

    # Parse task list (support single 'task' or list of 'tasks')
    raw_tasks = data.get("tasks")
    if raw_tasks is None and "task" in data:
        raw_tasks = [data["task"]]
    if not raw_tasks:
        print("Error: No tasks found in JSON file.")
        return

    parsed_tasks = []
    current_arrival = 0.0
    for idx, t_info in enumerate(raw_tasks, 1):
        t_id = t_info.get("task_id", idx)
        arr = float(t_info.get("arrival_time", current_arrival))
        current_arrival = max(current_arrival, arr)
        c_req = float(t_info.get("cpu_req", 4.0))
        r_req = float(t_info.get("ram_req", 8.0))
        dur = float(t_info.get("duration", 10.0))
        pri = int(t_info.get("priority", 1))
        slack = float(t_info.get("deadline_slack", 1.8))
        deadline = float(t_info.get("deadline", arr + (dur * slack)))

        parsed_tasks.append(Task(
            task_id=t_id,
            arrival_time=arr,
            cpu_req=c_req,
            ram_req=r_req,
            duration=dur,
            priority=pri,
            deadline=deadline
        ))

    priority_labels = {0: "Batch", 1: "Standard", 2: "Latency-Critical"}

    print(f"Total Tasks in Workload Stream: {len(parsed_tasks)}")
    overview_table = [
        [t.task_id, f"t={t.arrival_time:.1f}s", f"{t.cpu_req} cores", f"{t.ram_req} GB", f"{t.duration}s", priority_labels.get(t.priority, "?"), f"t={t.deadline:.1f}s"]
        for t in parsed_tasks
    ]
    print(tabulate(overview_table, headers=["Task ID", "Arrival", "CPU Req", "RAM Req", "Duration", "Priority", "Deadline"], tablefmt="fancy_grid"))

    # Step through each task sequentially
    completed_log = []
    decision_latencies = []

    for step_num, task in enumerate(parsed_tasks, 1):
        print(f"\n>>> [STEP {step_num}/{len(parsed_tasks)}] Task {task.task_id} Arriving at t={task.arrival_time:.1f}s")

        # Advance clock to task arrival if needed
        if sim_clock < task.arrival_time:
            sim_clock = task.arrival_time
            freed = cluster.release_completed(sim_clock)
            if freed:
                print(f"    (Clock advanced to {sim_clock:.1f}s: Released {len(freed)} finished task(s))")

        # Snapshot node states before scheduling
        node_status_rows = []
        for n in cluster.nodes:
            can_fit = n.can_accommodate(task.cpu_req, task.ram_req)
            align = cluster.calculate_packing_alignment(n.node_id, task)
            node_status_rows.append([
                f"Node {n.node_id} ({n.node_type})",
                f"{n.available_cpu:.1f} / {n.cpu_capacity:.1f}",
                f"{n.available_ram:.1f} / {n.ram_capacity:.1f}",
                f"{n.cpu_utilization*100:.1f}%",
                f"{n.ram_utilization*100:.1f}%",
                f"{n.processing_speed:.2f}x",
                f"{align:.3f}",
                "YES" if can_fit else "NO"
            ])
        print(tabulate(node_status_rows, headers=["Node", "Free CPU", "Free RAM", "CPU Util", "RAM Util", "Speed", "Cosine Pack", "Fit?"], tablefmt="simple"))

        # Generate State Vector & Mask
        remaining_q = len(parsed_tasks) - step_num
        state_vec = cluster.get_state_vector(task, queue_len=remaining_q, current_time=sim_clock)
        action_mask = cluster.get_action_mask(task)

        # Forward Inference
        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=scheduler.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=scheduler.device).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            features = scheduler.policy_net.feature_extractor(state_tensor)
            v_scalar = scheduler.policy_net.value_stream(features).item()
            advantages = scheduler.policy_net.advantage_stream(features).squeeze(0).cpu().numpy()
            q_masked = scheduler.policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()
            chosen_action = int(np.argmax(q_masked))
        infer_us = (time.perf_counter() - t0) * 1e6
        decision_latencies.append(infer_us)

        # Action evaluation
        action_names = [f"Node {i}" for i in range(cluster.num_nodes)] + ["Action 6 (Defer)"]
        action_str = action_names[chosen_action]
        best_q = q_masked[chosen_action]

        print(f"    Inference Latency: {infer_us:.1f} μs | Value V(s): {v_scalar:.3f} | Best Q(s,a): {best_q:.3f}")
        print(f"    --> Decision: {action_str.upper()}")

        # Execute placement or deferral
        if chosen_action < cluster.num_nodes:
            sel_node = cluster.nodes[chosen_action]
            sel_node.allocate(task, sim_clock)
            eff_dur = task.duration / sel_node.processing_speed
            is_met = (task.finish_time <= task.deadline)
            print(f"    Assigned to Node {chosen_action} ({sel_node.node_type}, {sel_node.processing_speed}x). Finishes at t={task.finish_time:.1f}s (Deadline: {task.deadline:.1f}s, SLA Met: {is_met})")
            completed_log.append(task)
        else:
            print("    ACTION 6 TRIGGERED: Task deferred! Advancing clock to next completion event...")
            earliest = cluster.earliest_completion_time()
            if earliest is not None:
                sim_clock = max(sim_clock, earliest)
                freed = cluster.release_completed(sim_clock)
                print(f"    Clock jumped to t={sim_clock:.1f}s (Freed {len(freed)} tasks). Re-allocating task {task.task_id}...")
            else:
                sim_clock += 1.0
            # Force allocation on best feasible node after advancement
            mask_after = cluster.get_action_mask(task, allow_defer=False)
            valid_nodes = np.where(mask_after[:cluster.num_nodes])[0]
            alt_node = int(valid_nodes[0]) if len(valid_nodes) > 0 else 0
            cluster.nodes[alt_node].allocate(task, sim_clock)
            completed_log.append(task)
            print(f"    Task {task.task_id} successfully allocated to Node {alt_node} at t={sim_clock:.1f}s.")

    # Drain cluster after all tasks dispatched
    while True:
        earliest = cluster.earliest_completion_time()
        if earliest is None:
            break
        sim_clock = earliest
        cluster.release_completed(sim_clock)

    # Final Summary Table for the Batch
    print("\n" + "=" * 90)
    print(f"  EXECUTION SUMMARY FOR: {os.path.basename(json_filepath)}")
    print("=" * 90)

    summary_rows = []
    sla_violations = 0
    for t in completed_log:
        if t.finish_time is not None:
            violated = t.finish_time > t.deadline
            if violated:
                sla_violations += 1
            summary_rows.append([
                t.task_id,
                f"Node {t.assigned_node_id}",
                f"{t.arrival_time:.1f}s",
                f"{t.start_time:.1f}s",
                f"{t.finish_time:.1f}s",
                f"{t.deadline:.1f}s",
                f"{t.turnaround_time:.1f}s",
                f"{t.wait_time:.1f}s",
                "VIOLATED" if violated else "MET"
            ])

    print(tabulate(summary_rows, headers=[
        "Task ID", "Assigned Node", "Arrival", "Start", "Finish", "Deadline", "Turnaround", "Wait", "SLA Status"
    ], tablefmt="fancy_grid"))

    total_makespan = max(t.finish_time for t in completed_log if t.finish_time is not None) - min(t.arrival_time for t in completed_log)
    avg_turnaround = np.mean([t.turnaround_time for t in completed_log])
    avg_wait = np.mean([t.wait_time for t in completed_log])
    mean_lat = np.mean(decision_latencies)

    print(f"\nBatch Performance Metrics:")
    print(f"  * Total Makespan:              {total_makespan:.2f} s")
    print(f"  * Average Turnaround Time:     {avg_turnaround:.2f} s")
    print(f"  * Average Wait Time:           {avg_wait:.2f} s")
    print(f"  * SLA Violations:              {sla_violations} / {len(completed_log)} ({sla_violations/len(completed_log)*100:.1f}%)")
    print(f"  * Mean Decision Latency:       {mean_lat:.1f} μs ({mean_lat/1000.0:.4f} ms)")
    print("=" * 90 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Manual Test Runner for NeuraSched")
    parser.add_argument("files", nargs="*", help="JSON test files to evaluate (e.g. test_example_1.json)")
    args = parser.parse_args()

    checkpoint_path = "checkpoints/best_d3qn_scheduler.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint '{checkpoint_path}' not found. Please train first.")
        sys.exit(1)

    scheduler = IntelligentD3QNScheduler(checkpoint_path=checkpoint_path)

    test_files = args.files
    if not test_files:
        test_files = sorted([f for f in os.listdir(".") if f.startswith("test_example_") and f.endswith(".json")])

    if not test_files:
        print("No test_example_*.json files found.")
        sys.exit(1)

    for tf in test_files:
        run_manual_test(tf, scheduler)

if __name__ == "__main__":
    main()
