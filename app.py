import json
import os
import re
import sys
import time

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.version_info[:2] == (3, 9):
    venv_lib = os.path.join(project_root, ".venv", "lib")
    if os.path.isdir(venv_lib):
        for entry in os.listdir(venv_lib):
            site_packages = os.path.join(venv_lib, entry, "site-packages")
            if os.path.isdir(site_packages) and site_packages not in sys.path:
                sys.path.insert(0, site_packages)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import streamlit as st

st.set_page_config(
    page_title="Task Scheduler",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

if sys.version_info[:2] != (3, 9):
    st.error("This project requires Python 3.9. Start it with `./run_frontend.sh`.")
    st.stop()

try:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from src.config import ClusterConfig
    from src.environment.cluster import Cluster
    from src.scheduler.intelligent_scheduler import IntelligentD3QNScheduler
    from src.workload.task import Task
except Exception as error:
    st.error("The app could not load its Python dependencies.")
    st.code(str(error))
    st.stop()

# Design tokens
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
SURFACE = "#FFFFFF"
ACCENT = "#0F766E"
CPU_COLOR = "#4F46E5"
RAM_COLOR = "#0891B2"
GOOD = "#15803D"
WARN = "#B45309"
BAD = "#B91C1C"
PRIORITY_COLORS = {"Batch": "#94A3B8", "Standard": "#4F46E5", "Critical": "#D97706"}

st.markdown(
    """
    <style>
    .stDeployButton, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"], header [data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu, footer { visibility: hidden !important; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }

    .stApp { background: #F6F7F9; color: #0F172A; }
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Header */
    .hdr { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px; margin-bottom: 4px; }
    .hdr-title { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; color: #0F172A; line-height: 1.2; }
    .hdr-sub { font-size: 14px; color: #64748B; margin-top: 4px; }

    /* Tabs */
    [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #E2E8F0; }
    button[data-baseweb="tab"] { padding: 10px 14px !important; background: transparent !important; }
    button[data-baseweb="tab"] p { font-size: 14px !important; font-weight: 600 !important; color: #64748B !important; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: #0F172A !important; }
    [data-baseweb="tab-highlight"] { background-color: #0F766E !important; }
    [data-baseweb="tab-border"] { display: none; }
    [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

    /* Section headings */
    .sec { display: flex; align-items: baseline; gap: 10px; margin: 6px 0 10px 0; }
    .sec-title { font-size: 17px; font-weight: 650; color: #0F172A; letter-spacing: -0.01em; }
    .sec-cap { font-size: 13px; color: #64748B; }

    /* Bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #E2E8F0 !important; border-radius: 12px !important; }


    .bar-label { font-size: 13px; font-weight: 600; color: #334155; margin-bottom: 2px; }
    .list-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin: 6px 0 2px 0; }
    .list-meta { font-size: 13px; color: #64748B; white-space: nowrap; }
    .list-desc { font-size: 13px; color: #64748B; margin-bottom: 10px; line-height: 1.5; }
    .empty { border: 1px dashed #CBD5E1; border-radius: 10px; padding: 36px 16px; text-align: center; color: #64748B; font-size: 14px; background: #fff; margin: 8px 0 12px 0; }

    /* Buttons */
    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
    .stButton > button[kind="primary"] { background: #0F766E; border-color: #0F766E; color: #fff; }
    .stButton > button[kind="primary"]:hover { background: #115E59; border-color: #115E59; color: #fff; }


    /* Metric cards */
    .kpi { background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; height: 100%; }
    .kpi-label { font-size: 12px; font-weight: 500; color: #64748B; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; margin-top: 6px; font-variant-numeric: tabular-nums; }
    .kpi-value.good { color: #15803D; }
    .kpi-value.warn { color: #B45309; }
    .kpi-value.bad { color: #B91C1C; }
    .kpi-sub { font-size: 12px; color: #64748B; margin-top: 2px; }

    /* Status banner */
    .status { border-radius: 10px; padding: 12px 16px; font-size: 14px; margin: 14px 0 6px 0; border: 1px solid; }
    .status b { font-weight: 700; }
    .status.good { background: #F0FDF4; border-color: #BBF7D0; color: #14532D; }
    .status.warn { background: #FFFBEB; border-color: #FDE68A; color: #78350F; }
    .status.bad { background: #FEF2F2; border-color: #FECACA; color: #7F1D1D; }

    /* Node cards */
    .node { background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; }
    .node.idle { opacity: .75; }
    .node-top { display: flex; justify-content: space-between; align-items: flex-start; }
    .node-name { font-size: 16px; font-weight: 700; color: #0F172A; }
    .node-type { font-size: 12px; color: #64748B; margin-top: 1px; }
    .node-count { text-align: right; }
    .node-count-num { font-size: 24px; font-weight: 700; color: #0F172A; line-height: 1; font-variant-numeric: tabular-nums; }
    .node-count-lbl { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: .05em; margin-top: 3px; }
    .meter { margin-top: 14px; }
    .meter-row { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 5px; }
    .meter-name { color: #334155; font-weight: 600; }
    .meter-val { color: #0F172A; font-weight: 600; font-variant-numeric: tabular-nums; }
    .meter-val span { color: #64748B; font-weight: 400; }
    .track { background: #F1F5F9; border-radius: 999px; height: 8px; overflow: hidden; position: relative; }
    .fill { height: 100%; border-radius: 999px; }
    .avg-tick { position: absolute; top: 0; width: 2px; height: 8px; background: #0F172A; opacity: .55; }
    .node-tasks { margin-top: 14px; padding-top: 12px; border-top: 1px solid #F1F5F9; display: flex; flex-wrap: wrap; gap: 6px; min-height: 26px; }
    .tchip { font-size: 11.5px; font-weight: 600; color: #334155; background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 6px; padding: 2px 8px; }
    .tchip.miss { color: #B91C1C; background: #FEF2F2; border-color: #FECACA; }
    .tchip.pre { color: #64748B; background: #fff; border-style: dashed; }
    .none { font-size: 12px; color: #94A3B8; }
    .legend { font-size: 12px; color: #64748B; margin: -4px 0 10px 0; }
    .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 5px -1px 12px; }
    .legend i.tick { width: 2px; background: #0F172A; opacity: .55; border-radius: 0; }

    /* Results table */
    .rt-wrap { overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 12px; background: #fff; }
    table.rt { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.rt th { text-align: left; font-size: 11.5px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: .04em; padding: 10px 14px; background: #F8FAFC; border-bottom: 1px solid #E2E8F0; white-space: nowrap; }
    table.rt td { padding: 10px 14px; border-bottom: 1px solid #F1F5F9; color: #334155; white-space: nowrap; font-variant-numeric: tabular-nums; }
    table.rt tr:last-child td { border-bottom: none; }
    .tl-head { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: -6px; }
    .tl-title { font-size: 14px; font-weight: 650; color: #0F172A; }
    .tl-key { font-size: 12px; color: #64748B; }
    .tl-key i { display: inline-block; width: 12px; height: 8px; border-radius: 2px; margin: 0 5px 0 12px; }
    .tl-key i.dl { width: 2px; height: 12px; background: #0F172A; border-radius: 0; margin-bottom: -2px; }
    .tl-key i.sl { width: 14px; height: 0; border-top: 2px dotted #94A3B8; border-radius: 0; margin-bottom: 3px; }
    table.rt td.id { font-weight: 650; color: #0F172A; }
    table.rt tr.r-miss { background: #FFFBF5; }
    table.rt tr.r-inf { background: #FEF7F7; }
    .badge { font-size: 11.5px; font-weight: 700; padding: 3px 9px; border-radius: 999px; }
    .badge.met { color: #15803D; background: #F0FDF4; border: 1px solid #BBF7D0; }
    .badge.missed { color: #B45309; background: #FFFBEB; border: 1px solid #FDE68A; }
    .badge.inf { color: #B91C1C; background: #FEF2F2; border: 1px solid #FECACA; }
    .pos { color: #15803D; font-weight: 600; }
    .neg { color: #B91C1C; font-weight: 600; }
    .dim { color: #94A3B8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_scheduler():
    checkpoint_path = os.path.join(project_root, "checkpoints", "best_d3qn_scheduler.pt")
    if not os.path.isfile(checkpoint_path):
        return None
    return IntelligentD3QNScheduler(checkpoint_path=checkpoint_path)


def task_rows_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as scenario_file:
        scenario = json.load(scenario_file)

    rows = []
    for index, item in enumerate(scenario.get("tasks", []), 1):
        arrival = float(item.get("arrival_time", 0.0))
        duration = float(item.get("duration", 10.0))
        priority = int(item.get("priority", 1))
        rows.append(
            {
                "Task": str(item.get("task_id", index)),
                "Arrival (s)": arrival,
                "CPU (cores)": float(item.get("cpu_req", 8.0)),
                "RAM (GB)": float(item.get("ram_req", 16.0)),
                "Duration (s)": duration,
                "Priority": {0: "Batch", 1: "Standard", 2: "Critical"}.get(priority, "Standard"),
                "Deadline (s)": float(
                    item.get("deadline", arrival + duration * float(item.get("deadline_slack", 1.8)))
                ),
            }
        )
    return rows, scenario.get("pre_allocated", scenario.get("pre_allocate_cluster", []))


def simulate_schedule(tasks, scheduler, pre_allocations=None):
    cluster = Cluster(scheduler.cluster_config)

    for allocation in pre_allocations or []:
        node = cluster.nodes[int(allocation["node_id"])]
        node.allocated_cpu = float(allocation.get("cpu_used", 0.0))
        node.allocated_ram = float(allocation.get("ram_used", 0.0))
        existing_task = Task(
            -1,
            0.0,
            node.allocated_cpu,
            node.allocated_ram,
            float(allocation.get("duration", 25.0)),
            1,
            60.0,
        )
        existing_task.finish_time = float(allocation.get("duration", 25.0))
        node.running_tasks.append(existing_task)

    results = []
    current_time = 0.0
    sorted_tasks = sorted(
        tasks, key=lambda queued_task: (queued_task.arrival_time, queued_task.priority * -1)
    )

    for task in sorted_tasks:
        task.reset()

        # Physical capability check across cluster
        capable_nodes = [
            n.node_id
            for n in cluster.nodes
            if n.cpu_capacity >= task.cpu_req and n.ram_capacity >= task.ram_req
        ]
        if not capable_nodes:
            results.append(
                {
                    "Task": str(task.task_id),
                    "Physical Capability": "Infeasible (No node has capacity)",
                    "Allotted Node": "Rejected (Infeasible)",
                    "Feasible at Arrival": "None",
                    "Start": None,
                    "Finish": None,
                    "Deadline": task.deadline,
                    "SLA Status": "Infeasible",
                    "Difference": "Infeasible",
                    "Wait": 0.0,
                    "Decision (us)": 0.0,
                }
            )
            continue

        # Advance time to task arrival and release completed tasks
        current_time = max(current_time, task.arrival_time)
        cluster.release_completed(current_time)

        # Check which nodes are physically feasible at the moment of arrival
        mask_at_arr = cluster.get_action_mask(task)
        feasible_at_arr = [f"Node {i}" for i in range(cluster.num_nodes) if mask_at_arr[i]]
        feasible_desc = ", ".join(feasible_at_arr) if feasible_at_arr else "None (Busy)"

        # If all capable nodes are currently busy, queue task and advance clock until a capable node becomes free
        while not np.any(cluster.get_action_mask(task)[: cluster.num_nodes]):
            earliest = cluster.earliest_completion_time()
            if earliest is None:
                break
            current_time = max(current_time, earliest)
            cluster.release_completed(current_time)

        action_mask = cluster.get_action_mask(task)
        started = time.perf_counter()
        action = scheduler.select_node(cluster, task, action_mask, current_time)
        decision_time_us = (time.perf_counter() - started) * 1_000_000

        valid_nodes = np.where(action_mask[: cluster.num_nodes])[0]
        if action >= cluster.num_nodes or not action_mask[action]:
            action = int(valid_nodes[0]) if len(valid_nodes) > 0 else capable_nodes[0]

        node = cluster.nodes[action]
        allocated = node.allocate(task, current_time)
        if not allocated:
            # Fallback advance until slot opens
            while not node.can_accommodate(task.cpu_req, task.ram_req):
                earliest = cluster.earliest_completion_time()
                if earliest is None:
                    break
                current_time = max(current_time, earliest)
                cluster.release_completed(current_time)
            node.allocate(task, current_time)

        margin = task.deadline - task.finish_time
        is_violated = task.finish_time > task.deadline

        diff_str = f"+{margin:.1f} s (Ahead)" if not is_violated else f"{margin:.1f} s (Violated)"
        result_str = "Met" if not is_violated else "Missed"

        capable_desc = ", ".join([f"Node {nid}" for nid in capable_nodes])

        results.append(
            {
                "Task": str(task.task_id),
                "Physical Capability": capable_desc,
                "Allotted Node": f"Node {node.node_id} ({node.node_type}, {node.processing_speed:.2f}x)",
                "Feasible at Arrival": feasible_desc,
                "Start": task.start_time,
                "Finish": task.finish_time,
                "Deadline": task.deadline,
                "SLA Status": result_str,
                "Difference": diff_str,
                "Wait": task.wait_time,
                "Decision (us)": decision_time_us,
            }
        )

    valid_runs = [r for r in results if r["SLA Status"] != "Infeasible"]
    summary = {
        "tasks": len(results),
        "met": sum(row["SLA Status"] == "Met" for row in results),
        "missed": sum(row["SLA Status"] == "Missed" for row in results),
        "infeasible": sum(row["SLA Status"] == "Infeasible" for row in results),
        "avg_wait": float(np.mean([row["Wait"] for row in valid_runs])) if valid_runs else 0.0,
        "makespan": float(max([row["Finish"] for row in valid_runs])) if valid_runs else 0.0,
    }

    node_availability = [
        {
            "Node": f"Node {node.node_id}",
            "Type": node.node_type,
            "Speed": node.processing_speed,
            "CPU total": node.cpu_capacity,
            "CPU used": node.allocated_cpu,
            "CPU free": node.available_cpu,
            "RAM total": node.ram_capacity,
            "RAM used": node.allocated_ram,
            "RAM free": node.available_ram,
            "Running tasks": [
                str(running_task.task_id) for running_task in node.running_tasks
            ],
        }
        for node in cluster.nodes
    ]

    return results, summary, node_availability


# Presets
DEFAULT_TASKS = [
    {
        "Task": "TASK-1",
        "Arrival (s)": 3.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 12.0,
        "Priority": "Standard",
        "Deadline (s)": 21.0,
    },
    {
        "Task": "TASK-2",
        "Arrival (s)": 4.0,
        "CPU (cores)": 16.0,
        "RAM (GB)": 64.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 16.0,
    },
    {
        "Task": "TASK-3",
        "Arrival (s)": 5.0,
        "CPU (cores)": 8.0,
        "RAM (GB)": 128.0,
        "Duration (s)": 20.0,
        "Priority": "Standard",
        "Deadline (s)": 41.0,
    },
    {
        "Task": "TASK-4",
        "Arrival (s)": 6.0,
        "CPU (cores)": 4.0,
        "RAM (GB)": 16.0,
        "Duration (s)": 8.0,
        "Priority": "Batch",
        "Deadline (s)": 30.0,
    },
    {
        "Task": "TASK-5",
        "Arrival (s)": 0.0,
        "CPU (cores)": 8.0,
        "RAM (GB)": 128.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 20.0,
    },
]

CONTENTION_TASKS = [
    {
        "Task": "STRESS-1",
        "Arrival (s)": 1.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 48.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 20.0,
    },
    {
        "Task": "STRESS-2",
        "Arrival (s)": 2.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 48.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 20.0,
    },
    {
        "Task": "STRESS-3",
        "Arrival (s)": 3.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 15.0,
    },
    {
        "Task": "STRESS-4",
        "Arrival (s)": 4.0,
        "CPU (cores)": 64.0,
        "RAM (GB)": 128.0,
        "Duration (s)": 15.0,
        "Priority": "Batch",
        "Deadline (s)": 40.0,
    },
    {
        "Task": "STRESS-5",
        "Arrival (s)": 5.0,
        "CPU (cores)": 32.0,
        "RAM (GB)": 256.0,
        "Duration (s)": 20.0,
        "Priority": "Standard",
        "Deadline (s)": 32.0,
    },
]

FAILING_CASE_TASKS = [
    {
        "Task": "FAIL-INF1",
        "Arrival (s)": 1.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 96.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 25.0,
    },
    {
        "Task": "FAIL-CON1",
        "Arrival (s)": 2.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 20.0,
    },
    {
        "Task": "FAIL-CON2",
        "Arrival (s)": 2.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 20.0,
    },
    {
        "Task": "FAIL-CON3",
        "Arrival (s)": 3.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 14.0,
    },
]

WORKING_CASE_TASKS = [
    {
        "Task": "WORK-1",
        "Arrival (s)": 1.0,
        "CPU (cores)": 96.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 20.0,
    },
    {
        "Task": "WORK-2",
        "Arrival (s)": 1.0,
        "CPU (cores)": 16.0,
        "RAM (GB)": 180.0,
        "Duration (s)": 12.0,
        "Priority": "Critical",
        "Deadline (s)": 28.0,
    },
    {
        "Task": "WORK-3",
        "Arrival (s)": 2.0,
        "CPU (cores)": 32.0,
        "RAM (GB)": 64.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 22.0,
    },
    {
        "Task": "WORK-4",
        "Arrival (s)": 3.0,
        "CPU (cores)": 16.0,
        "RAM (GB)": 32.0,
        "Duration (s)": 8.0,
        "Priority": "Batch",
        "Deadline (s)": 20.0,
    },
]

HEAVY_LOAD_TASKS = [
    {
        "Task": "LOAD-1",
        "Arrival (s)": 0.0,
        "CPU (cores)": 88.0,
        "RAM (GB)": 48.0,
        "Duration (s)": 12.0,
        "Priority": "Critical",
        "Deadline (s)": 30.0,
    },
    {
        "Task": "LOAD-2",
        "Arrival (s)": 0.5,
        "CPU (cores)": 84.0,
        "RAM (GB)": 56.0,
        "Duration (s)": 12.0,
        "Priority": "Standard",
        "Deadline (s)": 32.0,
    },
    {
        "Task": "LOAD-3",
        "Arrival (s)": 1.0,
        "CPU (cores)": 24.0,
        "RAM (GB)": 220.0,
        "Duration (s)": 15.0,
        "Priority": "Standard",
        "Deadline (s)": 40.0,
    },
    {
        "Task": "LOAD-4",
        "Arrival (s)": 1.5,
        "CPU (cores)": 28.0,
        "RAM (GB)": 232.0,
        "Duration (s)": 15.0,
        "Priority": "Batch",
        "Deadline (s)": 45.0,
    },
    {
        "Task": "LOAD-5",
        "Arrival (s)": 2.0,
        "CPU (cores)": 56.0,
        "RAM (GB)": 112.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 30.0,
    },
    {
        "Task": "LOAD-6",
        "Arrival (s)": 2.5,
        "CPU (cores)": 60.0,
        "RAM (GB)": 104.0,
        "Duration (s)": 10.0,
        "Priority": "Critical",
        "Deadline (s)": 28.0,
    },
    {
        "Task": "LOAD-7",
        "Arrival (s)": 3.0,
        "CPU (cores)": 48.0,
        "RAM (GB)": 40.0,
        "Duration (s)": 8.0,
        "Priority": "Batch",
        "Deadline (s)": 45.0,
    },
    {
        "Task": "LOAD-8",
        "Arrival (s)": 4.0,
        "CPU (cores)": 16.0,
        "RAM (GB)": 96.0,
        "Duration (s)": 10.0,
        "Priority": "Standard",
        "Deadline (s)": 45.0,
    },
]

TEST_CASES = {
    "failing": {
        "name": "Failing Scenario (Infeasible + Deadline Miss)",
        "tag": "Failing case",
        "body": "One task needs 96 cores and 96 GB, which no node can host. Three 96-core jobs then fight over two compute nodes, so one misses its deadline.",
        "rows": FAILING_CASE_TASKS,
    },
    "load": {
        "name": "Heavy Load Test",
        "tag": "Load test",
        "body": "Eight large jobs sized to fill all three node types at once. Later arrivals must queue until capacity frees up.",
        "rows": HEAVY_LOAD_TASKS,
    },
    "working": {
        "name": "Working Scenario (100% SLA Admitted)",
        "tag": "Passing case",
        "body": "Four jobs that each match a node type cleanly. No queueing and every deadline is met.",
        "rows": WORKING_CASE_TASKS,
    },
}

OTHER_PRESETS = {
    "Standard Multi-Task": DEFAULT_TASKS,
    "Resource Contention & Queue Delay Stress Test": CONTENTION_TASKS,
}

PRIORITY_CODES = {"Batch": 0, "Standard": 1, "Critical": 2}
TASK_COLUMNS = ["Task", "Arrival (s)", "CPU (cores)", "RAM (GB)", "Duration (s)", "Priority", "Deadline (s)"]
RESULT_KEYS = ("schedule_results", "schedule_summary", "node_availability", "schedule_tasks")


# =========================================================================
# Workload helpers
# =========================================================================
def load_workload(name, rows, pre_allocations=None, description=""):
    st.session_state["current_workload_name"] = name
    st.session_state["workload_description"] = description
    st.session_state["active_tasks_list"] = [dict(t) for t in rows]
    st.session_state["pre_allocations"] = list(pre_allocations or [])
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
    for k in RESULT_KEYS:
        st.session_state.pop(k, None)


def validate_rows(rows):
    errors = []
    seen = set()
    for i, r in enumerate(rows, 1):
        tid = str(r.get("Task") or "").strip()
        if not tid:
            errors.append(f"Row {i}: Task ID is empty.")
            continue
        if tid in seen:
            errors.append(f"Task ID '{tid}' is used more than once.")
        seen.add(tid)
        for col in TASK_COLUMNS[1:5] + ["Deadline (s)"]:
            v = r.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                errors.append(f"{tid}: '{col}' is empty.")
        if r.get("Priority") not in PRIORITY_CODES:
            errors.append(f"{tid}: choose a priority.")
        if not errors and float(r["Deadline (s)"]) < float(r["Arrival (s)"]):
            errors.append(f"{tid}: deadline is earlier than arrival.")
    return errors


def run_schedule(rows):
    scheduler = load_scheduler()
    if scheduler is None:
        st.session_state["run_error"] = "Model checkpoint missing at `checkpoints/best_d3qn_scheduler.pt`."
        return
    tasks_to_run = [
        Task(
            task_id=str(r["Task"]).strip(),
            arrival_time=float(r["Arrival (s)"]),
            cpu_req=float(r["CPU (cores)"]),
            ram_req=float(r["RAM (GB)"]),
            duration=float(r["Duration (s)"]),
            priority=PRIORITY_CODES[r["Priority"]],
            deadline=float(r["Deadline (s)"]),
        )
        for r in rows
    ]
    results, summary, node_availability = simulate_schedule(
        tasks_to_run, scheduler, st.session_state.get("pre_allocations", [])
    )
    st.session_state["schedule_results"] = results
    st.session_state["schedule_summary"] = summary
    st.session_state["node_availability"] = node_availability
    st.session_state["schedule_tasks"] = [dict(r) for r in rows]


def node_id_of(result_row):
    match = re.match(r"Node (\d+)", str(result_row.get("Allotted Node", "")))
    return int(match.group(1)) if match else None


def node_utilization(cluster_nodes, results, task_rows, pre_allocations, makespan):
    """Per-node peak and time-averaged CPU/RAM utilization, plus allotted tasks."""
    demand = {str(r["Task"]).strip(): (float(r["CPU (cores)"]), float(r["RAM (GB)"])) for r in task_rows}
    intervals = {n.node_id: [] for n in cluster_nodes}
    allotted = {n.node_id: [] for n in cluster_nodes}

    for alloc in pre_allocations or []:
        nid = int(alloc["node_id"])
        if nid in intervals:
            intervals[nid].append(
                (0.0, float(alloc.get("duration", 25.0)), float(alloc.get("cpu_used", 0.0)), float(alloc.get("ram_used", 0.0)))
            )

    for r in results:
        nid = node_id_of(r)
        if nid is None or r.get("Start") is None or r.get("Finish") is None:
            continue
        cpu, ram = demand.get(str(r["Task"]), (0.0, 0.0))
        intervals[nid].append((float(r["Start"]), float(r["Finish"]), cpu, ram))
        allotted[nid].append((str(r["Task"]), r.get("SLA Status") == "Missed"))

    horizon = max([makespan] + [iv[1] for ivs in intervals.values() for iv in ivs] + [1e-9])
    stats = []
    for n in cluster_nodes:
        ivs = intervals[n.node_id]
        peak_cpu = peak_ram = 0.0
        for t, _, _, _ in ivs:
            active = [iv for iv in ivs if iv[0] <= t < iv[1]]
            peak_cpu = max(peak_cpu, sum(iv[2] for iv in active))
            peak_ram = max(peak_ram, sum(iv[3] for iv in active))
        avg_cpu = sum((e - s) * c for s, e, c, _ in ivs) / horizon
        avg_ram = sum((e - s) * r for s, e, _, r in ivs) / horizon
        stats.append(
            {
                "node": n,
                "peak_cpu": peak_cpu,
                "peak_ram": peak_ram,
                "avg_cpu_pct": avg_cpu / n.cpu_capacity * 100.0,
                "avg_ram_pct": avg_ram / n.ram_capacity * 100.0,
                "tasks": allotted[n.node_id],
                "pre": any(True for a in (pre_allocations or []) if int(a["node_id"]) == n.node_id),
            }
        )
    return stats


# =========================================================================
# Render helpers
# =========================================================================
def section(title, caption=""):
    cap = f'<span class="sec-cap">{caption}</span>' if caption else ""
    st.markdown(
        f'<div class="sec"><span class="sec-title">{title}</span>{cap}</div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub="", tone=""):
    col.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {tone}">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def meter_color(pct, base):
    if pct >= 95:
        return BAD
    return base


def meter(name, used, total, unit, pct, avg_pct, base_color):
    width = min(pct, 100.0)
    tick = f'<div class="avg-tick" style="left: calc({min(avg_pct, 100.0):.1f}% - 1px);"></div>' if avg_pct > 0 else ""
    return (
        f'<div class="meter"><div class="meter-row"><span class="meter-name">{name}</span>'
        f'<span class="meter-val">{pct:.0f}% <span>&middot; {used:.0f} / {total:.0f} {unit}</span></span></div>'
        f'<div class="track"><div class="fill" style="width: {width:.1f}%; background: {meter_color(pct, base_color)};"></div>{tick}</div></div>'
    )


def node_card_html(s, has_run):
    n = s["node"]
    cpu_pct = s["peak_cpu"] / n.cpu_capacity * 100.0
    ram_pct = s["peak_ram"] / n.ram_capacity * 100.0
    chips = "".join(
        f'<span class="tchip{" miss" if missed else ""}">{tid}</span>' for tid, missed in s["tasks"]
    )
    if s["pre"]:
        chips = '<span class="tchip pre">pre-loaded</span>' + chips
    if not chips:
        chips = f'<span class="none">{"No tasks allotted" if has_run else "Waiting for schedule"}</span>'
    idle = " idle" if has_run and not s["tasks"] else ""
    return (
        f'<div class="node{idle}">'
        f'<div class="node-top"><div><div class="node-name">Node {n.node_id}</div>'
        f'<div class="node-type">{n.node_type.replace("-", " ").title()} &middot; {n.processing_speed:.2f}x speed</div></div>'
        f'<div class="node-count"><div class="node-count-num">{len(s["tasks"])}</div><div class="node-count-lbl">Tasks</div></div></div>'
        + meter("CPU", s["peak_cpu"], n.cpu_capacity, "cores", cpu_pct, s["avg_cpu_pct"], CPU_COLOR)
        + meter("RAM", s["peak_ram"], n.ram_capacity, "GB", ram_pct, s["avg_ram_pct"], RAM_COLOR)
        + f'<div class="node-tasks">{chips}</div></div>'
    )


def results_table_html(results):
    body = []
    for r in results:
        status = r.get("SLA Status", "")
        if status == "Infeasible":
            row_cls, badge = "r-inf", '<span class="badge inf">Infeasible</span>'
            node = '<span class="neg">Rejected</span>'
            diff = '<span class="dim">&ndash;</span>'
        else:
            nid = node_id_of(r)
            node = f"Node {nid}" if nid is not None else str(r.get("Allotted Node", "-"))
            margin = float(r["Deadline"]) - float(r["Finish"])
            if status == "Missed":
                row_cls, badge = "r-miss", '<span class="badge missed">Missed</span>'
                diff = f'<span class="neg">{margin:.1f} s</span>'
            else:
                row_cls, badge = "", '<span class="badge met">Met</span>'
                diff = f'<span class="pos">+{margin:.1f} s</span>'

        def fmt(v):
            return f"{v:.1f} s" if v is not None else '<span class="dim">&ndash;</span>'

        body.append(
            f'<tr class="{row_cls}"><td class="id">{r.get("Task", "-")}</td><td>{node}</td>'
            f'<td>{fmt(r.get("Start"))}</td><td>{fmt(r.get("Finish"))}</td><td>{fmt(r.get("Deadline"))}</td>'
            f'<td>{badge}</td><td>{diff}</td></tr>'
        )
    head = "".join(
        f"<th>{h}</th>"
        for h in ["Task", "Node", "Start", "Finish", "Deadline", "SLA", "Slack"]
    )
    return f'<div class="rt-wrap"><table class="rt"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def timeline_figure(results, task_rows):
    """Gantt view: one row per task showing queue wait, run time, deadline and slack."""
    arrival_of = {str(r["Task"]).strip(): float(r["Arrival (s)"]) for r in task_rows}
    ids = [str(r.get("Task")) for r in results]
    labels = list(ids)
    label_of = dict(zip(ids, labels))

    wait = {"y": [], "x": [], "base": []}
    run = {"y": [], "x": [], "base": [], "color": [], "hover": []}
    slack_x, slack_y = [], []
    dl_x, dl_y = [], []
    end_x, end_y, end_text, end_color = [], [], [], []
    rej_x, rej_y = [], []

    for r in results:
        tid = str(r.get("Task"))
        y = label_of[tid]
        deadline = float(r["Deadline"])
        arrival = arrival_of.get(tid, 0.0)
        dl_x.append(deadline)
        dl_y.append(y)
        if r.get("SLA Status") == "Infeasible" or r.get("Start") is None:
            rej_x.append(arrival)
            rej_y.append(y)
            continue
        start, finish = float(r["Start"]), float(r["Finish"])
        missed = finish > deadline
        if start > arrival:
            wait["y"].append(y)
            wait["x"].append(start - arrival)
            wait["base"].append(arrival)
        run["y"].append(y)
        run["x"].append(finish - start)
        run["base"].append(start)
        run["color"].append(BAD if missed else ACCENT)
        nid = node_id_of(r)
        run["hover"].append(
            f"<b>{tid}</b> on Node {nid}<br>"
            f"Arrived {arrival:.1f}s, waited {start - arrival:.1f}s<br>"
            f"Ran {start:.1f}s to {finish:.1f}s<br>"
            f"Deadline {deadline:.1f}s: {'missed by %.1fs' % (finish - deadline) if missed else 'met, %.1fs to spare' % (deadline - finish)}"
        )
        if not missed:
            slack_x += [finish, deadline, None]
            slack_y += [y, y, None]
        end_x.append(max(finish, deadline))
        end_y.append(y)
        end_text.append(f"Node {nid}  {'-' if missed else '+'}{abs(deadline - finish):.1f}s")
        end_color.append(BAD if missed else MUTED)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=wait["y"], x=wait["x"], base=wait["base"], orientation="h", name="Waiting",
        marker=dict(color="#E2E8F0", line=dict(width=0)),
        hovertemplate="Waiting in queue %{x:.1f}s<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=run["y"], x=run["x"], base=run["base"], orientation="h", name="Running",
        marker=dict(color=run["color"], cornerradius=4, line=dict(width=0)),
        hovertext=run["hover"], hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=slack_x, y=slack_y, mode="lines", hoverinfo="skip",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dl_x, y=dl_y, mode="markers",
        marker=dict(symbol="line-ns", size=16, line=dict(color=INK, width=2)),
        hovertemplate="Deadline %{x:.1f}s<extra></extra>",
    ))
    if rej_y:
        fig.add_trace(go.Scatter(
            x=rej_x, y=rej_y, mode="markers+text",
            marker=dict(symbol="x-thin", size=11, line=dict(color=BAD, width=2.5)),
            text=["Rejected, no node fits"] * len(rej_y), textposition="middle right",
            textfont=dict(color=BAD, size=11.5),
            hovertemplate="Rejected: no node has enough CPU and RAM<extra></extra>",
        ))

    x_end = max(dl_x + [b + w for b, w in zip(run["base"], run["x"])] + [1.0])
    pad = x_end * 0.02
    end_x = [x + pad for x in end_x]
    fig.add_trace(go.Scatter(
        x=end_x, y=end_y, mode="text", text=end_text, textposition="middle right",
        textfont=dict(color=end_color, size=11.5), hoverinfo="skip",
    ))
    fig.update_layout(
        barmode="overlay",
        showlegend=False,
        height=70 + 40 * len(ids),
        margin=dict(l=8, r=12, t=8, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12),
        bargap=0.48,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(color=INK, size=12), align="left"),
    )
    fig.update_xaxes(
        range=[0, x_end * 1.32], showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor=LINE,
        ticksuffix="s", tickfont=dict(size=11), fixedrange=True,
        title=dict(text="Simulation time", font=dict(size=11, color=MUTED), standoff=6),
    )
    fig.update_yaxes(
        type="category", categoryorder="array", categoryarray=labels, autorange="reversed",
        tickfont=dict(color=INK, size=12), ticks="", ticklabelstandoff=8, automargin=True,
        showgrid=False, fixedrange=True,
    )
    return fig


# =========================================================================
# Session defaults
# =========================================================================
if "active_tasks_list" not in st.session_state:
    load_workload("Standard Multi-Task", DEFAULT_TASKS)

base_cluster = Cluster(ClusterConfig())

# =========================================================================
# Header
# =========================================================================
total_cores = sum(n.cpu_capacity for n in base_cluster.nodes)
total_ram = sum(n.ram_capacity for n in base_cluster.nodes)
st.markdown(
    f"""
    <div class="hdr">
        <div>
            <div class="hdr-title">Task Scheduler</div>
            <div class="hdr-sub">D3QN task placement on a {len(base_cluster.nodes)}-node cluster
            ({total_cores:.0f} cores, {total_ram:.0f} GB RAM)</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_schedule, tab_benchmarks = st.tabs(["Scheduler", "Benchmarks"])

# =========================================================================
# TAB 1: SCHEDULER
# =========================================================================
with tab_schedule:
    # ---- Test cases ----------------------------------------------------
    scenario_files = sorted(
        name
        for name in os.listdir(project_root)
        if name.startswith("test_example_") and name.endswith(".json")
    )
    more_options = list(OTHER_PRESETS) + scenario_files

    with st.container(border=True):
        st.markdown('<div class="bar-label">Test cases</div>', unsafe_allow_html=True)
        tc = st.columns([1, 1, 1, 2.2, 0.8], vertical_alignment="bottom")
        for col, (key, case) in zip(tc[:3], TEST_CASES.items()):
            with col:
                if st.button(case["tag"], key=f"case_{key}", help=case["body"], width="stretch"):
                    load_workload(case["name"], case["rows"], [], case["body"])
                    run_schedule(case["rows"])
                    st.rerun()
        with tc[3]:
            picked = st.selectbox("Other scenarios", more_options, label_visibility="collapsed")
        with tc[4]:
            load_more = st.button("Load", key="load_more", width="stretch")
        if load_more:
            if picked in OTHER_PRESETS:
                rows, prealloc, desc = OTHER_PRESETS[picked], [], ""
            else:
                path = os.path.join(project_root, picked)
                rows, prealloc = task_rows_from_file(path)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        desc = json.load(fh).get("description", "")
                except Exception:
                    desc = ""
            load_workload(picked, rows, prealloc, desc)
            if rows:
                run_schedule(rows)
            st.rerun()

    st.write("")

    # ---- Workload: add form + task table ---------------------------------
    current_tasks = st.session_state["active_tasks_list"]
    prealloc = st.session_state.get("pre_allocations", [])

    col_add, col_list = st.columns([1, 2.3], gap="large")

    with col_add:
        section("Add task")
        with st.form("form_add_task", clear_on_submit=True, border=True):
            f1, f2 = st.columns(2)
            in_task_id = f1.text_input("Task ID", value=f"T{len(current_tasks) + 1}")
            in_priority = f2.selectbox("Priority", list(PRIORITY_CODES), index=1)
            f3, f4 = st.columns(2)
            in_cpu = f3.number_input("CPU (cores)", min_value=0.5, max_value=96.0, value=8.0, step=1.0)
            in_ram = f4.number_input("RAM (GB)", min_value=1.0, max_value=256.0, value=16.0, step=1.0)
            f5, f6 = st.columns(2)
            in_arrival = f5.number_input("Arrival (s)", min_value=0.0, value=0.0, step=0.5)
            in_duration = f6.number_input("Duration (s)", min_value=0.1, value=10.0, step=1.0)
            in_deadline = st.number_input(
                "Deadline (s)", min_value=0.0, value=0.0, step=1.0,
                help="Absolute time the task must finish by. Leave at 0 to use arrival + 2 x duration.",
            )
            btn_add_task = st.form_submit_button("Add task", type="primary", width="stretch")

    with col_list:
        name = st.session_state.get("current_workload_name", "Custom")
        desc = st.session_state.get("workload_description", "")
        meta = f"{len(current_tasks)} task{'s' if len(current_tasks) != 1 else ''}"
        if prealloc:
            meta += f", {len(prealloc)} nodes pre-loaded"
        st.markdown(
            f'<div class="list-head"><span class="sec-title">{name}</span><span class="list-meta">{meta}</span></div>'
            + (f'<div class="list-desc">{desc}</div>' if desc else ""),
            unsafe_allow_html=True,
        )

        if current_tasks:
            edited_df = st.data_editor(
                pd.DataFrame(current_tasks, columns=TASK_COLUMNS),
                key=f"task_editor_{st.session_state.get('editor_version', 0)}",
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Task": st.column_config.TextColumn("Task ID", required=True),
                    "Arrival (s)": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.1f", default=0.0),
                    "CPU (cores)": st.column_config.NumberColumn("CPU", min_value=0.5, max_value=96.0, step=0.5, format="%.1f", default=8.0),
                    "RAM (GB)": st.column_config.NumberColumn("RAM (GB)", min_value=1.0, max_value=256.0, step=1.0, format="%.0f", default=16.0),
                    "Duration (s)": st.column_config.NumberColumn(min_value=0.1, step=1.0, format="%.1f", default=10.0),
                    "Priority": st.column_config.SelectboxColumn(options=list(PRIORITY_CODES), default="Standard", required=True),
                    "Deadline (s)": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.1f", default=30.0),
                },
            )
            st.caption("Click a cell to edit. Select rows and press Delete to remove them.")
        else:
            edited_df = pd.DataFrame(columns=TASK_COLUMNS)
            st.markdown(
                '<div class="empty">No tasks yet. Add one on the left or load a test case.</div>',
                unsafe_allow_html=True,
            )

        a1, a2, _ = st.columns([1.3, 1, 2])
        with a1:
            btn_run_schedule = st.button("Run schedule", type="primary", width="stretch")
        with a2:
            btn_clear = st.button("Clear all", width="stretch", disabled=not current_tasks)

    def table_rows():
        return [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
            for row in edited_df.to_dict("records")
        ]

    if btn_add_task:
        rows = table_rows()
        deadline = float(in_deadline) if in_deadline > 0 else float(in_arrival) + 2.0 * float(in_duration)
        new_row = {
            "Task": in_task_id.strip(),
            "Arrival (s)": float(in_arrival),
            "CPU (cores)": float(in_cpu),
            "RAM (GB)": float(in_ram),
            "Duration (s)": float(in_duration),
            "Priority": in_priority,
            "Deadline (s)": deadline,
        }
        errors = validate_rows(rows + [new_row])
        if errors:
            with col_add:
                st.error("  \n".join(errors))
        else:
            st.session_state["active_tasks_list"] = rows + [new_row]
            st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
            if not st.session_state.get("current_workload_name"):
                st.session_state["current_workload_name"] = "Custom workload"
            if not any(n.cpu_capacity >= new_row["CPU (cores)"] and n.ram_capacity >= new_row["RAM (GB)"] for n in base_cluster.nodes):
                st.session_state["add_notice"] = f"{new_row['Task']} added, but no node has {new_row['CPU (cores)']:.0f} cores and {new_row['RAM (GB)']:.0f} GB together. It will be rejected."
            st.rerun()

    if st.session_state.get("add_notice"):
        with col_add:
            st.warning(st.session_state.pop("add_notice"))

    if btn_clear:
        load_workload("Custom workload", [], [])
        st.rerun()

    if btn_run_schedule:
        rows = table_rows()
        errors = validate_rows(rows) if rows else ["Add at least one task before running the schedule."]
        if errors:
            with col_list:
                st.error("  \n".join(errors))
        else:
            st.session_state["active_tasks_list"] = rows
            run_schedule(rows)

    if st.session_state.get("run_error"):
        st.error(st.session_state.pop("run_error"))

    st.write("")

    # ---- Results ---------------------------------------------------------
    has_run = "schedule_results" in st.session_state
    section("Nodes")

    if has_run:
        results = st.session_state["schedule_results"]
        summary = st.session_state["schedule_summary"]
        run_rows = st.session_state.get("schedule_tasks", current_tasks)

        total = summary["tasks"]
        met, missed, infeasible = summary["met"], summary["missed"], summary["infeasible"]
        rate = (met / total * 100.0) if total else 0.0

        k = st.columns(6)
        kpi(k[0], "SLA success", f"{rate:.0f}%", f"{met} of {total} tasks",
            "good" if rate == 100 else ("warn" if rate >= 75 else "bad"))
        kpi(k[1], "Met", f"{met}", "on time", "good" if met else "")
        kpi(k[2], "Missed", f"{missed}", "deadline violated", "warn" if missed else "")
        kpi(k[3], "Infeasible", f"{infeasible}", "no node fits", "bad" if infeasible else "")
        kpi(k[4], "Makespan", f"{summary['makespan']:.1f}s", "last task finishes")
        kpi(k[5], "Avg wait", f"{summary['avg_wait']:.1f}s", "time in queue")

        if infeasible or missed:
            parts = []
            if infeasible:
                parts.append(f"<b>{infeasible}</b> task(s) rejected because no node has enough CPU and RAM")
            if missed:
                parts.append(f"<b>{missed}</b> task(s) missed their deadline due to queueing")
            tone = "bad" if infeasible else "warn"
            st.markdown(f'<div class="status {tone}">{"; ".join(parts)}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="status good">All <b>{total}</b> tasks were placed and finished before their deadline.</div>',
                unsafe_allow_html=True,
            )
        node_stats = node_utilization(base_cluster.nodes, results, run_rows, prealloc, summary["makespan"])
    else:
        node_stats = node_utilization(base_cluster.nodes, [], [], prealloc, 0.0)

    st.write("")
    st.markdown(
        f'<div class="legend">Bar shows peak usage, <i class="tick"></i>marks the average'
        f'<i style="background:{CPU_COLOR}"></i>CPU<i style="background:{RAM_COLOR}"></i>RAM</div>',
        unsafe_allow_html=True,
    )
    node_cols = st.columns(3, gap="medium")
    for idx, s in enumerate(node_stats):
        with node_cols[idx % 3]:
            st.markdown(node_card_html(s, has_run), unsafe_allow_html=True)

    if has_run:
        st.write("")
        section("Results")
        rc1, rc2 = st.columns([1.1, 1], gap="medium")
        with rc1:
            st.markdown(results_table_html(results), unsafe_allow_html=True)
        with rc2:
            with st.container(border=True):
                st.markdown(
                    '<div class="tl-head"><span class="tl-title">Timeline</span><span class="tl-key">'
                    '<i style="background:#E2E8F0"></i>queued'
                    f'<i style="background:{ACCENT}"></i>on time'
                    f'<i style="background:{BAD}"></i>late'
                    '<i class="dl"></i>deadline'
                    '<i class="sl"></i>spare time</span></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    timeline_figure(results, run_rows),
                    theme=None,
                    config={"displayModeBar": False},
                )
        st.write("")
        st.download_button(
            "Download results CSV",
            pd.DataFrame(results).to_csv(index=False),
            file_name="task-scheduling-results.csv",
            mime="text/csv",
        )

# =========================================================================
# TAB 2: BENCHMARKS
# =========================================================================
with tab_benchmarks:
    st.markdown(
        '<div class="sec"><span class="sec-title">Benchmark suite</span>'
        '<span class="sec-cap">D3QN against FIFO, SJF, EDF, Tetris and Round-Robin.</span></div>',
        unsafe_allow_html=True,
    )

    benchmark_path = os.path.join(project_root, "benchmark_results.json")
    if os.path.isfile(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        b_summary = b_data.get("summary", {})

        bench_table = []
        for name, vals in b_summary.items():
            if isinstance(vals, dict):
                bench_table.append(
                    {
                        "Scheduler": name,
                        "Makespan (s)": vals.get("makespan"),
                        "Avg turnaround (s)": vals.get("avg_turnaround_time"),
                        "P99 turnaround (s)": vals.get("p99_turnaround_time"),
                        "Avg wait (s)": vals.get("avg_wait_time"),
                        "CPU util (%)": vals.get("avg_cpu_utilization", 0),
                        "RAM util (%)": vals.get("avg_ram_utilization", 0),
                        "SLA violations (%)": vals.get("sla_violation_rate", 0),
                    }
                )
        if bench_table:
            st.dataframe(
                pd.DataFrame(bench_table),
                width="stretch",
                hide_index=True,
                column_config={
                    c: st.column_config.NumberColumn(format="%.1f")
                    for c in bench_table[0]
                    if c != "Scheduler"
                },
            )

    img_path = os.path.join(project_root, "benchmark_comparison.png")
    if os.path.isfile(img_path):
        st.write("")
        with st.container(border=True):
            st.image(img_path, width="stretch")
