"""
Unit tests for Cluster and Node Environment.
"""
import numpy as np
import pytest
from src.environment.node import ComputeNode
from src.environment.cluster import Cluster
from src.environment.simulator import ClusterSimulator
from src.workload.task import Task
from src.config import ClusterConfig

def test_node_allocation_and_capacity():
    node = ComputeNode(node_id=0, cpu_capacity=16.0, ram_capacity=32.0)
    assert node.available_cpu == 16.0
    assert node.available_ram == 32.0
    assert node.cpu_utilization == 0.0

    task = Task(task_id=1, arrival_time=0.0, cpu_req=8.0, ram_req=16.0, duration=10.0, priority=1, deadline=20.0)
    assert node.can_accommodate(task.cpu_req, task.ram_req) is True

    allocated = node.allocate(task, current_time=0.0)
    assert allocated is True
    assert node.available_cpu == 8.0
    assert node.available_ram == 16.0
    assert node.cpu_utilization == 0.5
    assert node.ram_utilization == 0.5

    # Cannot exceed remaining capacity
    assert node.can_accommodate(10.0, 10.0) is False

    # Fast forward time to release task
    freed = node.release_completed(current_time=10.5)
    assert len(freed) == 1
    assert freed[0].task_id == 1
    assert node.available_cpu == 16.0
    assert node.available_ram == 32.0

def test_cluster_action_mask():
    config = ClusterConfig(heterogeneous_specs=[
        (8.0, 16.0, "small"),
        (32.0, 64.0, "large")
    ])
    cluster = Cluster(config)
    assert cluster.num_nodes == 2

    # Task requiring 16 CPU: node 1 is feasible
    task = Task(task_id=1, arrival_time=0.0, cpu_req=16.0, ram_req=8.0, duration=5.0, priority=0, deadline=15.0)
    mask = cluster.get_action_mask(task)
    assert mask[0] == False
    assert mask[1] == True

    # Task requiring 64 CPU: cannot fit on either node right now, so Defer (Action 2) is True!
    huge_task = Task(task_id=2, arrival_time=0.0, cpu_req=64.0, ram_req=8.0, duration=5.0, priority=0, deadline=15.0)
    huge_mask = cluster.get_action_mask(huge_task)
    assert huge_mask[0] == False
    assert huge_mask[1] == False
    assert huge_mask[2] == True # Defer action is mandatory when saturated

def test_simulator_step_interface():
    sim = ClusterSimulator()
    tasks = [
        Task(task_id=0, arrival_time=0.0, cpu_req=4.0, ram_req=8.0, duration=5.0, priority=1, deadline=10.0),
        Task(task_id=1, arrival_time=2.0, cpu_req=4.0, ram_req=8.0, duration=5.0, priority=1, deadline=12.0)
    ]
    state, mask, first_task = sim.reset(tasks)
    assert state is not None
    assert mask.any()
    assert first_task.task_id == 0

    next_state, next_mask, reward, done, info = sim.step(0)
    assert not done
    assert isinstance(reward, float)

    # Step second task
    next_state, next_mask, reward, done, info = sim.step(0)
    assert done
    metrics = sim.get_metrics([0.001, 0.001])
    assert metrics.total_tasks == 2
    assert metrics.makespan > 0
