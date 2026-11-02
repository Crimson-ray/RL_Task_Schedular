"""
Unit tests for synthetic trace generation and task models.
"""
import os
import tempfile
from src.workload.task import Task
from src.workload.trace_generator import TraceGenerator
from src.config import WorkloadConfig

def test_task_sla_metrics():
    # Met deadline
    t1 = Task(0, 0.0, 4.0, 8.0, 10.0, 1, 20.0)
    t1.start_time = 2.0
    t1.finish_time = 12.0
    assert t1.is_sla_violated is False
    assert t1.tardiness == 0.0
    assert t1.turnaround_time == 12.0
    assert t1.wait_time == 2.0

    # Violated deadline
    t2 = Task(1, 0.0, 4.0, 8.0, 15.0, 2, 10.0)
    t2.start_time = 0.0
    t2.finish_time = 15.0
    assert t2.is_sla_violated is True
    assert t2.tardiness == 5.0

def test_trace_generation_determinism():
    gen = TraceGenerator(WorkloadConfig(num_tasks=50))
    trace_a = gen.generate_trace(seed=42)
    trace_b = gen.generate_trace(seed=42)

    assert len(trace_a) == 50
    assert len(trace_b) == 50
    for ta, tb in zip(trace_a, trace_b):
        assert ta.arrival_time == tb.arrival_time
        assert ta.cpu_req == tb.cpu_req
        assert ta.ram_req == tb.ram_req
        assert ta.duration == tb.duration
        assert ta.deadline == tb.deadline

def test_csv_serialization():
    gen = TraceGenerator()
    trace = gen.generate_trace(num_tasks=10, seed=123)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        temp_path = f.name

    try:
        gen.save_to_csv(trace, temp_path)
        loaded = gen.load_from_csv(temp_path)
        assert len(loaded) == 10
        assert loaded[0].task_id == trace[0].task_id
        assert abs(loaded[0].arrival_time - trace[0].arrival_time) < 1e-4
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
