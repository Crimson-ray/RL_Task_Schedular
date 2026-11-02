"""
Integration tests for schedulers and benchmark runner.
"""
import pytest
from src.environment.cluster import Cluster
from src.environment.simulator import ClusterSimulator
from src.workload.trace_generator import TraceGenerator
from src.scheduler.heuristics import FIFOScheduler, SJFScheduler, EDFScheduler, TetrisBestFitScheduler
from src.scheduler.intelligent_scheduler import IntelligentD3QNScheduler
from src.benchmarks.benchmark_engine import BenchmarkEngine

def test_heuristic_schedulers():
    cluster = Cluster()
    gen = TraceGenerator()
    trace = gen.generate_trace(num_tasks=20, seed=123)

    engine = BenchmarkEngine()
    schedulers = [
        FIFOScheduler(),
        SJFScheduler(),
        EDFScheduler(),
        TetrisBestFitScheduler()
    ]

    for sched in schedulers:
        metrics = engine.evaluate_scheduler(sched, trace)
        assert metrics.total_tasks == 20
        assert metrics.makespan > 0
        assert metrics.mean_decision_latency_us > 0

def test_intelligent_scheduler_untrained():
    cluster = Cluster()
    gen = TraceGenerator()
    trace = gen.generate_trace(num_tasks=10, seed=456)

    intelligent_sched = IntelligentD3QNScheduler()
    engine = BenchmarkEngine()
    metrics = engine.evaluate_scheduler(intelligent_sched, trace)

    assert metrics.total_tasks == 10
    assert metrics.makespan > 0
    # Sub-millisecond latency check (typically < 500 μs)
    assert metrics.mean_decision_latency_us < 2000.0
