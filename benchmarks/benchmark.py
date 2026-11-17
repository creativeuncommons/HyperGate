#!/usr/bin/env python3
"""
HyperGate Benchmark Suite
=========================
Reproducible latency, throughput, and memory (RSS) benchmarking suite.
Measures:
  1. System-One Classification latency (p50, p90, p99)
  2. Deterministic Cost Accounting latency
  3. End-to-end Router Overhead (p50, p90, p99 target < 1.0ms)
  4. Memory RSS footprint
"""

import os
import sys
import time
import statistics
import resource
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(PROJECT_ROOT, "hypergate-py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_pricing.json")

if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from hypergate import HyperGateRouter, SystemOneClassifier, PricingEngine


def get_current_rss_kb() -> int:
    """Returns current process Resident Set Size (RSS) in KB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes; on Linux, in KB
    if sys.platform == "darwin":
        return int(usage.ru_maxrss / 1024)
    return int(usage.ru_maxrss)


def run_benchmarks(iterations: int = 5000):
    print(f"============================================================")
    print(f"⚡ HyperGate Micro-Benchmark Suite ({iterations:,} iterations)")
    print(f"============================================================\n")

    initial_rss = get_current_rss_kb()
    print(f"📊 Baseline Process Memory (RSS): {initial_rss:,} KB (~{initial_rss/1024:.2f} MB)")

    # 1. System-One Classifier Benchmark
    classifier = SystemOneClassifier()
    test_prompts = [
        [{"role": "user", "content": "ping"}],
        [{"role": "user", "content": "Calculate 42 * 1024 / 8"}],
        [{"role": "user", "content": "Write a python script for data validation and schema normalization"}],
        [{"role": "user", "content": "Derive the mathematical formal proof of circuit breaker convergence under race conditions"}],
    ]

    classifier_latencies_us = []
    for i in range(iterations):
        p = test_prompts[i % len(test_prompts)]
        t0 = time.perf_counter()
        classifier.classify(p)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
        classifier_latencies_us.append(elapsed_us)

    # 2. Pricing Engine Benchmark
    pricing = PricingEngine(CONFIG_PATH)
    pricing_latencies_us = []
    sample_text = "The quick brown fox jumps over the lazy dog. " * 10
    for i in range(iterations):
        t0 = time.perf_counter()
        tokens = pricing.estimate_tokens(sample_text)
        pricing.calculate_cost("gpt-4o", tokens, tokens)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
        pricing_latencies_us.append(elapsed_us)

    # 3. End-to-End Router Overhead Benchmark
    router = HyperGateRouter(CONFIG_PATH)
    router_latencies_us = []
    sample_req = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "ping"}],
        "trace_id": "bench_trace_01"
    }
    sample_headers = {
        "x-hypergate-trace-id": "bench_trace_01",
        "x-hypergate-budget-limit": "100.00"
    }

    for _ in range(iterations):
        t0 = time.perf_counter()
        status, data, headers = router.handle_chat_completion(sample_req, sample_headers)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000.0
        router_latencies_us.append(elapsed_us)

    final_rss = get_current_rss_kb()
    rss_delta = final_rss - initial_rss

    # Summarize Latency Metrics
    def summarize(latencies_us, name):
        sorted_l = sorted(latencies_us)
        n = len(sorted_l)
        p50 = sorted_l[int(n * 0.50)]
        p90 = sorted_l[int(n * 0.90)]
        p99 = sorted_l[int(n * 0.99)]
        p999 = sorted_l[int(n * 0.999)]
        avg = statistics.mean(sorted_l)
        throughput = 1_000_000.0 / avg if avg > 0 else 0
        return {
            "name": name,
            "avg_us": avg,
            "p50_us": p50,
            "p90_us": p90,
            "p99_us": p99,
            "p999_us": p999,
            "p50_ms": p50 / 1000.0,
            "p99_ms": p99 / 1000.0,
            "throughput_rps": throughput
        }

    c_summary = summarize(classifier_latencies_us, "System-One Heuristic Classifier")
    p_summary = summarize(pricing_latencies_us, "Deterministic Token Pricing Engine")
    r_summary = summarize(router_latencies_us, "End-to-End In-Process Gateway Dispatch")

    print("\n------------------------------------------------------------")
    print("📈 Benchmark Results Summary")
    print("------------------------------------------------------------")
    print(f"Component: {c_summary['name']}")
    print(f"  p50: {c_summary['p50_us']:.2f} µs | p90: {c_summary['p90_us']:.2f} µs | p99: {c_summary['p99_us']:.2f} µs | Throughput: {c_summary['throughput_rps']:,.0f} ops/sec")
    print(f"\nComponent: {p_summary['name']}")
    print(f"  p50: {p_summary['p50_us']:.2f} µs | p90: {p_summary['p90_us']:.2f} µs | p99: {p_summary['p99_us']:.2f} µs | Throughput: {p_summary['throughput_rps']:,.0f} ops/sec")
    print(f"\nComponent: {r_summary['name']}")
    print(f"  p50: {r_summary['p50_ms']:.4f} ms ({r_summary['p50_us']:.1f} µs)")
    print(f"  p90: {r_summary['p90_us']/1000.0:.4f} ms")
    print(f"  p99: {r_summary['p99_ms']:.4f} ms (< 1.0 ms target: {'✅ MET' if r_summary['p99_ms'] < 1.0 else '⚠️ SLOW'})")
    print(f"  Throughput: {r_summary['throughput_rps']:,.0f} req/sec")

    print(f"\n🧠 Final Process Memory: {final_rss:,} KB (Delta: +{rss_delta:,} KB)")
    print("============================================================\n")

    return {
        "rss_initial_kb": initial_rss,
        "rss_final_kb": final_rss,
        "classifier": c_summary,
        "pricing": p_summary,
        "router": r_summary
    }


if __name__ == "__main__":
    run_benchmarks(5000)
