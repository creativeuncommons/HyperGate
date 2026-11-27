#!/usr/bin/env python3
"""
HyperGate Telemetry & Audit CLI Tool
====================================
Inspects persistent agent execution traces, cost breakdown, latency curves,
and circuit-breaker trip events.

Usage:
  python3 hypergate-py/audit.py --list
  python3 hypergate-py/audit.py --trace <trace_id>
  python3 hypergate-py/audit.py --summary
"""

import os
import sys
import argparse
from typing import Optional

# Allow relative import or direct execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from hypergate import TelemetrySink


def format_currency(val: float) -> str:
    return f"${val:.6f}"


def print_trace_details(sink: TelemetrySink, trace_id: str):
    records = sink.get_trace_records(trace_id)
    if not records:
        print(f"❌ No records found for trace ID: {trace_id}")
        return

    print(f"\n================================================================================")
    print(f"📊 HyperGate Agent Execution Trace: {trace_id}")
    print(f"================================================================================")
    print(f"Total Turns: {len(records)} | Initialized: {records[0].get('timestamp_iso')}")
    print(f"{'Turn':<6} | {'Tier':<16} | {'Model':<16} | {'Tokens':<8} | {'Cost':<12} | {'Rolling':<12} | {'Overhead':<10} | {'Status'}")
    print("-" * 88)

    for r in records:
        status_str = "✅ 200 OK" if r.get("status_code") == 200 else f"🛑 {r.get('status_code')} TRIPPED"
        print(
            f"{r.get('turn'):<6} | "
            f"{r.get('tier'):<16} | "
            f"{r.get('model')[:16]:<16} | "
            f"{r.get('total_tokens', 0):<8} | "
            f"{format_currency(r.get('turn_cost_usd', 0.0)):<12} | "
            f"{format_currency(r.get('rolling_cost_usd', 0.0)):<12} | "
            f"{r.get('overhead_ms', 0.0):.3f} ms   | "
            f"{status_str}"
        )
        if r.get("error"):
            print(f"   ⚠️ Trip Reason: {r.get('error')}")

    print("=" * 88 + "\n")


def print_traces_list(sink: TelemetrySink, limit: int = 50):
    traces = sink.list_traces(limit=limit)
    if not traces:
        print("ℹ️ No agent traces recorded yet. Specify --trace-dir <dir> or set $HYPERGATE_TRACE_DIR.")
        return

    print(f"\n================================================================================")
    print(f"📋 Recent Agent Execution Traces (Showing last {len(traces)})")
    print(f"================================================================================")
    print(f"{'Trace ID':<24} | {'Turns':<6} | {'Total Cost':<12} | {'Tiers Used':<22} | {'Circuit'}")
    print("-" * 80)

    for t in traces:
        circuit_str = "🛑 TRIPPED" if t.get("circuit_tripped") else "✅ Normal"
        print(
            f"{t.get('trace_id'):<24} | "
            f"{t.get('turns'):<6} | "
            f"{format_currency(t.get('total_cost_usd', 0.0)):<12} | "
            f"{t.get('tiers_used', 'N/A')[:22]:<22} | "
            f"{circuit_str}"
        )
    print("=" * 80 + "\n")


def print_summary(sink: TelemetrySink):
    s = sink.get_summary()
    print(f"\n================================================================================")
    print(f"📈 HyperGate Gateway Telemetry & Analytics Summary")
    print(f"================================================================================")
    print(f"Total Unique Traces:        {s.get('total_traces', 0):,}")
    print(f"Total Agent Turns:          {s.get('total_turns', 0):,}")
    print(f"Total Tokens Processed:     {s.get('total_tokens_processed', 0):,}")
    print(f"Total Expenditure (USD):    {format_currency(s.get('total_spent_usd', 0.0))}")
    print(f"Average Gateway Overhead:   {s.get('avg_overhead_ms', 0.0):.3f} ms")
    print(f"Circuit-Breaker Trips:      {s.get('total_circuits_tripped', 0)}")
    print("\nRouting Tier Breakdown:")
    print(f"  {'Tier Name':<20} | {'Turns Handled':<14} | {'Dollar Spend'}")
    print("  " + "-" * 50)
    for tier, data in s.get("tier_breakdown", {}).items():
        print(f"  {tier:<20} | {data.get('turns', 0):<14} | {format_currency(data.get('spent', 0.0))}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HyperGate Telemetry & Audit CLI")
    parser.add_argument("--trace", "-t", help="Trace ID to inspect turn-by-turn")
    parser.add_argument("--list", "-l", action="store_true", help="List recent agent execution traces")
    parser.add_argument("--summary", "-s", action="store_true", help="Display aggregate telemetry summary")
    parser.add_argument("--trace-dir", help="Custom trace directory path (default: ~/.hypergate/traces)")
    parser.add_argument("--limit", type=int, default=50, help="Max traces to list")

    args = parser.parse_args()
    sink = TelemetrySink(base_dir=args.trace_dir)
    print(f"📂 Resolved Trace Directory: {sink.base_dir}")

    if args.trace:
        print_trace_details(sink, args.trace)
    elif args.list:
        print_traces_list(sink, limit=args.limit)
    elif args.summary:
        print_summary(sink)
    else:
        # Default to summary + list
        print_summary(sink)
        print_traces_list(sink, limit=10)


if __name__ == "__main__":
    main()
