#!/usr/bin/env python3
"""
HyperGate Telemetry Concurrency, Async Batching & Sharding Verification
=======================================================================
Validates:
1. Single-writer async batching stats (`sqlite_errors == 0`, `dropped_records == 0`).
2. Lock-free concurrent `?mode=ro` readers alongside heavy multi-threaded writes.
3. PID sharding (`shard_by_pid=True`) and cross-shard aggregation in `get_trace_records`,
   `list_traces`, and `get_summary`.
4. `GatewayTelemetryStore` Protocol compliance and custom store injection.
"""

import os
import sys
import shutil
import sqlite3
import tempfile
import threading
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(PROJECT_ROOT, "hypergate-py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_pricing.json")

if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from hypergate import (
    TelemetryRecord,
    TelemetrySink,
    GatewayTelemetryStore,
    HyperGateRouter,
)


class TestHyperGateTelemetryConcurrency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="hg_concurrency_")
        self._sinks = []

    def tearDown(self):
        for s in self._sinks:
            try:
                s.close()
            except Exception:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_async_batching_and_stats(self):
        sink = TelemetrySink(base_dir=self.temp_dir, batch_size=32, flush_interval_ms=25.0)
        self._sinks.append(sink)

        for i in range(60):
            sink.record_turn(
                TelemetryRecord(
                    trace_id="trc_batch_01",
                    turn=i + 1,
                    model="gpt-4o-mini",
                    tier="TIER_1_FAST",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    turn_cost_usd=0.0001,
                    rolling_cost_usd=0.0001 * (i + 1),
                    budget_ceiling_usd=1.0,
                    overhead_ms=0.25,
                    circuit_tripped=False,
                    status_code=200,
                )
            )

        self.assertTrue(sink.flush(timeout=5.0))
        self.assertEqual(sink.stats["enqueued_records"], 60)
        self.assertEqual(sink.stats["committed_records"], 60)
        self.assertEqual(sink.stats["dropped_records"], 0)
        self.assertEqual(sink.stats["sqlite_errors"], 0)
        self.assertGreaterEqual(sink.stats["batches_committed"], 1)
        self.assertLess(sink.stats["batches_committed"], 60)

        records = sink.get_trace_records("trc_batch_01")
        self.assertEqual(len(records), 60)
        summary = sink.get_summary()
        self.assertEqual(summary["total_turns"], 60)
        self.assertEqual(summary["total_traces"], 1)

    def test_02_concurrent_lock_free_readers_and_writers(self):
        sink = TelemetrySink(base_dir=self.temp_dir, batch_size=64, flush_interval_ms=10.0)
        self._sinks.append(sink)
        errors = []
        stop_readers = threading.Event()

        def reader():
            try:
                while not stop_readers.is_set():
                    sink.list_traces(limit=20)
                    sink.get_summary()
                    sink.get_trace_records("trc_conc_rw")
            except Exception as e:
                errors.append(f"Reader error: {e}")

        def writer(wid: int):
            try:
                for i in range(25):
                    sink.record_turn(
                        TelemetryRecord(
                            trace_id="trc_conc_rw",
                            turn=(wid * 25) + i + 1,
                            model="gpt-4o",
                            tier="TIER_2_BALANCED",
                            prompt_tokens=20,
                            completion_tokens=30,
                            total_tokens=50,
                            turn_cost_usd=0.001,
                            rolling_cost_usd=0.001 * (i + 1),
                            budget_ceiling_usd=5.0,
                            overhead_ms=0.4,
                            circuit_tripped=False,
                            status_code=200,
                        )
                    )
            except Exception as e:
                errors.append(f"Writer {wid} error: {e}")

        r_thread = threading.Thread(target=reader)
        w_threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
        r_thread.start()
        for wt in w_threads:
            wt.start()
        for wt in w_threads:
            wt.join(timeout=5.0)
        stop_readers.set()
        r_thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(sink.get_trace_records("trc_conc_rw")), 100)

    def test_03_pid_sharding_and_cross_shard_aggregation(self):
        sink = TelemetrySink(base_dir=self.temp_dir, shard_by_pid=True)
        self._sinks.append(sink)

        expected_db = os.path.join(self.temp_dir, f"hypergate_telemetry_pid_{os.getpid()}.db")
        self.assertEqual(os.path.abspath(sink.db_path), os.path.abspath(expected_db))

        sink.record_turn(
            TelemetryRecord(
                trace_id="trc_shared_x",
                turn=1,
                model="gpt-4o-mini",
                tier="TIER_1_FAST",
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                turn_cost_usd=0.001,
                rolling_cost_usd=0.001,
                budget_ceiling_usd=1.0,
                overhead_ms=1.0,
                circuit_tripped=False,
                status_code=200,
            )
        )
        sink.flush(timeout=5.0)
        self.assertTrue(os.path.exists(expected_db))

        # Inject a second simulated PID shard DB sharing trc_shared_x plus a second trace
        peer_db = os.path.join(self.temp_dir, "hypergate_telemetry_pid_999999.db")
        conn = sqlite3.connect(peer_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hypergate_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL, turn INTEGER NOT NULL, timestamp_iso TEXT NOT NULL,
                model TEXT NOT NULL, tier TEXT NOT NULL,
                prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                turn_cost_usd REAL, rolling_cost_usd REAL, budget_ceiling_usd REAL,
                overhead_ms REAL, circuit_tripped INTEGER, status_code INTEGER, error TEXT
            );
        """)
        conn.execute(
            """
            INSERT INTO hypergate_turns (
                trace_id, turn, timestamp_iso, model, tier,
                prompt_tokens, completion_tokens, total_tokens,
                turn_cost_usd, rolling_cost_usd, budget_ceiling_usd,
                overhead_ms, circuit_tripped, status_code, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "trc_shared_x", 2, "2026-11-22T12:00:01+00:00", "gpt-4o", "TIER_2_BALANCED",
                30, 30, 60, 0.003, 0.004, 1.0, 3.0, 0, 200, None
            ),
        )
        conn.commit()
        conn.close()

        recs = sink.get_trace_records("trc_shared_x")
        self.assertEqual(len(recs), 2)

        traces = sink.list_traces(limit=10)
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["turns"], 2)

        summary = sink.get_summary()
        self.assertEqual(summary["total_traces"], 1)
        self.assertEqual(summary["total_turns"], 2)
        self.assertAlmostEqual(summary["total_spent_usd"], 0.004, places=6)
        self.assertAlmostEqual(summary["avg_overhead_ms"], 2.0, places=3)
        self.assertEqual(summary["tier_breakdown"]["TIER_1_FAST"]["turns"], 1)
        self.assertEqual(summary["tier_breakdown"]["TIER_2_BALANCED"]["turns"], 1)

    def test_04_protocol_compliance_and_router_injection(self):
        sink = TelemetrySink(base_dir=self.temp_dir)
        self._sinks.append(sink)
        self.assertIsInstance(sink, GatewayTelemetryStore)

        class CustomGatewaySink:
            def __init__(self):
                self.records = []
            def record_turn(self, record):
                self.records.append(record)
            def get_trace_records(self, trace_id: str):
                return []
            def list_traces(self, limit: int = 50):
                return []
            def get_summary(self):
                return {"total_turns": len(self.records)}
            def flush(self, timeout: float = 5.0) -> bool:
                return True
            def close(self):
                pass

        custom = CustomGatewaySink()
        self.assertIsInstance(custom, GatewayTelemetryStore)

        router = HyperGateRouter(CONFIG_PATH, telemetry_sink=custom)
        status, _, _ = router.handle_chat_completion(
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
            headers={"x-hypergate-trace-id": "trc_custom_sink"},
        )
        self.assertEqual(status, 200)
        router.flush()
        router.close()
        self.assertEqual(custom.get_summary()["total_turns"], 1)


if __name__ == "__main__":
    unittest.main()
