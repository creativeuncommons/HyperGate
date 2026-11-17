#!/usr/bin/env python3
"""
HyperGate Automated Stress & Circuit-Breaker Verification Suite
===============================================================
Simulates runaway agent tool loops (50 rapid iterations) and asserts
that HyperGate halts execution precisely when cost exceeds the budget ceiling.
"""

import os
import sys
import unittest
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(PROJECT_ROOT, "hypergate-py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_pricing.json")

if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from hypergate import HyperGateRouter, HyperGateClient, CircuitBreakerError


class TestCircuitBreakerKillSwitch(unittest.TestCase):
    def test_runaway_loop_hard_stop(self):
        """
        Runs 50 iterations with a tight budget ceiling ($0.005000).
        Asserts hard stop precisely at iteration N with HTTP 429 and CircuitBreakerError.
        """
        budget_limit = 0.005000  # $0.005 limit
        trace_id = "test_runaway_loop_trace_01"
        router = HyperGateRouter(CONFIG_PATH)
        client = HyperGateClient(
            router=router,
            trace_id=trace_id,
            budget_limit_usd=budget_limit
        )

        completed_turns = 0
        circuit_tripped = False
        tripped_error = None

        # Each call will cost ~$0.00125 (gpt-4o with prompt/comp)
        for i in range(1, 51):
            try:
                resp = client.chat_completion(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": f"Loop iteration {i}: analyze current system metrics"}]
                )
                completed_turns += 1
                self.assertIn("choices", resp)
                # Verify telemetry headers
                headers = client.last_response_headers
                self.assertEqual(headers.get("x-hypergate-trace-id"), trace_id)
                self.assertEqual(headers.get("x-hypergate-circuit-tripped"), "false")
            except CircuitBreakerError as e:
                circuit_tripped = True
                tripped_error = e
                break

        self.assertTrue(circuit_tripped, "Circuit breaker should have tripped before 50 iterations!")
        self.assertIsNotNone(tripped_error)
        self.assertLessEqual(completed_turns, 20, f"Should have halted within 20 turns; ran {completed_turns}")

        # Assert CircuitBreakerError schema & attributes
        self.assertEqual(tripped_error.trace_id, trace_id)
        self.assertEqual(tripped_error.limit_usd, budget_limit)
        self.assertAlmostEqual(tripped_error.rolling_cost_usd, budget_limit, delta=0.001)
        self.assertIn("Circuit Breaker Tripped", str(tripped_error))

        # Assert custom headers on tripped response
        headers = client.last_response_headers
        self.assertEqual(headers.get("x-hypergate-circuit-tripped"), "true")
        self.assertEqual(headers.get("x-hypergate-cost-usd"), "0.000000")

        # Assert that zero subsequent calls are allowed
        with self.assertRaises(CircuitBreakerError) as ctx:
            client.chat_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Attempt call after circuit tripped"}]
            )
        self.assertEqual(ctx.exception.trace_id, trace_id)

        # Assert persistent telemetry sink captured the trace
        records = router.telemetry.get_trace_records(trace_id)
        self.assertGreater(len(records), 0, "Telemetry records must be persisted")
        self.assertEqual(records[0]["trace_id"], trace_id)
        # Verify that the final record captured the tripped state (429)
        last_rec = records[-1]
        self.assertEqual(last_rec["status_code"], 429)
        self.assertEqual(last_rec["circuit_tripped"], 1)
        self.assertIn("exceeded", last_rec["error"])

    def test_system_one_fast_path_headers(self):
        """Tests telemetry headers on System-One fast path requests."""
        trace_id = "test_fast_path_trace_02"
        router = HyperGateRouter(CONFIG_PATH)
        client = HyperGateClient(
            router=router,
            trace_id=trace_id,
            budget_limit_usd=0.50
        )

        resp = client.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}]
        )
        self.assertIn("choices", resp)
        headers = client.last_response_headers
        self.assertEqual(headers.get("x-hypergate-trace-id"), trace_id)
        self.assertEqual(headers.get("x-hypergate-routing-tier"), "TIER_1_FAST")
        self.assertEqual(headers.get("x-hypergate-circuit-tripped"), "false")
        overhead_ms = float(headers.get("x-hypergate-latency-overhead-ms", "999.0"))
        self.assertLess(overhead_ms, 20.0, "Overhead should be under 20ms")


if __name__ == "__main__":
    unittest.main()
