#!/usr/bin/env python3
"""
HyperGate Deterministic Pricing & Ledger Accounting Tests
==========================================================
Verifies zero-drift token and cost accounting across all supported model families.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(PROJECT_ROOT, "hypergate-py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "model_pricing.json")

if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from hypergate import PricingEngine, BudgetLedger, SystemOneClassifier


class TestCostAccounting(unittest.TestCase):
    def setUp(self):
        self.pricing = PricingEngine(CONFIG_PATH)
        self.ledger = BudgetLedger(default_limit_usd=0.50, default_max_turns=30)
        self.classifier = SystemOneClassifier()

    def test_token_estimation(self):
        # 4 chars per token heuristic
        text_40_chars = "1234567890" * 4  # 40 chars
        tokens = self.pricing.estimate_tokens(text_40_chars)
        self.assertEqual(tokens, 10)

        empty_tokens = self.pricing.estimate_tokens("")
        self.assertEqual(empty_tokens, 0)

    def test_deterministic_pricing_calculations(self):
        # gpt-4o: $2.50 prompt, $10.00 completion per 1M
        # 1,000 prompt tokens = $0.0025, 1,000 completion tokens = $0.0100 -> total $0.0125
        cost_gpt4o = self.pricing.calculate_cost("gpt-4o", 1000, 1000)
        self.assertAlmostEqual(cost_gpt4o, 0.0125, places=6)

        # gpt-4o-mini: $0.15 prompt, $0.60 completion per 1M
        # 10,000 prompt tokens = $0.0015, 10,000 completion tokens = $0.0060 -> total $0.0075
        cost_mini = self.pricing.calculate_cost("gpt-4o-mini", 10000, 10000)
        self.assertAlmostEqual(cost_mini, 0.0075, places=6)

        # gemini-2.0-flash-lite: $0.075 prompt, $0.30 completion per 1M
        cost_gemini = self.pricing.calculate_cost("gemini-2.0-flash-lite", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost_gemini, 0.375, places=4)

        # o3-mini: $1.10 prompt, $4.40 completion per 1M
        cost_o3 = self.pricing.calculate_cost("o3-mini", 500_000, 500_000)
        self.assertAlmostEqual(cost_o3, 2.75, places=4)

        # Cache / Local SLM: $0.00
        cost_cache = self.pricing.calculate_cost("cache", 50000, 50000)
        self.assertEqual(cost_cache, 0.0)

    def test_tier_resolution(self):
        self.assertEqual(self.pricing.get_tier("cache"), "TIER_0_CACHE")
        self.assertEqual(self.pricing.get_tier("gpt-4o-mini"), "TIER_1_FAST")
        self.assertEqual(self.pricing.get_tier("gemini-2.0-flash-lite"), "TIER_1_FAST")
        self.assertEqual(self.pricing.get_tier("gpt-4o"), "TIER_2_BALANCED")
        self.assertEqual(self.pricing.get_tier("claude-3-5-sonnet"), "TIER_2_BALANCED")
        self.assertEqual(self.pricing.get_tier("o3-mini"), "TIER_3_REASONING")
        self.assertEqual(self.pricing.get_tier("claude-3-opus"), "TIER_3_REASONING")

    def test_budget_ledger_accumulation(self):
        trace_id = "test_trc_acc_01"
        sess = self.ledger.get_or_create(trace_id, budget_limit=0.10)
        self.assertEqual(sess.rolling_cost_usd, 0.0)
        self.assertFalse(sess.circuit_tripped)

        # Record 1st call: $0.04
        rolling, tripped = self.ledger.record_cost(trace_id, 0.04)
        self.assertAlmostEqual(rolling, 0.04, places=6)
        self.assertFalse(tripped)

        # Record 2nd call: $0.04 -> total $0.08
        rolling, tripped = self.ledger.record_cost(trace_id, 0.04)
        self.assertAlmostEqual(rolling, 0.08, places=6)
        self.assertFalse(tripped)

        # Record 3rd call: $0.03 -> total $0.11 (exceeds $0.10 limit)
        rolling, tripped = self.ledger.record_cost(trace_id, 0.03)
        self.assertAlmostEqual(rolling, 0.11, places=6)
        self.assertTrue(tripped)

        # Preflight check should now fail
        allowed, reason = self.ledger.check_preflight(trace_id, 0.001)
        self.assertFalse(allowed)
        self.assertIn("previously exceeded", reason)

    def test_system_one_classifier(self):
        # Heuristic fast path match
        tier, reason = self.classifier.classify([{"role": "user", "content": "ping"}])
        self.assertEqual(tier, 1)

        # Arithmetic match
        tier, reason = self.classifier.classify([{"role": "user", "content": "12 + 45 / 3"}])
        self.assertEqual(tier, 1)

        # Frontier reasoning escalation
        tier, reason = self.classifier.classify([{"role": "user", "content": "Derive the mathematical formal proof of convergence"}])
        self.assertEqual(tier, 3)

        # Balanced workhorse default
        tier, reason = self.classifier.classify([{"role": "user", "content": "Write a python script to parse CSV files and generate a chart"}])
        self.assertEqual(tier, 2)


if __name__ == "__main__":
    unittest.main()
