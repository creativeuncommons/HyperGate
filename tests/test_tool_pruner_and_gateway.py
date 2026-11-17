#!/usr/bin/env python3
"""
Unit tests for ToolPruner (Semantic & Keyword Top-K Discovery) and
ZeroMCPSupervisor Error Normalization in HyperGate.
"""
import unittest
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "hypergate-py"))

from hypergate import ToolPruner, ZeroMCPSupervisor, HyperGateRouter

class TestToolPrunerAndGateway(unittest.TestCase):

    def setUp(self):
        self.sample_tools = [
            {
                "name": "eval_math",
                "description": "Safely evaluates IEEE-754 arithmetic expressions (+, -, *, /)",
                "inputSchema": {"type": "object", "properties": {"expression": {"type": "string"}}}
            },
            {
                "name": "get_nba_game_odds",
                "description": "Fetches current Vegas point spreads and over/under odds for NBA basketball games",
                "inputSchema": {"type": "object", "properties": {"date": {"type": "string"}}}
            },
            {
                "name": "lookup_stock_ticker",
                "description": "Retrieves real-time NYSE/NASDAQ equity market ticker quotes",
                "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}}
            },
            {
                "name": "send_slack_message",
                "description": "Dispatches an alert message to a team Slack channel",
                "inputSchema": {"type": "object", "properties": {"channel": {"type": "string"}, "text": {"type": "string"}}}
            },
            {
                "name": "query_nfl_weather",
                "description": "Returns wind speed, temperature, and precipitation forecast for NFL stadium",
                "inputSchema": {"type": "object", "properties": {"stadium_id": {"type": "string"}}}
            },
            {
                "name": "system_diagnostics",
                "description": "Returns host platform, CPU, and memory utilization metrics",
                "always_include": True,
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

    def test_tool_pruning_relevance(self):
        # 1. Basketball query should select get_nba_game_odds and always-included tool
        prompt = "What is the spread and moneyline on the Lakers game tonight?"
        pruned = ToolPruner.prune_tools(self.sample_tools, prompt, top_k=2)

        # Must be capped at top_k (or pinned + candidates)
        self.assertLessEqual(len(pruned), 2)
        tool_names = [t["name"] for t in pruned]
        self.assertIn("system_diagnostics", tool_names)  # always_include
        self.assertIn("get_nba_game_odds", tool_names)

        # 2. Math query
        prompt2 = "Can you calculate 45 * 128 - 99?"
        pruned2 = ToolPruner.prune_tools(self.sample_tools, prompt2, top_k=2)
        tool_names2 = [t["name"] for t in pruned2]
        self.assertIn("eval_math", tool_names2)

    def test_tool_pruning_pinned_tools(self):
        prompt = "Random irrelevant chatter about pizza and coffee"
        pruned = ToolPruner.prune_tools(
            self.sample_tools,
            prompt,
            top_k=3,
            pin_tools=["lookup_stock_ticker"]
        )
        tool_names = [t["name"] for t in pruned]
        self.assertIn("lookup_stock_ticker", tool_names)
        self.assertIn("system_diagnostics", tool_names)

    def test_router_integration_with_tools_pruning(self):
        router = HyperGateRouter()
        req = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Check the spread on the Warriors game"}],
            "tools": self.sample_tools
        }
        headers = {"x-hypergate-tools-top-k": "3"}
        code, payload, resp_headers = router.handle_chat_completion(req, headers=headers)

        self.assertEqual(code, 200)
        self.assertIn("x-hypergate-tools-pruned", resp_headers)
        self.assertEqual(resp_headers["x-hypergate-tools-pruned"], "6->3")
        self.assertEqual(len(req["tools"]), 3)

    def test_supervisor_error_normalization(self):
        # Spawn a dummy supervisor running python that exits with code 1 immediately
        supervisor = ZeroMCPSupervisor(
            server_cmd=[sys.executable, "-c", "import sys; sys.stderr.write('Fatal crash\\n'); sys.exit(1)"],
            budget_limit_usd=1.0,
            max_turns=10
        )
        try:
            # Normalized response (default)
            resp = supervisor.send_rpc("ping", req_id=42, normalize_errors=True)
            self.assertEqual(resp["id"], 42)
            self.assertIn("error", resp)
            self.assertEqual(resp["error"]["code"], -32603)
            self.assertIn("terminated unexpectedly", resp["error"]["message"])
            self.assertEqual(resp["error"]["data"]["type"], "GATEWAY_ERROR")
        finally:
            supervisor.kill()

    def test_multi_turn_context_weighting(self):
        # Multi-turn conversation:
        # Turn 1: user mentions weather
        # Turn 2: assistant answers weather
        # Turn 3: user urgently asks for stock market ticker
        messages = [
            {"role": "user", "content": "What is the weather forecast in Chicago?"},
            {"role": "assistant", "content": "The weather forecast in Chicago is 55 degrees and windy."},
            {"role": "user", "content": "Quick, lookup the stock ticker quote for AAPL!"}
        ]
        # Pruning with top_k=2 should prioritize lookup_stock_ticker over query_nfl_weather
        pruned = ToolPruner.prune_tools(self.sample_tools, messages, top_k=2)
        tool_names = [t["name"] for t in pruned]
        self.assertIn("lookup_stock_ticker", tool_names)

    def test_virtual_toolpack_promotion(self):
        catalog = [
            {"name": "nba_odds", "description": "Fetches NBA spread lines", "toolpack": "sports_analytics", "inputSchema": {}},
            {"name": "nfl_stats", "description": "Fetches NFL player metrics", "toolpack": "sports_analytics", "inputSchema": {}},
            {"name": "mlb_pitchers", "description": "Fetches MLB strikeout records", "toolpack": "sports_analytics", "inputSchema": {}},
            {"name": "sql_select", "description": "Executes SELECT queries", "toolpack": "sql_queries", "inputSchema": {}},
            {"name": "sql_explain", "description": "Explains query planner", "toolpack": "sql_queries", "inputSchema": {}},
            {"name": "disk_cleanup", "description": "Prunes temp files", "toolpack": "system_ops", "inputSchema": {}}
        ]
        # Query that specifically triggers sports analytics
        prompt = "Analyze the spread and total points on the upcoming basketball game"
        pruned = ToolPruner.prune_tools(catalog, prompt, top_k=3, enable_toolpacks=True)
        pruned_names = [t["name"] for t in pruned]

        # The entire sports_analytics toolpack should be promoted!
        self.assertIn("nba_odds", pruned_names)
        self.assertIn("nfl_stats", pruned_names)
        self.assertIn("mlb_pitchers", pruned_names)

    def test_supervisor_transient_retry_success(self):
        import tempfile
        # Script that writes a counter to temp file. Fails on attempt 1, succeeds on attempt 2
        counter_file = os.path.join(tempfile.gettempdir(), f"mcp_retry_test_{os.getpid()}.txt")
        if os.path.exists(counter_file):
            os.unlink(counter_file)

        script = f"""
import sys, os, json
cnt_file = '{counter_file}'
count = 0
if os.path.exists(cnt_file):
    with open(cnt_file, 'r') as f:
        count = int(f.read().strip() or '0')
count += 1
with open(cnt_file, 'w') as f:
    f.write(str(count))

line = sys.stdin.readline()
if count == 1:
    # First attempt: simulate transient crash
    sys.exit(1)
else:
    # Second attempt: succeed
    req = json.loads(line)
    resp = {{"jsonrpc": "2.0", "id": req.get("id"), "result": {{"status": "recovered", "attempt": count}}}}
    sys.stdout.write(json.dumps(resp) + "\\n")
    sys.stdout.flush()
"""
        supervisor = ZeroMCPSupervisor(
            server_cmd=[sys.executable, "-c", script],
            budget_limit_usd=1.0,
            max_turns=10,
            max_retries=2,
            initial_backoff_sec=0.01
        )
        try:
            resp = supervisor.send_rpc("ping", req_id=99, normalize_errors=True)
            self.assertEqual(resp.get("id"), 99)
            self.assertIn("result", resp)
            self.assertEqual(resp["result"]["status"], "recovered")
            self.assertEqual(resp["result"]["attempt"], 2)
        finally:
            supervisor.kill()
            if os.path.exists(counter_file):
                os.unlink(counter_file)

if __name__ == "__main__":
    unittest.main()
