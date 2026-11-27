#!/usr/bin/env python3
"""
Regression Tests for HyperGate Gateway Hardening, HTTP Parity & CLI
===================================================================
Verifies:
  1. HTTP network round-trip (`HyperGateServer` + `HyperGateClient(base_url=...)`)
     normalizes urllib capitalized headers, preserves `trace_id`, and trips 429 breaker.
  2. Malformed requests (`content: null`, multimodal list content, non-dict body,
     invalid numeric headers) return structured 400/200 JSON without tracebacks.
  3. Bounded `BudgetLedger` evicts idle/LRU sessions beyond `max_sessions`.
  4. Importing `hypergate` does not create CWD telemetry files; home dir beats CWD.
  5. `ZeroMCPSupervisor` skips notifications while waiting for matching request ID
     and drains noisy stderr without deadlocking.
  6. `audit.py --summary` subprocess returns rc=0 and prints resolved trace dir.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PY_DIR = PROJECT_ROOT / "hypergate-py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from hypergate import (  # noqa: E402
    BudgetLedger,
    CircuitBreakerError,
    HyperGateClient,
    HyperGateGatewayHandler,
    HyperGateRouter,
    HyperGateServer,
    TelemetrySink,
    ZeroMCPSupervisor,
)


class TestHyperGateHardening(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="hypergate_hardening_")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_http_round_trip_header_case_and_breaker_trip(self) -> None:
        srv = HyperGateServer(host="127.0.0.1", port=0)
        srv.router.telemetry.close()
        srv.router = HyperGateRouter(trace_dir=self.test_dir, async_batching=False)
        srv.start(blocking=False)
        port = srv._server.server_address[1]
        try:
            client = HyperGateClient(
                base_url=f"http://127.0.0.1:{port}",
                trace_id="http_session_01",
                budget_limit_usd=0.00001,
            )
            with self.assertRaises(CircuitBreakerError):
                for idx in range(15):
                    client.chat_completion(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": f"Turn {idx}: analyze architecture " * 4}],
                    )
            self.assertEqual(client.last_response_headers.get("x-hypergate-trace-id"), "http_session_01")
        finally:
            srv.stop()
            srv.router.close()

    def test_null_and_list_content_and_invalid_inputs_return_clean_json(self) -> None:
        router = HyperGateRouter(trace_dir=self.test_dir, async_batching=False)
        try:
            # 1. content: None and list-of-parts should succeed with 200
            st1, body1, _ = router.handle_chat_completion(
                {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "assistant", "content": None},
                        {"role": "user", "content": [{"type": "text", "text": "ping status"}]},
                    ],
                },
                headers={"X-HyperGate-Trace-Id": "trc_mm_01"},
            )
            self.assertEqual(st1, 200)
            self.assertIn("choices", body1)

            # 2. Non-dict body -> 400
            st2, body2, _ = router.handle_chat_completion(["bad"], headers={})  # type: ignore[arg-type]
            self.assertEqual(st2, 400)
            self.assertIn("error", body2)

            # 3. Non-numeric budget header -> 400
            st3, body3, _ = router.handle_chat_completion(
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-HyperGate-Budget-Limit": "not-a-number"},
            )
            self.assertEqual(st3, 400)
            self.assertEqual(body3["error"]["code"], "invalid_header")
        finally:
            router.close()

    def test_budget_ledger_bounds_memory(self) -> None:
        ledger = BudgetLedger(default_limit_usd=1.0, max_sessions=5, idle_ttl_sec=3600.0)
        for i in range(12):
            ledger.get_or_create(f"sess_{i}")
        self.assertLessEqual(len(ledger._sessions), 5)

    def test_import_no_cwd_side_effect_and_home_precedence(self) -> None:
        fake_home = Path(self.test_dir) / "home"
        fake_cwd = Path(self.test_dir) / "cwd"
        fake_home.mkdir()
        fake_cwd.mkdir()
        env = dict(os.environ)
        env.pop("HYPERGATE_TRACE_DIR", None)
        env["HOME"] = str(fake_home)
        env["PYTHONPATH"] = str(PY_DIR)
        res = subprocess.run(
            [sys.executable, "-c", "import hypergate; print('OK')"],
            cwd=str(fake_cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse((fake_cwd / ".hypergate").exists())

        old_home = os.environ.get("HOME")
        old_cwd = os.getcwd()
        old_td = os.environ.pop("HYPERGATE_TRACE_DIR", None)
        try:
            os.environ["HOME"] = str(fake_home)
            os.chdir(str(fake_cwd))
            sink = TelemetrySink(async_batching=False)
            try:
                self.assertEqual(
                    Path(sink.base_dir).resolve(),
                    (fake_home / ".hypergate" / "traces").resolve(),
                )
            finally:
                sink.close()
        finally:
            os.chdir(old_cwd)
            if old_home is not None:
                os.environ["HOME"] = old_home
            if old_td is not None:
                os.environ["HYPERGATE_TRACE_DIR"] = old_td

    def test_supervisor_skips_notifications_and_drains_stderr(self) -> None:
        script = (
            "import sys, json\n"
            "req = json.loads(sys.stdin.readline())\n"
            "sys.stderr.write('noisy stderr line\\n' * 3000)\n"
            "sys.stderr.flush()\n"
            "sys.stdout.write(json.dumps({'jsonrpc':'2.0','method':'notifications/progress','params':{'p':1}}) + '\\n')\n"
            "sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'ok':True}}) + '\\n')\n"
            "sys.stdout.flush()\n"
        )
        sup = ZeroMCPSupervisor(server_cmd=[sys.executable, "-c", script], rpc_timeout=5.0)
        try:
            resp = sup.send_rpc("tools/call", {"name": "echo"}, req_id=77)
            self.assertEqual(resp.get("id"), 77)
            self.assertEqual(resp.get("result"), {"ok": True})
        finally:
            sup.kill()

    def test_audit_cli_subprocess(self) -> None:
        res = subprocess.run(
            [sys.executable, str(PY_DIR / "audit.py"), "--trace-dir", self.test_dir, "--summary"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn(self.test_dir, res.stdout)

    def test_supervisor_timeout_kills_stale_child_no_cross_talk(self) -> None:
        marker = str(Path(self.test_dir) / "first_spawn.flag")
        script = (
            "import sys, json, os, time\n"
            f"marker = {marker!r}\n"
            "is_first = not os.path.exists(marker)\n"
            "if is_first:\n"
            "    open(marker, 'w').write('1')\n"
            "line = sys.stdin.readline()\n"
            "if line:\n"
            "    req = json.loads(line)\n"
            "    if is_first:\n"
            "        time.sleep(0.35)\n"
            "        sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'reply':'STALE_REPLY'}}) + '\\n')\n"
            "    else:\n"
            "        sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'reply':'FRESH_REPLY'}}) + '\\n')\n"
            "    sys.stdout.flush()\n"
        )
        sup = ZeroMCPSupervisor(
            server_cmd=[sys.executable, "-c", script],
            max_retries=0,
            rpc_timeout=0.15,
        )
        try:
            r1 = sup.send_rpc("tools/call", {"name": "slow"}, req_id=1)
            self.assertIn("error", r1)
            self.assertIsNone(sup.process)
            time.sleep(0.4)
            r2 = sup.send_rpc("tools/call", {"name": "fast"}, req_id=1)
            self.assertEqual(r2.get("id"), 1)
            self.assertEqual((r2.get("result") or {}).get("reply"), "FRESH_REPLY")
        finally:
            sup.kill()


if __name__ == "__main__":
    unittest.main()
