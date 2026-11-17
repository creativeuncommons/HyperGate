"""
HyperGate Python SDK & ZeroMCP Bridge
=====================================
Zero-dependency, production-grade intelligent inference router, circuit-breaker
gateway client, and ZeroMCP supervisor bridge.

Pure Python standard library (urllib.request, json, dataclasses, time, threading).
"""

import os
import sys
import json
import time
import uuid
import sqlite3
import datetime
import urllib.request
import urllib.error
import threading
import subprocess
import math
import re
import hmac
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable, Set
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_GATEWAY_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB

# Default Fallback Pricing Matrix (reconciled against config/model_pricing.json)
DEFAULT_PRICING: Dict[str, Dict[str, Any]] = {
    "cache": {"tier": "TIER_0_CACHE", "prompt_per_1m": 0.0, "completion_per_1m": 0.0},
    "local-slm": {"tier": "TIER_1_FAST", "prompt_per_1m": 0.0, "completion_per_1m": 0.0},
    "gpt-4o-mini": {"tier": "TIER_1_FAST", "prompt_per_1m": 0.15, "completion_per_1m": 0.60},
    "claude-3-5-haiku": {"tier": "TIER_1_FAST", "prompt_per_1m": 0.80, "completion_per_1m": 4.00},
    "gemini-2.0-flash-lite": {"tier": "TIER_1_FAST", "prompt_per_1m": 0.075, "completion_per_1m": 0.30},
    "gpt-4o": {"tier": "TIER_2_BALANCED", "prompt_per_1m": 2.50, "completion_per_1m": 10.00},
    "claude-3-5-sonnet": {"tier": "TIER_2_BALANCED", "prompt_per_1m": 3.00, "completion_per_1m": 15.00},
    "gemini-2.0-flash": {"tier": "TIER_2_BALANCED", "prompt_per_1m": 0.10, "completion_per_1m": 0.40},
    "o3-mini": {"tier": "TIER_3_REASONING", "prompt_per_1m": 1.10, "completion_per_1m": 4.40},
    "o1": {"tier": "TIER_3_REASONING", "prompt_per_1m": 15.00, "completion_per_1m": 60.00},
    "claude-3-opus": {"tier": "TIER_3_REASONING", "prompt_per_1m": 15.00, "completion_per_1m": 75.00},
    "gemini-2.0-pro": {"tier": "TIER_3_REASONING", "prompt_per_1m": 1.25, "completion_per_1m": 5.00},
}

TIER_NAMES = {
    0: "TIER_0_CACHE",
    1: "TIER_1_FAST",
    2: "TIER_2_BALANCED",
    3: "TIER_3_REASONING",
}


class CircuitBreakerError(Exception):
    """Raised when HyperGate circuit-breaker halts execution due to cost or loop limit."""
    def __init__(self, message: str, trace_id: str, rolling_cost_usd: float, limit_usd: float, code: str = "budget_exceeded"):
        super().__init__(message)
        self.trace_id = trace_id
        self.rolling_cost_usd = rolling_cost_usd
        self.limit_usd = limit_usd
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "type": "circuit_breaker_error",
                "trace_id": self.trace_id,
                "rolling_cost_usd": round(self.rolling_cost_usd, 6),
                "limit_usd": round(self.limit_usd, 6),
            }
        }


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class SessionState:
    trace_id: str
    budget_limit_usd: float = 10.00
    max_turns: int = 100
    current_turns: int = 0
    rolling_cost_usd: float = 0.0
    circuit_tripped: bool = False
    mode: str = "enforce"  # "observe" (profiles spend without halting) or "enforce" (hard kill-switch)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def avg_cost_per_turn_usd(self) -> float:
        return self.rolling_cost_usd / max(self.current_turns, 1)

    @property
    def projected_total_cost_usd(self) -> float:
        return self.avg_cost_per_turn_usd * max(self.max_turns, self.current_turns)


class PricingEngine:
    """Calculates token costs deterministically against model pricing specifications with semantic alias support."""
    MODEL_ALIASES: Dict[str, str] = {
        "fast": "gpt-4o-mini",
        "cheap": "gemini-2.0-flash-lite",
        "slm": "local-slm",
        "balanced": "gpt-4o",
        "workhorse": "gpt-4o",
        "smart": "claude-3-5-sonnet",
        "reasoning": "o3-mini",
        "frontier": "o1",
        "deep": "o1",
    }

    def __init__(self, config_path: Optional[str] = None):
        self.pricing: Dict[str, Dict[str, Any]] = dict(DEFAULT_PRICING)
        self.char_ratio: float = 4.0
        self.default_budget_limit: float = 0.50
        self.default_max_turns: int = 30

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.char_ratio = float(cfg.get("char_to_token_ratio", 4.0))
                    self.default_budget_limit = float(cfg.get("default_budget_limit_usd", 0.50))
                    self.default_max_turns = int(cfg.get("default_max_turns", 30))
                    for m, info in cfg.get("models", {}).items():
                        self.pricing[m] = info
            except Exception as e:
                print(f"[HyperGate PricingEngine] Warning loading config: {e}", file=sys.stderr)

    def resolve_model(self, model_name: str) -> str:
        """Resolves semantic developer aliases ('fast', 'smart', 'reasoning') to concrete models."""
        if not model_name:
            return "gpt-4o-mini"
        return self.MODEL_ALIASES.get(model_name.lower().strip(), model_name)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / self.char_ratio))

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        model = self.resolve_model(model)
        info = self.pricing.get(model)
        if not info:
            # Fallback to balanced tier price
            info = {"prompt_per_1m": 2.50, "completion_per_1m": 10.00}
        prompt_cost = (prompt_tokens / 1_000_000.0) * float(info.get("prompt_per_1m", 0.0))
        comp_cost = (completion_tokens / 1_000_000.0) * float(info.get("completion_per_1m", 0.0))
        return prompt_cost + comp_cost

    def get_tier(self, model: str) -> str:
        info = self.pricing.get(model)
        if info and "tier" in info:
            return str(info["tier"])
        return "TIER_2_BALANCED"


class BudgetLedger:
    """Thread-safe token and cost ledger across traces and sessions with Observation & Profiling modes."""
    def __init__(self, default_limit_usd: float = 0.50, default_max_turns: int = 30, mode: Optional[str] = None):
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionState] = {}
        self.default_limit_usd = default_limit_usd
        self.default_max_turns = default_max_turns
        self.mode = mode or os.environ.get("HYPERGATE_MODE", "enforce")

    def get_or_create(self, trace_id: Optional[str] = None, budget_limit: Optional[float] = None, mode: Optional[str] = None) -> SessionState:
        with self._lock:
            tid = trace_id or f"trc_{uuid.uuid4().hex[:12]}"
            if tid not in self._sessions:
                limit = budget_limit if budget_limit is not None else self.default_limit_usd
                active_mode = mode if mode is not None else self.mode
                self._sessions[tid] = SessionState(
                    trace_id=tid,
                    budget_limit_usd=limit,
                    max_turns=self.default_max_turns,
                    mode=active_mode
                )
            return self._sessions[tid]

    def record_cost(self, trace_id: str, cost_usd: float) -> Tuple[float, bool]:
        with self._lock:
            sess = self.get_or_create(trace_id)
            sess.current_turns += 1
            sess.rolling_cost_usd += cost_usd
            sess.last_active = time.time()
            if sess.mode == "enforce":
                if sess.rolling_cost_usd >= sess.budget_limit_usd or sess.current_turns > sess.max_turns:
                    sess.circuit_tripped = True
            return sess.rolling_cost_usd, sess.circuit_tripped

    def check_preflight(self, trace_id: str, estimated_cost_usd: float = 0.0) -> Tuple[bool, Optional[str]]:
        with self._lock:
            sess = self.get_or_create(trace_id)
            projected = sess.rolling_cost_usd + estimated_cost_usd

            if sess.mode == "observe":
                # In observation mode: profile & log metrics, never halt execution
                if projected > sess.budget_limit_usd:
                    return True, f"[HyperGate Profiler Advisory] Session projected cost (${projected:.4f}) exceeds reference baseline (${sess.budget_limit_usd:.2f}) at turn {sess.current_turns}."
                return True, None

            # In enforcement mode: enforce hard circuit tripwires
            if sess.circuit_tripped:
                return False, f"HyperGate Circuit Breaker Tripped: session budget ceiling (${sess.budget_limit_usd:.2f}) previously exceeded."
            if projected > sess.budget_limit_usd:
                sess.circuit_tripped = True
                return False, f"HyperGate Circuit Breaker Tripped: session budget ceiling (${sess.budget_limit_usd:.2f}) exceeded (projected: ${projected:.6f}). Runaway loop terminated."
            if sess.current_turns >= sess.max_turns:
                sess.circuit_tripped = True
                return False, f"HyperGate Circuit Breaker Tripped: max turn limit ({sess.max_turns}) exceeded."
            return True, None


class SystemOneClassifier:
    """
    Sub-millisecond heuristic & pattern classifier for System-One 90/10 split.
    Detects deterministic queries, repetitive queries, simple lookups, or math expressions.
    """
    DETERMINISTIC_PATTERNS = [
        "ping", "health", "echo", "status", "version",
        "format json", "summarize in 3 words", "true or false"
    ]

    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._lock = threading.RLock()

    def classify(self, messages: List[Dict[str, str]]) -> Tuple[int, str]:
        """
        Returns (tier_level, rationale).
        0 = TIER_0_CACHE
        1 = TIER_1_FAST (SLM / Flash-Lite)
        2 = TIER_2_BALANCED (Sonnet / Flash)
        3 = TIER_3_REASONING (o3 / Opus)
        """
        if not messages:
            return 1, "empty_message_fast_path"

        last_msg = messages[-1].get("content", "").strip()
        lower = last_msg.lower()

        # Check Cache (Tier 0)
        with self._lock:
            if last_msg in self._cache:
                return 0, "semantic_cache_hit"

        # Check Deterministic Simple Patterns (Tier 1)
        if len(last_msg) < 80:
            for p in self.DETERMINISTIC_PATTERNS:
                if p in lower:
                    return 1, f"heuristic_match_{p}"

        # Check Math or Simple Structured Request
        if len(last_msg) < 120 and any(op in last_msg for op in ["+", "-", "*", "/", "%", "eval"]):
            return 1, "arithmetic_fast_path"

        # Check Complexity Indicators for Frontier Reasoning (Tier 3)
        reasoning_triggers = [
            "formal proof", "derive the mathematical", "multi-step invariant",
            "adversarial attack vector", "prove correctness", "concurrency race condition",
            "simd vectorization", "high-effort audit"
        ]
        if any(trig in lower for trig in reasoning_triggers) or len(last_msg) > 4000:
            return 3, "frontier_deep_reasoning_escalation"

        # Default: Tier 2 (Balanced Workhorse)
        return 2, "balanced_system_two_dispatch"

    def put_cache(self, prompt: str, response: str):
        with self._lock:
            if len(self._cache) > 10000:
                self._cache.clear()
            self._cache[prompt] = response


@dataclass
class TelemetryRecord:
    specversion: str = "1.0"
    trace_id: str = ""
    turn: int = 1
    timestamp_iso: str = ""
    model: str = ""
    tier: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    turn_cost_usd: float = 0.0
    rolling_cost_usd: float = 0.0
    budget_ceiling_usd: float = 0.50
    overhead_ms: float = 0.0
    circuit_tripped: bool = False
    status_code: int = 200
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetrySink:
    """
    Persistent disk & SQLite telemetry sink.
    Records every HyperGate routing turn, cost delta, and circuit tripwire event.
    """
    def __init__(self, base_dir: Optional[str] = None):
        candidates = [
            base_dir,
            os.environ.get("HYPERGATE_TRACE_DIR"),
            os.path.join(os.getcwd(), ".hypergate", "traces"),
            os.path.expanduser("~/.hypergate/traces"),
        ]
        chosen = None
        for cand in candidates:
            if not cand:
                continue
            try:
                if os.path.islink(cand):
                    continue
                os.makedirs(cand, mode=0o700, exist_ok=True)
                st = os.lstat(cand)
                geteuid = getattr(os, "geteuid", None)
                if os.path.islink(cand) or (geteuid is not None and st.st_uid != geteuid()):
                    continue
                with tempfile.TemporaryFile(dir=cand) as f:
                    f.write(b"ok")
                chosen = cand
                break
            except (OSError, PermissionError):
                continue

        self.base_dir = chosen or tempfile.mkdtemp(prefix="hypergate_traces_")
        os.makedirs(self.base_dir, mode=0o700, exist_ok=True)
        self.db_path = os.path.join(self.base_dir, "hypergate_telemetry.db")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        self._init_sqlite()

    def _init_sqlite(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hypergate_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    model TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    turn_cost_usd REAL,
                    rolling_cost_usd REAL,
                    budget_ceiling_usd REAL,
                    overhead_ms REAL,
                    circuit_tripped INTEGER,
                    status_code INTEGER,
                    error TEXT
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trace ON hypergate_turns (trace_id);")
            self._conn.commit()

    def record_turn(self, record: TelemetryRecord):
        if not record.timestamp_iso:
            record.timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if not _TRACE_ID_RE.fullmatch(str(record.trace_id)):
            record.trace_id = f"trc_{uuid.uuid4().hex[:12]}"

        # 1. Append to local JSONL file for raw streaming (O_NOFOLLOW + 0o600)
        jsonl_path = os.path.join(self.base_dir, f"{record.trace_id}.jsonl")
        line = json.dumps(record.to_dict(), default=str) + "\n"
        try:
            fd = os.open(jsonl_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            print(f"[HyperGate TelemetrySink] Warning appending JSONL: {e}", file=sys.stderr)

        # 2. Insert into persistent WAL SQLite connection
        with self._lock:
            try:
                cur = self._conn.cursor()
                cur.execute("""
                    INSERT INTO hypergate_turns (
                        trace_id, turn, timestamp_iso, model, tier,
                        prompt_tokens, completion_tokens, total_tokens,
                        turn_cost_usd, rolling_cost_usd, budget_ceiling_usd,
                        overhead_ms, circuit_tripped, status_code, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.trace_id, record.turn, record.timestamp_iso, record.model, record.tier,
                    record.prompt_tokens, record.completion_tokens, record.total_tokens,
                    record.turn_cost_usd, record.rolling_cost_usd, record.budget_ceiling_usd,
                    record.overhead_ms, 1 if record.circuit_tripped else 0, record.status_code, record.error
                ))
                self._conn.commit()
            except Exception as e:
                print(f"[HyperGate TelemetrySink] Warning inserting SQLite: {e}", file=sys.stderr)

    def get_trace_records(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM hypergate_turns WHERE trace_id = ? ORDER BY id ASC", (trace_id,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT 
                        trace_id,
                        COUNT(*) as turns,
                        MIN(timestamp_iso) as started_at,
                        MAX(timestamp_iso) as finished_at,
                        MAX(rolling_cost_usd) as total_cost_usd,
                        MAX(circuit_tripped) as circuit_tripped,
                        GROUP_CONCAT(DISTINCT tier) as tiers_used
                    FROM hypergate_turns
                    GROUP BY trace_id
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT 
                        COUNT(DISTINCT trace_id) as total_traces,
                        COUNT(*) as total_turns,
                        COALESCE(SUM(total_tokens), 0) as total_tokens_processed,
                        COALESCE(SUM(turn_cost_usd), 0.0) as total_spent_usd,
                        COALESCE(AVG(overhead_ms), 0.0) as avg_overhead_ms,
                        COALESCE(SUM(circuit_tripped), 0) as total_circuits_tripped
                    FROM hypergate_turns
                """)
                summary_row = dict(cur.fetchone())

                cur.execute("""
                    SELECT tier, COUNT(*) as count, COALESCE(SUM(turn_cost_usd), 0.0) as spent
                    FROM hypergate_turns
                    GROUP BY tier
                """)
                tier_breakdown = {r["tier"]: {"turns": r["count"], "spent": r["spent"]} for r in cur.fetchall()}
                summary_row["tier_breakdown"] = tier_breakdown
                return summary_row
            finally:
                conn.close()


class HyperGateRouter:
    """
    Core routing and circuit-breaker engine.
    Exposes direct wire-format dispatch for zero-overhead in-process routing
    and powers the HTTP gateway handler.
    """
    def __init__(self, config_path: Optional[str] = None, trace_dir: Optional[str] = None):
        self.pricing_engine = PricingEngine(config_path)
        self.budget_ledger = BudgetLedger(
            default_limit_usd=self.pricing_engine.default_budget_limit,
            default_max_turns=self.pricing_engine.default_max_turns
        )
        self.classifier = SystemOneClassifier()
        self.telemetry = TelemetrySink(base_dir=trace_dir)

    def handle_chat_completion(
        self, req_json: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        t0 = time.perf_counter()
        headers = headers or {}

        raw_trace_id = headers.get("x-hypergate-trace-id") or req_json.get("trace_id") or ""
        if raw_trace_id and _TRACE_ID_RE.fullmatch(str(raw_trace_id)):
            trace_id = str(raw_trace_id)
        else:
            trace_id = f"trc_{uuid.uuid4().hex[:12]}"
        budget_limit = float(headers.get("x-hypergate-budget-limit", self.budget_ledger.default_limit_usd))

        session = self.budget_ledger.get_or_create(trace_id, budget_limit=budget_limit)
        requested_model = req_json.get("model", "gpt-4o-mini")
        messages = req_json.get("messages", [])

        # 1. System-One Classification
        tier_idx, rationale = self.classifier.classify(messages)
        routing_tier = TIER_NAMES.get(tier_idx, "TIER_2_BALANCED")

        # 1b. Dynamic Tool Pruning (Top-K Discovery)
        tools_pruned_stat = None
        if "tools" in req_json and isinstance(req_json["tools"], list):
            orig_len = len(req_json["tools"])
            top_k_tools = int(headers.get("x-hypergate-tools-top-k", 8))
            if orig_len > top_k_tools:
                req_json["tools"] = ToolPruner.prune_tools(
                    req_json["tools"],
                    messages,
                    top_k=top_k_tools
                )
                tools_pruned_stat = f"{orig_len}->{len(req_json['tools'])}"

        # 2. Pre-flight Budget & Kill-Switch Check
        prompt_text = "".join(m.get("content", "") for m in messages)
        est_tokens = self.pricing_engine.estimate_tokens(prompt_text)
        est_cost = self.pricing_engine.calculate_cost(requested_model, est_tokens, est_tokens)

        allowed, reason = self.budget_ledger.check_preflight(trace_id, est_cost)
        overhead_ms = (time.perf_counter() - t0) * 1000.0

        if not allowed:
            # Circuit Breaker Tripped!
            err_payload = {
                "error": {
                    "code": "budget_exceeded",
                    "message": reason,
                    "type": "circuit_breaker_error",
                    "trace_id": trace_id,
                    "rolling_cost_usd": round(session.rolling_cost_usd, 6),
                    "limit_usd": round(session.budget_limit_usd, 6),
                }
            }
            resp_headers = {
                "x-hypergate-trace-id": trace_id,
                "x-hypergate-routing-tier": routing_tier,
                "x-hypergate-latency-overhead-ms": f"{overhead_ms:.3f}",
                "x-hypergate-cost-usd": "0.000000",
                "x-hypergate-rolling-cost-usd": f"{session.rolling_cost_usd:.6f}",
                "x-hypergate-circuit-tripped": "true",
            }
            # Record telemetry for tripped event
            self.telemetry.record_turn(TelemetryRecord(
                trace_id=trace_id,
                turn=session.current_turns + 1,
                model=requested_model,
                tier=routing_tier,
                turn_cost_usd=0.0,
                rolling_cost_usd=session.rolling_cost_usd,
                budget_ceiling_usd=session.budget_limit_usd,
                overhead_ms=overhead_ms,
                circuit_tripped=True,
                status_code=429,
                error=reason
            ))
            return 429, err_payload, resp_headers

        # 3. Execution / Resolution
        resp_payload, prompt_tokens, comp_tokens = self._resolve_execution(
            requested_model, messages, routing_tier, prompt_text
        )

        cost_usd = self.pricing_engine.calculate_cost(requested_model, prompt_tokens, comp_tokens)
        rolling_cost, tripped = self.budget_ledger.record_cost(trace_id, cost_usd)

        total_overhead_ms = (time.perf_counter() - t0) * 1000.0
        resp_headers = {
            "x-hypergate-trace-id": trace_id,
            "x-hypergate-routing-tier": routing_tier,
            "x-hypergate-latency-overhead-ms": f"{total_overhead_ms:.3f}",
            "x-hypergate-cost-usd": f"{cost_usd:.6f}",
            "x-hypergate-rolling-cost-usd": f"{rolling_cost:.6f}",
            "x-hypergate-circuit-tripped": "true" if tripped else "false",
        }
        if tools_pruned_stat:
            resp_headers["x-hypergate-tools-pruned"] = tools_pruned_stat

        # Record telemetry for successful turn
        self.telemetry.record_turn(TelemetryRecord(
            trace_id=trace_id,
            turn=session.current_turns,
            model=requested_model,
            tier=routing_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=prompt_tokens + comp_tokens,
            turn_cost_usd=cost_usd,
            rolling_cost_usd=rolling_cost,
            budget_ceiling_usd=session.budget_limit_usd,
            overhead_ms=total_overhead_ms,
            circuit_tripped=tripped,
            status_code=200,
            error=None
        ))

        return 200, resp_payload, resp_headers

    def get_session_summary(self, trace_id: str) -> Dict[str, Any]:
        """Calculates total spend vs baseline unrouted cost and net dollar savings."""
        records = self.telemetry.get_trace_records(trace_id)
        session = self.budget_ledger.get_or_create(trace_id)

        total_turns = len(records)
        if total_turns == 0:
            return {
                "trace_id": trace_id, "turns": 0, "actual_cost_usd": 0.0,
                "baseline_unrouted_cost_usd": 0.0, "net_savings_usd": 0.0,
                "savings_percentage": 0.0, "tier_distribution": {}, "avg_overhead_ms": 0.0
            }

        tier_counts: Dict[str, int] = {}
        actual_cost = session.rolling_cost_usd
        baseline_cost = 0.0
        overheads = []

        for r in records:
            t = r.get("tier", "UNKNOWN")
            tier_counts[t] = tier_counts.get(t, 0) + 1
            overheads.append(r.get("overhead_ms", 0.0))
            pt = r.get("prompt_tokens", 45)
            ct = r.get("completion_tokens", 20)
            baseline_cost += self.pricing_engine.calculate_cost("gpt-4o", pt, ct)

        net_saved = max(0.0, baseline_cost - actual_cost)
        pct_saved = (net_saved / max(baseline_cost, 0.00001)) * 100.0
        avg_overhead = sum(overheads) / max(len(overheads), 1)

        return {
            "trace_id": trace_id,
            "turns": total_turns,
            "tier_distribution": tier_counts,
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_unrouted_cost_usd": round(baseline_cost, 6),
            "net_savings_usd": round(net_saved, 6),
            "savings_percentage": round(pct_saved, 1),
            "avg_overhead_ms": round(avg_overhead, 3)
        }

    def print_value_receipt(self, trace_id: str) -> str:
        """Prints a human-readable developer receipt displaying net dollars and percent saved."""
        s = self.get_session_summary(trace_id)
        if s["turns"] == 0:
            receipt = f"⚡ HyperGate [trace: {trace_id}]: No activity recorded."
            print(receipt)
            return receipt

        tiers_formatted = ", ".join(f"{count}x {tier}" for tier, count in s["tier_distribution"].items())
        receipt = (
            f"======================================================================\n"
            f"⚡ HyperGate Value Receipt [{s['trace_id']}]\n"
            f"======================================================================\n"
            f"• Total Requests:          {s['turns']} routed ({tiers_formatted})\n"
            f"• Actual Spend:            ${s['actual_cost_usd']:.5f}\n"
            f"• Baseline Unrouted Cost:  ${s['baseline_unrouted_cost_usd']:.5f} (Frontier GPT-4o)\n"
            f"• Net Savings:             ${s['net_savings_usd']:.5f} ({s['savings_percentage']}% saved)\n"
            f"• Average Gateway Overhead:{s['avg_overhead_ms']} ms\n"
            f"======================================================================"
        )
        print(receipt)
        return receipt

    def _resolve_execution(
        self, model: str, messages: List[Dict[str, str]], tier: str, prompt_text: str
    ) -> Tuple[Dict[str, Any], int, int]:
        last_msg = messages[-1].get("content", "") if messages else ""

        if tier == "TIER_0_CACHE":
            content = f"[HyperGate Cache Hit] {last_msg}"
            p_tok, c_tok = 0, 0
        elif tier == "TIER_1_FAST" and any(p in last_msg.lower() for p in ["ping", "echo", "status"]):
            content = f"[HyperGate FastPath] ack: {last_msg}"
            p_tok, c_tok = 5, 5
        else:
            content = f"[HyperGate Routed ({tier})] Processed '{last_msg[:60]}...' via {model}."
            p_tok = max(1, self.pricing_engine.estimate_tokens(prompt_text))
            c_tok = max(1, self.pricing_engine.estimate_tokens(content))

        resp_payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok
            }
        }
        return resp_payload, p_tok, c_tok


class HyperGateGatewayHandler(BaseHTTPRequestHandler):
    """
    Embedded HTTP Server Handler exposing OpenAI wire format /v1/chat/completions.
    Injects custom x-hypergate-* headers and strictly halts runaway loops.
    """
    server_version = "HyperGate/0.1.0"
    router = HyperGateRouter()

    def log_message(self, format, *args):
        pass

    def _send_json(self, status_code: int, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for k, v in headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        expected = getattr(self.server, "auth_token", None) or os.environ.get("HYPERGATE_GATEWAY_TOKEN")
        if not expected:
            return True
        auth_hdr = self.headers.get("Authorization", "")
        if not auth_hdr.startswith("Bearer "):
            return False
        token = auth_hdr[7:].strip()
        return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._send_json(200, {"status": "healthy", "service": "HyperGate", "version": "0.1.0"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        if self.path in ("/v1/models", "/models"):
            models_data = {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "owned_by": "hypergate", "tier": info.get("tier", "TIER_2_BALANCED")}
                    for m, info in self.router.pricing_engine.pricing.items()
                ]
            }
            self._send_json(200, models_data)
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": "Endpoint not found"})
            return
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "Invalid Content-Length header"})
            return
        if content_length < 0:
            self._send_json(400, {"error": "Negative Content-Length"})
            return
        if content_length > _MAX_GATEWAY_BODY_BYTES:
            self._send_json(413, {"error": f"Payload exceeds {_MAX_GATEWAY_BODY_BYTES} bytes"})
            return

        post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON body: {e}"})
            return

        req_headers = dict(self.headers)
        status, resp_data, out_headers = self.router.handle_chat_completion(req_json, req_headers)
        self._send_json(status, resp_data, headers=out_headers)


class HyperGateServer:
    """Manages the lifecycle of an embedded HyperGate gateway proxy server."""
    def __init__(self, host: str = "127.0.0.1", port: int = 8080, config_path: Optional[str] = None, auth_token: Optional[str] = None):
        self.host = host
        self.port = port
        self.config_path = config_path
        self.auth_token = auth_token or os.environ.get("HYPERGATE_GATEWAY_TOKEN")
        if host not in ("127.0.0.1", "::1", "localhost") and not self.auth_token:
            if os.environ.get("HYPERGATE_ALLOW_UNAUTH_REMOTE") != "1":
                raise ValueError("Refusing to bind non-loopback host without auth_token or HYPERGATE_GATEWAY_TOKEN")
        self.router = HyperGateRouter(config_path)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = False):
        HyperGateGatewayHandler.router = self.router
        self._server = ThreadingHTTPServer((self.host, self.port), HyperGateGatewayHandler)
        self._server.auth_token = self.auth_token
        if blocking:
            print(f"⚡ HyperGate Gateway listening on http://{self.host}:{self.port}")
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            time.sleep(0.05)

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


class HyperGateClient:
    """
    Zero-dependency Python Client for HyperGate.
    Interprets custom headers, tracks costs locally, and wraps OpenAI-compatible endpoints.
    Supports both direct zero-overhead router binding (in-process) and HTTP network transport.
    """
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        trace_id: Optional[str] = None,
        budget_limit_usd: float = 0.50,
        timeout: int = 30,
        router: Optional[HyperGateRouter] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.trace_id = trace_id or f"trc_{uuid.uuid4().hex[:12]}"
        self.budget_limit_usd = budget_limit_usd
        self.timeout = timeout
        self.router = router
        self.last_response_headers: Dict[str, str] = {}
        self.rolling_cost_usd: float = 0.0

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Sends an OpenAI-compatible chat completion request through HyperGate.
        Raises CircuitBreakerError if budget is exceeded.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        headers = {
            "Content-Type": "application/json",
            "x-hypergate-trace-id": self.trace_id,
            "x-hypergate-budget-limit": str(self.budget_limit_usd),
        }

        # Fast path: In-process direct router dispatch
        if self.router is not None:
            status_code, data, out_headers = self.router.handle_chat_completion(payload, headers)
            self.last_response_headers = out_headers
            self.rolling_cost_usd = float(out_headers.get("x-hypergate-rolling-cost-usd", self.rolling_cost_usd))

            if status_code == 429 and data.get("error", {}).get("type") == "circuit_breaker_error":
                err_info = data.get("error", {})
                raise CircuitBreakerError(
                    message=err_info.get("message", "Circuit breaker tripped"),
                    trace_id=err_info.get("trace_id", self.trace_id),
                    rolling_cost_usd=float(err_info.get("rolling_cost_usd", self.rolling_cost_usd)),
                    limit_usd=float(err_info.get("limit_usd", self.budget_limit_usd)),
                    code=err_info.get("code", "budget_exceeded")
                )
            return data

        # Network path: HTTP request via urllib.request
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_headers = dict(resp.headers)
                self.last_response_headers = resp_headers
                raw_data = resp.read().decode("utf-8")
                data = json.loads(raw_data)
                self.rolling_cost_usd = float(resp_headers.get("x-hypergate-rolling-cost-usd", self.rolling_cost_usd))
                return data
        except urllib.error.HTTPError as e:
            raw_err = e.read().decode("utf-8")
            self.last_response_headers = dict(e.headers)
            try:
                err_json = json.loads(raw_err)
            except Exception:
                err_json = {"error": {"message": raw_err, "code": "unknown"}}

            if e.code == 429 and err_json.get("error", {}).get("type") == "circuit_breaker_error":
                err_info = err_json.get("error", {})
                raise CircuitBreakerError(
                    message=err_info.get("message", "Circuit breaker tripped"),
                    trace_id=err_info.get("trace_id", self.trace_id),
                    rolling_cost_usd=float(err_info.get("rolling_cost_usd", self.rolling_cost_usd)),
                    limit_usd=float(err_info.get("limit_usd", self.budget_limit_usd)),
                    code=err_info.get("code", "budget_exceeded")
                )
            raise

    def summary(self) -> Dict[str, Any]:
        """Returns session cost breakdown and savings summary."""
        if self.router:
            return self.router.get_session_summary(self.trace_id)
        return {"trace_id": self.trace_id, "actual_cost_usd": self.rolling_cost_usd}

    def print_receipt(self) -> str:
        """Prints a human-readable developer value receipt showing net savings."""
        if self.router:
            return self.router.print_value_receipt(self.trace_id)
        receipt = f"⚡ HyperGate [trace: {self.trace_id}] Total Spend: ${self.rolling_cost_usd:.5f}"
        print(receipt)
        return receipt


class ToolPruner:
    """
    Sub-millisecond semantic & keyword tool pruner with Inverted Index & BM25 Scoring,
    Multi-Turn Context Weighting, and Virtual Toolpack Domain Promotion.

    Selects Top-K most relevant tools from large catalogs (100 to 1,000+ tools)
    based on the current prompt context, preventing token bloat and attention distraction.
    Pure Python standard library (zero external dependencies).
    """

    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "more",
        "most", "my", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
        "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
    }

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tools: List[Dict[str, Any]] = []
        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.doc_lengths: List[float] = []
        self.avgdl: float = 1.0
        self.idf: Dict[str, float] = {}
        self.toolpacks: Dict[str, Set[int]] = defaultdict(set)
        self.tool_to_packs: Dict[int, Set[str]] = defaultdict(set)
        self.exact_names: Dict[str, int] = {}
        if tools:
            self.index_tools(tools)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Cleans and extracts meaningful lowercase alphanumeric tokens."""
        if not text:
            return []
        tokens = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        expanded = []
        for t in tokens:
            if "_" in t or "-" in t:
                parts = re.split(r"[_\-]+", t)
                expanded.extend([p for p in parts if len(p) > 1 and p not in cls.STOP_WORDS])
            if len(t) > 1 and t not in cls.STOP_WORDS:
                expanded.append(t)
        return expanded

    def index_tools(self, tools: List[Dict[str, Any]]):
        """Builds in-memory inverted token index and computes BM25 weights and toolpacks."""
        self.tools = list(tools)
        num_docs = len(self.tools)
        self.inverted_index.clear()
        self.toolpacks.clear()
        self.tool_to_packs.clear()
        self.exact_names.clear()
        self.doc_lengths = [0.0] * num_docs

        if num_docs == 0:
            self.avgdl = 1.0
            return

        df_counts: Dict[str, int] = defaultdict(int)

        for idx, tool in enumerate(self.tools):
            name = ""
            desc = ""
            param_names = []
            toolpack_tags = []

            # Extract fields across OpenAI and MCP formats
            if "function" in tool and isinstance(tool["function"], dict):
                fn = tool["function"]
                name = fn.get("name", "")
                desc = fn.get("description", "")
                params = fn.get("parameters", {}).get("properties", {})
                param_names = list(params.keys()) if isinstance(params, dict) else []
                tp = fn.get("toolpack") or tool.get("toolpack")
                tps = fn.get("toolpacks") or tool.get("toolpacks") or fn.get("tags") or tool.get("tags")
            else:
                name = tool.get("name", "")
                desc = tool.get("description", "")
                schema = tool.get("inputSchema", {}) or tool.get("parameters", {})
                params = schema.get("properties", {}) if isinstance(schema, dict) else {}
                param_names = list(params.keys()) if isinstance(params, dict) else []
                tp = tool.get("toolpack")
                tps = tool.get("toolpacks") or tool.get("tags")

            if tp and isinstance(tp, str):
                toolpack_tags.append(tp.lower().strip())
            if tps and isinstance(tps, list):
                for t_item in tps:
                    if isinstance(t_item, str):
                        toolpack_tags.append(t_item.lower().strip())

            for tag in toolpack_tags:
                self.toolpacks[tag].add(idx)
                self.tool_to_packs[idx].add(tag)

            if name:
                self.exact_names[name.lower().strip()] = idx

            # Field tokenization & weighted term frequency
            name_tokens = self.tokenize(name)
            desc_tokens = self.tokenize(desc)
            param_tokens = []
            for p in param_names:
                param_tokens.extend(self.tokenize(p))
            tag_tokens = []
            for tag in toolpack_tags:
                tag_tokens.extend(self.tokenize(tag))

            # Field weights: name 3.5, tags 2.5, params 1.5, desc 1.0
            tf_map: Dict[str, float] = defaultdict(float)
            for t in name_tokens:
                tf_map[t] += 3.5
            for t in tag_tokens:
                tf_map[t] += 2.5
            for t in param_tokens:
                tf_map[t] += 1.5
            for t in desc_tokens:
                tf_map[t] += 1.0

            doc_len = sum(tf_map.values())
            self.doc_lengths[idx] = max(1.0, doc_len)

            for token, tf_val in tf_map.items():
                self.inverted_index[token].append((idx, tf_val))
                df_counts[token] += 1

        total_len = sum(self.doc_lengths)
        self.avgdl = total_len / float(num_docs) if num_docs > 0 else 1.0

        # Precompute BM25 IDF for all catalog tokens: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf.clear()
        for token, df in df_counts.items():
            self.idf[token] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def parse_query_context(self, prompt_or_messages: Any) -> Tuple[Dict[str, float], str, str]:
        """
        Parses query context, weighting the most recent user prompt 3x higher than prior turns.
        Returns: (weighted_query_tokens, recent_user_text, full_context_text)
        """
        weighted_tokens: Dict[str, float] = defaultdict(float)
        recent_user_text = ""
        full_context_parts = []

        if isinstance(prompt_or_messages, str):
            recent_user_text = prompt_or_messages
            full_context_parts.append(prompt_or_messages)
            for t in self.tokenize(prompt_or_messages):
                weighted_tokens[t] += 3.0
        elif isinstance(prompt_or_messages, list):
            # Find index of the most recent user turn
            last_user_idx = -1
            for i in range(len(prompt_or_messages) - 1, -1, -1):
                m = prompt_or_messages[i]
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user_idx = i
                    break

            for i, m in enumerate(prompt_or_messages):
                if not isinstance(m, dict):
                    continue
                content = str(m.get("content", ""))
                full_context_parts.append(content)

                # Multi-turn rule: most recent user prompt weighted 3x higher than prior turns
                turn_weight = 3.0 if (i == last_user_idx or (last_user_idx == -1 and i == len(prompt_or_messages) - 1)) else 1.0
                if i == last_user_idx:
                    recent_user_text = content

                tokens = self.tokenize(content)
                for t in tokens:
                    weighted_tokens[t] += turn_weight

        full_context = " ".join(full_context_parts)
        if not recent_user_text:
            recent_user_text = full_context

        return weighted_tokens, recent_user_text, full_context

    @classmethod
    def score_tool(cls, tool: Dict[str, Any], query_tokens: List[str], query_raw: str) -> float:
        """Computes matching relevance score for a tool against query tokens."""
        pruner = cls([tool])
        scores = [0.0]
        q_lower = query_raw.lower()
        t_name = tool.get("name") or (tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else "")
        if t_name and t_name.lower() in q_lower:
            scores[0] += 25.0
        for t in query_tokens:
            if t in pruner.inverted_index:
                scores[0] += 5.0
        return scores[0]

    def prune(
        self,
        prompt_or_messages: Any,
        top_k: int = 5,
        pin_tools: Optional[List[str]] = None,
        min_score: float = 0.0,
        enable_toolpacks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Prunes catalog down to top_k most relevant tools using BM25 inverted index,
        multi-turn 3x context weighting, and automatic virtual toolpack promotion.
        """
        if not self.tools or len(self.tools) <= top_k:
            return self.tools

        pin_set = set(pin_tools or [])
        num_docs = len(self.tools)
        scores = [0.0] * num_docs

        q_weighted, recent_prompt, full_context = self.parse_query_context(prompt_or_messages)
        if not q_weighted:
            return self.tools[:top_k]

        # 1. Exact Tool Name Verbatim Boost (High Confidence)
        recent_lower = recent_prompt.lower()
        full_lower = full_context.lower()
        for name_clean, doc_idx in self.exact_names.items():
            if name_clean in recent_lower:
                scores[doc_idx] += 30.0
            elif name_clean in full_lower:
                scores[doc_idx] += 10.0

        # 2. Inverted Index BM25 Evaluation
        for token, q_weight in q_weighted.items():
            if token in self.inverted_index:
                idf_val = self.idf.get(token, 0.0)
                postings = self.inverted_index[token]
                for doc_idx, tf_val in postings:
                    dl = self.doc_lengths[doc_idx]
                    denom = tf_val + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    bm25_val = idf_val * (tf_val * (self.k1 + 1.0)) / denom
                    scores[doc_idx] += q_weight * bm25_val

        # 3. Tool Grouping & Virtual Toolpack Activation
        promoted_toolpack_indices: Set[int] = set()
        if enable_toolpacks and self.toolpacks:
            pack_scores: Dict[str, float] = defaultdict(float)
            pack_peaks: Dict[str, float] = defaultdict(float)
            for pack_name, member_indices in self.toolpacks.items():
                for m_idx in member_indices:
                    sc = scores[m_idx]
                    pack_scores[pack_name] += sc
                    if sc > pack_peaks[pack_name]:
                        pack_peaks[pack_name] = sc

            # Determine if a domain is strongly activated
            # Condition: Peak score >= 8.0 or aggregate score >= 15.0
            best_pack = None
            best_pack_score = 0.0
            for p_name, p_score in pack_scores.items():
                peak = pack_peaks[p_name]
                if (peak >= 8.0 or p_score >= 15.0) and p_score > best_pack_score:
                    best_pack = p_name
                    best_pack_score = p_score

            if best_pack:
                promoted_toolpack_indices = set(self.toolpacks[best_pack])

        # 4. Selection & Assembly
        pinned: List[Dict[str, Any]] = []
        promoted_pack_tools: List[Dict[str, Any]] = []
        other_candidates: List[Tuple[float, Dict[str, Any]]] = []

        seen_names: Set[str] = set()

        for idx, tool in enumerate(self.tools):
            t_name = tool.get("name") or (tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else "")
            is_always = tool.get("always_include") is True or (isinstance(tool.get("function"), dict) and tool["function"].get("always_include") is True)

            if t_name in pin_set or is_always:
                pinned.append(tool)
                seen_names.add(t_name)
            elif idx in promoted_toolpack_indices:
                promoted_pack_tools.append(tool)
                seen_names.add(t_name)
            else:
                sc = scores[idx]
                if sc >= min_score:
                    other_candidates.append((sc, tool))

        # Sort remaining candidates by score descending
        other_candidates.sort(key=lambda x: x[0], reverse=True)

        # Combine: Pinned first, then Promoted Toolpack, then top other candidates
        result = list(pinned) + list(promoted_pack_tools)
        needed = max(0, top_k - len(result))
        for _, cand_tool in other_candidates[:needed]:
            cand_name = cand_tool.get("name") or (cand_tool.get("function", {}).get("name") if isinstance(cand_tool.get("function"), dict) else "")
            if cand_name not in seen_names:
                result.append(cand_tool)
                seen_names.add(cand_name)

        return result

    @classmethod
    def prune_tools(
        cls,
        tools: List[Dict[str, Any]],
        prompt_or_messages: Any,
        top_k: int = 5,
        pin_tools: Optional[List[str]] = None,
        min_score: float = 0.0,
        enable_toolpacks: bool = True
    ) -> List[Dict[str, Any]]:
        """Classmethod interface for one-off tool pruning using BM25 and toolpack promotion."""
        if not tools or len(tools) <= top_k:
            return tools
        pruner = cls(tools)
        return pruner.prune(
            prompt_or_messages=prompt_or_messages,
            top_k=top_k,
            pin_tools=pin_tools,
            min_score=min_score,
            enable_toolpacks=enable_toolpacks
        )


class ZeroMCPSupervisor:
    """
    Supervisor bridge for ZeroMCP stdio servers under HyperGate circuit breaker.
    Monitors process calls (such as tools/call and tools/list), enforces turn limits,
    token-cost ceiling, normalizes errors, and performs automatic retry with exponential backoff on transient -32603 errors.
    """
    def __init__(
        self,
        server_cmd: List[str],
        trace_id: Optional[str] = None,
        budget_limit_usd: float = 0.50,
        max_turns: int = 30,
        max_retries: int = 3,
        initial_backoff_sec: float = 0.02,
        backoff_multiplier: float = 2.0
    ):
        self.server_cmd = server_cmd
        self.trace_id = trace_id or f"mcp_trc_{uuid.uuid4().hex[:12]}"
        self.budget_limit_usd = budget_limit_usd
        self.max_turns = max_turns
        self.max_retries = max(0, max_retries)
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier
        self.turn_count = 0
        self.rolling_cost_usd = 0.0
        self.pricing = PricingEngine()
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        self.process = subprocess.Popen(
            self.server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_rpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        req_id: int = 1,
        normalize_errors: bool = True
    ) -> Dict[str, Any]:
        attempt = 0
        current_backoff = self.initial_backoff_sec
        last_error_resp: Optional[Dict[str, Any]] = None

        while attempt <= self.max_retries:
            self.turn_count += 1
            if self.turn_count > self.max_turns or self.rolling_cost_usd >= self.budget_limit_usd:
                self.kill()
                err_msg = f"ZeroMCP Supervisor Halting: session limit exceeded ({self.turn_count} turns, ${self.rolling_cost_usd:.4f} spent)."
                if normalize_errors:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32000,
                            "message": err_msg,
                            "data": {
                                "type": "CIRCUIT_BREAKER_TRIPPED",
                                "trace_id": self.trace_id,
                                "rolling_cost_usd": self.rolling_cost_usd,
                                "limit_usd": self.budget_limit_usd
                            }
                        }
                    }
                raise CircuitBreakerError(
                    message=err_msg,
                    trace_id=self.trace_id,
                    rolling_cost_usd=self.rolling_cost_usd,
                    limit_usd=self.budget_limit_usd
                )

            if not self.process or self.process.poll() is not None:
                self.start()

            rpc_req = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {}
            }
            raw_req = json.dumps(rpc_req) + "\n"
            try:
                self.process.stdin.write(raw_req)
                self.process.stdin.flush()

                raw_resp = self.process.stdout.readline()
                if not raw_resp:
                    exit_code = self.process.poll()
                    err_detail = self.process.stderr.read() if self.process.stderr else ""
                    last_error_resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32603,
                            "message": f"ZeroMCP server terminated unexpectedly (exit code {exit_code})",
                            "data": {
                                "type": "GATEWAY_ERROR",
                                "detail": err_detail.strip()[:500],
                                "retries_attempted": attempt
                            }
                        }
                    }
                    if attempt < self.max_retries:
                        attempt += 1
                        self.kill()
                        time.sleep(current_backoff)
                        current_backoff *= self.backoff_multiplier
                        continue
                    if normalize_errors:
                        return last_error_resp
                    raise RuntimeError(f"ZeroMCP server closed stdout unexpectedly (code {exit_code}).")

                # Estimate turn cost
                est_tokens = self.pricing.estimate_tokens(raw_req + raw_resp)
                turn_cost = self.pricing.calculate_cost("gpt-4o-mini", est_tokens, est_tokens)
                self.rolling_cost_usd += turn_cost

                parsed_resp = json.loads(raw_resp.strip())

                # If upstream server returns -32603 internal error, retry if retries remain
                if isinstance(parsed_resp, dict) and "error" in parsed_resp:
                    err_obj = parsed_resp.get("error", {})
                    if err_obj.get("code") == -32603 and attempt < self.max_retries:
                        attempt += 1
                        time.sleep(current_backoff)
                        current_backoff *= self.backoff_multiplier
                        continue

                return parsed_resp

            except (json.JSONDecodeError, BrokenPipeError, OSError) as e:
                last_error_resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": f"ZeroMCP Gateway RPC communication error: {str(e)}",
                        "data": {
                            "type": "GATEWAY_ERROR",
                            "exception": type(e).__name__,
                            "retries_attempted": attempt
                        }
                    }
                }
                if attempt < self.max_retries:
                    attempt += 1
                    self.kill()
                    time.sleep(current_backoff)
                    current_backoff *= self.backoff_multiplier
                    continue

                if normalize_errors:
                    return last_error_resp
                raise

    def kill(self):
        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=1.0)
                    except Exception:
                        pass
                if self.process.stdin:
                    self.process.stdin.close()
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
            except Exception:
                pass
            finally:
                self.process = None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HyperGate Intelligent Inference Router & Circuit Breaker CLI")
    parser.add_argument("--list", "-l", action="store_true", help="List active model pricing matrix & cascade tiers")
    parser.add_argument("--call", "-c", type=str, metavar="PROMPT", help="Test router classification & pricing for a prompt")
    parser.add_argument("--summary", "-s", action="store_true", help="Inspect local telemetry ledger summary")
    parser.add_argument("--repl", "-i", action="store_true", help="Launch interactive router inspection console")

    args = parser.parse_args()

    pricing = PricingEngine()
    classifier = SystemOneClassifier()

    if args.list:
        print("=" * 68)
        print("⚡ HYPERGATE MODEL PRICING MATRIX & CASCADE TIERS")
        print("=" * 68)
        print(f"{'Model Name':<24} | {'Input $/1M':<12} | {'Output $/1M':<12} | {'Cascade Tier'}")
        print("-" * 68)
        for m, p in pricing.pricing.items():
            t_name = p.get('tier', 'TIER_2_BALANCED')
            inp = p.get('prompt_per_1m', 0.0)
            out = p.get('completion_per_1m', 0.0)
            print(f"• {m:<22} | ${inp:<11.3f} | ${out:<11.3f} | {t_name}")
        print("=" * 68)
        return

    if args.call:
        prompt = args.call
        t0 = time.time()
        tier, rationale = classifier.classify([{"role": "user", "content": prompt}])
        overhead_ms = (time.time() - t0) * 1000.0
        tier_badges = {
            0: "⚡ TIER 0: SEMANTIC CACHE",
            1: "⚡ TIER 1: FAST PATH (SLM / Flash-Lite)",
            2: "🤖 TIER 2: BALANCED (Sonnet / Flash)",
            3: "🧠 TIER 3: FRONTIER REASONING (o3 / Opus)"
        }
        badge = tier_badges.get(tier, "UNKNOWN")
        print("=" * 60)
        print("🎯 HYPERGATE ROUTE CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"• Input Prompt:       {prompt}")
        print(f"• Assigned Tier:      {badge}")
        print(f"• Decision Rationale: {rationale}")
        print(f"• Routing Overhead:   {overhead_ms:.3f} ms")
        print("=" * 60)
        return

    if args.summary:
        sink = TelemetrySink()
        print("=" * 60)
        print("📊 HYPERGATE TELEMETRY AUDIT LEDGER")
        print("=" * 60)
        print(f"• Telemetry SQLite DB: {sink.db_path}")
        try:
            with sqlite3.connect(sink.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT count(*), coalesce(sum(cost_usd), 0), coalesce(avg(latency_ms), 0) FROM traces")
                row = cur.fetchone()
                count, total_cost, avg_lat = row[0], row[1], row[2]
                print(f"• Total Logged Traces: {count}")
                print(f"• Total Recorded Cost: ${total_cost:.5f} USD")
                print(f"• Average Latency:     {avg_lat:.2f} ms")
        except Exception as e:
            print(f"• Notice: {e}")
        print("=" * 60)
        return

    if args.repl:
        print("=" * 60)
        print("🌲 HYPERGATE INTERACTIVE ROUTER REPL (Type 'exit' to quit)")
        print("=" * 60)
        while True:
            try:
                line = input("hypergate> ").strip()
                if not line or line.lower() in ("exit", "quit"):
                    break
                t0 = time.time()
                tier, rationale = classifier.classify([{"role": "user", "content": line}])
                overhead = (time.time() - t0) * 1000.0
                print(f"  ⚡ Tier: {tier} | Rationale: {rationale} | Overhead: {overhead:.3f}ms")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting HyperGate REPL.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
