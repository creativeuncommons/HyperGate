# HyperGate Specification & Technical Architecture

## 1. Overview & Objectives

**HyperGate** is an ultra-low-latency, zero-external-dependency inference gateway, dynamic tool pruning catalog, and circuit-breaking router designed for autonomous AI multi-agent architectures.

### Primary Objectives
* **Sub-Millisecond Routing Overhead**: Guarantee end-to-end routing, system-one classification, and tool pruning across 500+ candidate tools in $< 1.0\text{ ms}$.
* **Zero Dependency Mandate**: Pure Python standard library implementation (`collections`, `math`, `re`, `json`, `time`, `threading`, `typing`, `subprocess`).
* **Dynamic Semantic Tool Pruning**: Eliminate up to $95\%+$ of tool schema token bloat before prompt submission using in-memory inverted index BM25 term weighting.
* **Virtual Toolpack Routing**: Automatically group tools into functional domains and promote entire related toolpacks when domain relevance thresholds trigger.
* **Upstream Resilience**: Intercept transient gateway failures (`-32603`, transport disconnects) with exponential backoff retry and defensive child-process lifecycle management.
* **Deterministic Circuit Breaking**: Fail-closed protection against cascading upstream model or tool provider outages.

---

## 2. Core Architectural Invariants

### Invariant 1: Zero External Dependencies
* The engine executes entirely on the Python standard library.
* Zero external wheels, C-extensions, or heavy semantic embedding vector databases.
* Instant cold-start startup ($< 5\text{ ms}$).

### Invariant 2: Sub-Millisecond Latency Budget
* Dynamic tool pruning across 500+ tools must complete in $< 1.0\text{ ms}$ (measured empirical median: $0.194\text{ ms}$).
* System-One classification completes in $\sim 1.0\text{ µs}$ ($1,000,000+\text{ ops/sec}$).
* Deterministic token pricing and cost accounting completes in $\sim 0.38\text{ µs}$ ($2,600,000+\text{ ops/sec}$).

### Invariant 3: Okapi BM25 Term Weighting & Inverted Index
The tool pruner maintains an in-memory inverted postings index mapping token stems to matching tool identifiers:
$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* Standard tuning parameters: $k_1 = 1.5, b = 0.75$.
* Postings list traversal evaluates only documents sharing query tokens, skipping non-matching tools entirely.
* Field Frequency Multipliers:
  * **Tool Name**: $3.5\times$
  * **Toolpack / Domain Tags**: $2.5\times$
  * **Parameter Schema & Type Properties**: $1.5\times$
  * **Description Prose**: $1.0\times$

### Invariant 4: Multi-Turn Conversational Context Weighting
To eliminate goal drift while preserving context awareness:
$$\text{Query Term Count}(t) = 3.0 \times \text{Count}_{\text{Latest Prompt}}(t) + 1.0 \times \text{Count}_{\text{Prior Turns}}(t)$$
Tokens appearing in the most recent user turn receive $3.0\times$ weighting over trailing conversation history.

### Invariant 5: Virtual Toolpack Domain Promotion
* Tools declare domain affinities via `toolpack` tags (e.g. `sports_analytics`, `sql_queries`, `system_ops`).
* When an individual tool matches with peak score $\ge 8.0$, or when cumulative domain activations reach $\ge 15.0$, HyperGate automatically promotes the entire related toolpack into the candidate pool.

### Invariant 6: Resilient Gateway Error Recovery (`ZeroMCPSupervisor`)
* Upstream tool execution errors mapped to `-32603` (Internal Error) or socket disconnections are automatically intercepted.
* Configurable retry policy: `max_retries: int = 3`, `initial_backoff_sec: float = 0.02`, `backoff_multiplier: float = 2.0`.
* Child process lifecycle management explicitly reaps terminated processes (`process.wait(timeout=1.0)`), eliminating zombie processes and preventing resource leaks.

---

## 3. Protocol & State Machine Specifications

### 3.1 Circuit Breaker State Machine
```
   ┌──────────┐   Error Rate >= Threshold    ┌──────────┐
   │  CLOSED  │ ───────────────────────────► │   OPEN   │
   └──────────┘                              └──────────┘
        ▲                                          │
        │ Success Count >= Reset Req               │ Cooldown Expired
        │                                          ▼
   ┌──────────┐                              ┌──────────┐
   │  CLOSED  │ ◄─────────────────────────── │HALF_OPEN │
   └──────────┘   Probe Success              └──────────┘
```
1. **CLOSED**: Normal traffic flow. Consecutive failures and rolling error windows tracked.
2. **OPEN**: Upstream trips threshold ($> 50\%$ failure rate or $N$ consecutive failures). Fast-fails incoming calls with HTTP 503 / JSON-RPC -32000.
3. **HALF_OPEN**: Cooldown window expires (default: 30s). Limited probe requests admitted. One failure resets state to OPEN; consecutive successes transition to CLOSED.

### 3.2 Pricing & Cost Accounting Schema
```json
{
  "model": "claude-opus-5-5",
  "input_tokens": 1420,
  "output_tokens": 310,
  "input_cost_usd": 0.02130,
  "output_cost_usd": 0.02325,
  "total_cost_usd": 0.04455,
  "pricing_timestamp": "2026-11-17T12:00:00Z"
}
```

---

## 4. Verification & Testing Standards

1. **Unit Verification**: 100% pass rate on all circuit breaker, pricing engine, tool pruner, and gateway tests.
2. **Benchmark Verification**: Automated continuous regression testing asserting $< 1.0\text{ ms}$ query latency across 500+ tools.
3. **Byzantine Consensus Gate**: Mandatory 2-of-3 majority approval across CODE, DOC, UX, and RELEASE modes prior to release promotion.
