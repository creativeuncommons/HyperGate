# Changelog

All notable changes to **HyperGate** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
* **Inverted Index & BM25 Scoring for Dynamic Tool Discovery (`ToolPruner`)**:
  * Implemented an in-memory inverted token index with Okapi BM25 term weighting ($k_1 = 1.5, b = 0.75$).
  * Added field weighting across tool names ($3.5\times$), toolpack tags ($2.5\times$), parameters ($1.5\times$), and descriptions ($1.0\times$).
  * Supports high-throughput instantiated catalogs (`pruner = ToolPruner(catalog)`) and backward-compatible classmethod calls (`ToolPruner.prune_tools`).
* **Multi-Turn Agent Context Analysis**:
  * Evaluates conversational message arrays, weighting tokens from the most recent user turn $3.0\times$ higher than prior conversational turns to prevent goal drift.
* **Virtual Toolpacks & Domain Promotion**:
  * Tools can declare domain groupings via `toolpack` (or `tags`).
  * If a prompt strongly activates a domain, HyperGate automatically promotes the entire related toolpack into the LLM context.
* **Upstream Gateway Error Recovery (`ZeroMCPSupervisor`)**:
  * Automatic retry with exponential backoff on transient `-32603` errors (communication errors, sudden child process termination, or upstream internal errors) before bubbling up to the agent.
* **Tool Discovery Benchmark Suite (`tests/test_tool_discovery_benchmark.py`)**:
  * Evaluates latency and token savings across 100, 500, and 1,000 synthetic tools.
  * Asserts strict $< 1.0\text{ ms}$ latency invariant across 500+ tools (verified at $0.194\text{ ms}$, saving $> 95\%$ context tokens).

---

## [0.1.0] - 2026-11-10

### Initial Release: System-One Intelligent Inference Router & Circuit-Breaker Gateway

#### ⚡ Core Capabilities
- **System-One Fast Path (The 90/10 Split)**:
  - Sub-microsecond heuristic classification engine categorizing requests across 4 distinct routing tiers:
    - `TIER_0_CACHE`: In-memory zero-cost semantic cache hit ($0\text{ ms}$, $\$0.00$).
    - `TIER_1_FAST`: Ultra-low-latency SLMs and flash-lite models for deterministic and repetitive tasks.
    - `TIER_2_BALANCED`: Workhorse models (Sonnet 3.5, Gemini 2.0 Flash, GPT-4o) for standard agent tasks.
    - `TIER_3_REASONING`: Frontier deep reasoning (o3-mini, o1, Opus 3.5) for multi-step logic.
- **Deterministic Cost Kill-Switch**:
  - Enforces zero-drift budget ceilings (default `$0.50` per task loop) and turn limits (default 30 turns).
  - Pre-flight estimation halts runaway infinite tool loops before burning upstream tokens.
  - Returns canonical HTTP `429 Too Many Requests` with explicit `circuit_breaker_error` JSON payload.
- **Protocol Header Contract**:
  - Implements canonical telemetry headers:
    - `x-hypergate-trace-id`
    - `x-hypergate-routing-tier`
    - `x-hypergate-latency-overhead-ms`
    - `x-hypergate-cost-usd`
    - `x-hypergate-rolling-cost-usd`
    - `x-hypergate-circuit-tripped`
- **Zero-Dependency Python SDK (`hypergate-py`)**:
  - Pure Python standard library (`urllib.request`, `json`, `dataclasses`, `time`, `threading`).
  - Supports both direct in-process router dispatch and network proxying.
  - Native `ZeroMCPSupervisor` bridge for Model Context Protocol stdio servers.
- **High-Performance Rust Core (`hypergate-rs`)**:
  - Built with `axum`, `tokio`, `dashmap`, and `serde`.
  - Sub-millisecond added routing overhead ($< 1.0\text{ ms}$).
  - Multi-stage Docker containerization for minimal RSS footprints.
- **Incubator Tri-Model Consensus Gate**:
  - Automated 2-of-3 majority peer-review gate spanning Claude Opus 5.5, Codex Astra, and Gemini 4 Argon.
  - One-click promotion pipeline (`scripts/promote_to_standalone.py`) ensuring clean separation of incubator governance from standalone production release.
