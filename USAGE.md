# HyperGate Integration Manual & Wire Protocol Specification

HyperGate is an ultra-low-latency, zero-overhead intelligent inference router and circuit-breaker gateway designed to sit between autonomous AI agents, LLM providers, and local tools.

---

## 1. Architectural Architecture & Routing Hierarchy

HyperGate classifies incoming requests dynamically into a 4-tier cascade:

```
[Agent Outbound Request]
           │
           ▼
┌────────────────────────────────────────┐
│        System-One Fast Path            │
│  Sub-microsecond Heuristic Classifier  │
└────────────────────────────────────────┘
     │            │            │            │
     ▼            ▼            ▼            ▼
[Tier 0]      [Tier 1]      [Tier 2]     [Tier 3]
Semantic      Fast SLM      Balanced     Frontier
 Cache        Flash-Lite    Workhorse    Reasoning
 (0 ms)       (<20 ms)      Sonnet/Flash o3-mini/Opus
```

### Cascade Tier Matrix

| Tier Name | Target Latency | Pricing Multiplier | Primary Models | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`TIER_0_CACHE`** | $< 1.0\text{ ms}$ | $0.0\times$ | In-memory cache | Exact repeat queries, static prompts |
| **`TIER_1_FAST`** | $< 25.0\text{ ms}$ | $\le 0.05\times$ | `gpt-4o-mini`, `gemini-2.0-flash-lite`, `local-slm` | Health checks, acknowledgments, arithmetic, schema formatting |
| **`TIER_2_BALANCED`** | $\sim 500\text{ ms}$ | $1.0\times$ | `gpt-4o`, `claude-3-5-sonnet`, `gemini-2.0-flash` | Code generation, data analysis, tool execution dispatch |
| **`TIER_3_REASONING`** | $> 2000\text{ ms}$ | $5.0\times$ | `o3-mini`, `o1`, `claude-3-opus`, `gemini-2.0-pro` | Multi-step mathematical proofs, formal logic, concurrency analysis |

---

## 2. HTTP Protocol & Header Contract

HyperGate exposes an OpenAI-compatible wire format:
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

### Canonical Telemetry Headers

Every response processed or intercepted by HyperGate injects the following custom HTTP headers:

| Header Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `x-hypergate-trace-id` | `string` | Unique session or agent task loop identifier | `trc_8f1a92e4bc01` |
| `x-hypergate-routing-tier` | `string` | Resolved cascade tier | `TIER_1_FAST` |
| `x-hypergate-latency-overhead-ms`| `float` | Routing and classification latency added by HyperGate | `0.003` |
| `x-hypergate-cost-usd` | `float` | Exact calculated request cost formatted to 6 decimal places | `0.000142` |
| `x-hypergate-rolling-cost-usd` | `float` | Cumulative trace cost for the current session loop | `0.004960` |
| `x-hypergate-circuit-tripped` | `boolean` | `true` if circuit-breaker was tripped; `false` otherwise | `false` |

---

## 3. Circuit-Breaker Invariant (The Hard Kill-Switch)

When an agent execution loop causes either:
1. $\text{Rolling Cost} + \text{Estimated Preflight Cost} > \text{Max Budget Ceiling}$ (default `\$0.50`), OR
2. $\text{Current Turns} \ge \text{Max Turns}$ (default `30` turns)

HyperGate **aborts upstream model dispatch immediately with zero tokens burned** and returns HTTP `429 Too Many Requests`:

### HTTP 429 Error Body
```json
{
  "error": {
    "code": "budget_exceeded",
    "message": "HyperGate Circuit Breaker Tripped: session budget ceiling ($0.50) exceeded. Runaway loop terminated.",
    "type": "circuit_breaker_error",
    "trace_id": "trc_8f1a92e4bc01",
    "rolling_cost_usd": 0.502140,
    "limit_usd": 0.500000
  }
}
```

---

## 4. Python SDK Integration (`hypergate-py`)

`hypergate-py` is strictly **zero external dependencies** and uses pure Python standard library (`urllib.request`, `json`, `dataclasses`, `time`, `threading`).

### Option A: Network Gateway Mode (HTTP)
```python
from hypergate import HyperGateClient, CircuitBreakerError

# Connect to running HyperGate proxy (Rust or Docker)
client = HyperGateClient(
    base_url="http://127.0.0.1:8080",
    trace_id="session_task_44a",
    budget_limit_usd=0.50
)

try:
    response = client.chat_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}]
    )
    print("Content:", response["choices"][0]["message"]["content"])
    print("Routing Tier:", client.last_response_headers.get("x-hypergate-routing-tier"))
    print("Rolling Cost:", client.rolling_cost_usd)

except CircuitBreakerError as e:
    print(f"🛑 Runaway loop terminated: {e}")
```

### Option B: In-Process Zero-Overhead Router Mode
For local agent loops where launching a separate daemon is undesirable:
```python
from hypergate import HyperGateRouter, HyperGateClient, CircuitBreakerError

# In-process router binding (3-microsecond routing latency)
router = HyperGateRouter("config/model_pricing.json")
client = HyperGateClient(router=router, budget_limit_usd=0.25)

for turn in range(50):
    try:
        resp = client.chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Turn {turn}: execute step"}]
        )
    except CircuitBreakerError as e:
        print(f"Tripped at turn {turn}: {e}")
        break
```

---

## 5. ZeroMCP Supervisor Bridge

HyperGate natively coordinates Model Context Protocol (MCP) stdio servers:

```python
from hypergate import ZeroMCPSupervisor, CircuitBreakerError

supervisor = ZeroMCPSupervisor(
    server_cmd=["python3", "0_Projects/ZeroMCP/zeromcp-py/server.py"],
    budget_limit_usd=0.10,
    max_turns=20
)

try:
    supervisor.start()
    res = supervisor.send_rpc("tools/call", {
        "name": "eval_math",
        "arguments": {"expr": "42 * 10"}
    })
    print("Tool result:", res)
except CircuitBreakerError as e:
    print(f"MCP server terminated by circuit breaker: {e}")
finally:
    supervisor.kill()
```

---

## 6. Rust Core Gateway Deployment (`hypergate-rs`)

### Running Locally
```bash
cd hypergate-rs
cargo run --release
```

### Environment Variables
- `HYPERGATE_HOST`: Bind address (default: `127.0.0.1`)
- `HYPERGATE_PORT`: HTTP port (default: `8080`)
- `HYPERGATE_CONFIG`: Path to deterministic pricing matrix (default: `config/model_pricing.json`)

### Docker Container
```bash
# Build multi-stage hardened image
docker build -t creativeuncommons/hypergate:v0.1.0 .

# Run edge container
docker run -d -p 8080:8080 creativeuncommons/hypergate:v0.1.0
```

---

## 7. Tri-Model Consensus Gate Execution (Incubator Only)

The incubator includes an automated 2-of-3 majority gate across Claude Opus 5.5, Codex Astra, and Gemini 4 Argon:

```bash
# Audit a code implementation or PR diff
python3 review_consensus_sentinel.py hypergate-py/hypergate.py --title "Python Gateway Core" --mode code

# Audit technical documentation
python3 review_consensus_sentinel.py USAGE.md --title "Usage Protocol Manual" --mode doc
```

Ballot reports are written to `scratch/consensus_reports/`.

---

## 8. Standalone Promotion Pipeline

To lift HyperGate from the monorepo incubator into a standalone open-source repository:

```bash
python3 scripts/promote_to_standalone.py \
  --dest dist_promoted/HyperGate \
  --remote git@github.com:creativeuncommons/HyperGate.git
```
This sanitizes incubator bridges, runs test suites, and initializes a clean Git repository with commit metadata and version tag `v0.1.0`.

---

## 9. Persistent Telemetry Sink & Audit CLI

HyperGate automatically records every routing decision, token attribution, latency curve, and circuit-breaker event locally with zero external network overhead.

### Storage Layout
- **Append-Only JSONL Streams**: `~/.hypergate/traces/{trace_id}.jsonl` (or `$HYPERGATE_TRACE_DIR`)
- **Relational SQLite Store**: `~/.hypergate/traces/hypergate_telemetry.db`

### Audit CLI Commands
Inspect agent execution graphs and circuit trip events:

```bash
# Display aggregate gateway metrics and tier breakdown
python3 -m hypergate.audit --summary

# List recent agent traces with total cost and circuit statuses
python3 -m hypergate.audit --list

# Turn-by-turn deep-dive into a specific agent trace
python3 -m hypergate.audit --trace <trace_id>
```

---

## 10. Dynamic Tool Pruning & Catalog Routing (`ToolPruner`)

Modern agent pipelines frequently register hundreds or thousands of tool schemas across disparate domains. Exposing the full catalog in every turn dilutes LLM attention, inflates latency, and wastes context window tokens.

`ToolPruner` is a sub-millisecond, pure standard library dynamic tool discovery and pruning engine.

### 10.1 Inverted Index & BM25 Term Weighting
`ToolPruner` constructs an in-memory inverted token index with Okapi BM25 term weighting ($k_1 = 1.5, b = 0.75$):
* **Field-Weighted Term Frequencies**:
  * Tool Name: **$3.5\times$**
  * Virtual Toolpack Tags: **$2.5\times$**
  * Parameter Schema Properties: **$1.5\times$**
  * Tool Description: **$1.0\times$**
* **Instant Postings Traversal**: By scoring only candidate tools in intersecting postings lists, pruning across **500+ tools executes in $< 0.25\text{ ms}$** (well below the $1.0\text{ ms}$ latency budget).

### 10.2 Multi-Turn Agent Context Analysis (3x Recency Weight)
In multi-turn conversations, agents often shift focus between turns (e.g. from general database inspection to a specific sports odds query).
`ToolPruner` evaluates multi-turn message arrays and weights query tokens from the **most recent user prompt $3.0\times$ higher** than tokens in earlier conversational turns, guaranteeing that immediate user intent dominates candidate ranking.

### 10.3 Tool Grouping & Virtual Toolpack Promotion
Tools can declare domain groupings via `toolpack` (or `tags`):
```json
{
  "name": "get_nba_game_odds",
  "description": "Fetches current Vegas point spreads and over/under lines",
  "toolpack": "sports_analytics",
  "inputSchema": {"type": "object", "properties": {"date": {"type": "string"}}}
}
```
When a prompt strongly activates a domain (e.g. peak score $\ge 8.0$ or aggregate score $\ge 15.0$), HyperGate **automatically promotes the entire related toolpack** into the LLM context, ensuring the model receives all co-dependent tools without manual configuration.

### 10.4 Python SDK Usage & HTTP Integration
```python
from hypergate import ToolPruner

# One-off stateless pruning
pruned_tools = ToolPruner.prune_tools(
    tools=catalog,
    prompt_or_messages=messages,
    top_k=5,
    pin_tools=["system_diagnostics"],
    enable_toolpacks=True
)

# High-throughput instantiated catalog index (<0.1 ms query latency)
pruner = ToolPruner(catalog)
selected = pruner.prune(query, top_k=5)
```
Via HTTP Gateway:
```http
POST /v1/chat/completions
x-hypergate-tools-top-k: 5
```

### 10.5 Upstream Gateway Error Recovery (`ZeroMCPSupervisor`)
`ZeroMCPSupervisor` includes automatic transient error recovery:
* **Exponential Backoff**: On transient upstream `-32603` errors or unexpected process termination, the supervisor automatically restarts the child process and retries the request with configurable exponential backoff (`max_retries=3`, `initial_backoff_sec=0.02`, `backoff_multiplier=2.0`).
* **Zero Disruption**: Transient network drops or sub-process restarts are healed transparently before bubbling up to the agent.

### 10.6 Discovery Benchmark Metrics
Run the benchmark suite:
```bash
python3 tests/test_tool_discovery_benchmark.py --benchmark
```
Verified performance:
* **100 Tools**: $0.047\text{ ms}$ query latency, **$95.5\%$ token savings**
* **500 Tools**: $0.194\text{ ms}$ query latency, **$95.8\%$ token savings** ($< 1.0\text{ ms}$ core invariant met)
* **1,000 Tools**: $0.375\text{ ms}$ query latency, **$95.9\%$ token savings**
