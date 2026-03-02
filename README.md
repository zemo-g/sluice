# sluice

Priority-queue LLM router for Apple Silicon. Prevents GPU OOM by serializing local inference.

## The Problem

Run two MLX models on a Mac and hit them concurrently:

```
[metal::malloc] Resource limit exceeded
Segmentation fault: 11
```

Apple's unified memory has no process isolation for GPU allocations. Two inference requests overlapping on the same Metal device = OOM kernel panic or silent corruption. This is [documented](https://github.com/ml-explore/mlx-examples/issues/831) and [unresolved](https://github.com/ml-explore/mlx-examples/issues/1198) in mlx-lm.

## The Fix

A priority queue that serializes GPU access. One request at a time. Higher-priority requests jump the line.

```
                    ┌─────────────────────────────┐
                    │         sluice :5590         │
                    │                              │
                    │  ┌────────────────────────┐  │
  POST /v1/query ──>│  │   Priority Queue       │  │
  {"model":"fast"}  │  │   ┌──┐┌──┐┌──┐┌──┐    │  │──> :8081 fast model
                    │  │   │P0││P1││P2││P4│    │  │
  POST /v1/query ──>│  │   └──┘└──┘└──┘└──┘    │  │──> :8080 reasoning model
  {"model":"reason"}│  └────────────────────────┘  │
                    │                              │
  POST /v1/query ──>│  cloud bypass (no queue) ────│──> Anthropic API
  {"model":"cloud"} │                              │
                    └─────────────────────────────┘
```

- **Local models**: queued, serialized, one-at-a-time. No concurrent GPU access.
- **Cloud model**: bypasses the queue entirely. No GPU involved.
- **Priority levels**: P_CRITICAL(0) through P_BG(4). Critical requests preempt background work.

## Quickstart

```bash
pip install sluice-llm

# Start your MLX servers (or any OpenAI-compatible backend)
mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080 &
mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --port 8081 &

# Start sluice
sluice
# Listening on http://0.0.0.0:5590

# Query
curl -X POST localhost:5590/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"model":"fast","prompt":"What is 2+2?"}'
```

### Python client

```python
from sluice import sluice, P_CRITICAL

# Simple query (returns "" on any failure — never crashes)
result = sluice.query("reasoning", "Explain quicksort", priority=P_CRITICAL)

# Cloud query (requires ANTHROPIC_API_KEY + pip install sluice-llm[cloud])
result = sluice.query("cloud", "Review this code", system="You are a code reviewer.")

# Health check
health = sluice.health()
# {"status": "ok", "models_ready": 2, "queue": {"pending": 0, ...}}
```

## Configuration

All settings via environment variables. No config files.

| Variable | Default | Description |
|----------|---------|-------------|
| `SLUICE_PORT` | `5590` | Server port |
| `SLUICE_REASONING_URL` | `http://localhost:8080/v1` | Reasoning model endpoint |
| `SLUICE_REASONING_MODEL` | `mlx-community/Qwen3-8B-4bit` | Reasoning model ID |
| `SLUICE_FAST_URL` | `http://localhost:8081/v1` | Fast model endpoint |
| `SLUICE_FAST_MODEL` | `mlx-community/Qwen3-1.7B-4bit` | Fast model ID |
| `SLUICE_CLOUD_MODEL` | `claude-sonnet-4-6` | Cloud model (Anthropic) |
| `SLUICE_LOG_DIR` | `~/.sluice/logs` | Log directory |
| `SLUICE_MAX_QUEUE_DEPTH` | `20` | Max queued requests before rejecting |
| `SLUICE_MAX_TOKENS` | `8192` | Max output tokens |
| `SLUICE_MAX_PROMPT_CHARS` | `48000` | Max prompt length |
| `ANTHROPIC_API_KEY` | — | Enables cloud model (optional) |

## API

### `GET /v1/health`

```json
{
  "status": "ok",
  "models_ready": 2,
  "models_total": 2,
  "backends": {"http://localhost:8080/v1": "ready", "http://localhost:8081/v1": "ready"},
  "model_status": {"reasoning": "ready", "fast": "ready", "cloud": "api"},
  "queue": {"pending": 0, "active": false, "total_served": 142, "avg_latency_ms": 3400}
}
```

### `POST /v1/query`

```json
{
  "model": "reasoning",
  "prompt": "Your prompt here",
  "system": "Optional system prompt",
  "max_tokens": 2048,
  "temperature": 0.3,
  "priority": 2
}
```

Response:
```json
{"status": "ok", "source": "local", "result": "The model's response..."}
```

Status codes: `200` ok, `400` bad request, `503` queue full, `504` timeout.

### `GET /v1/queue/status`

```json
{
  "pending": 3,
  "active": true,
  "active_elapsed_s": 4.2,
  "total_served": 142,
  "total_rejected": 0,
  "total_timeouts": 1,
  "avg_latency_ms": 3400,
  "uptime_s": 86400
}
```

## How It Works

### Priority Queue

Requests enter a min-heap sorted by `(priority, sequence_number)`. A single worker thread pops one request at a time and executes it. This guarantees:

1. **No concurrent GPU access** — impossible to OOM from request overlap
2. **Priority ordering** — critical requests execute before background tasks
3. **Fairness** — same-priority requests execute in FIFO order
4. **Backpressure** — queue full (503) tells callers to back off

### Model Routing

Four aliases map to backends:

| Alias | Backend | Use Case |
|-------|---------|----------|
| `reasoning` | Local (default :8080) | Deep analysis, CoT reasoning |
| `fast` | Local (default :8081) | Classification, parsing, ~0.3s |
| `tiny` | Alias for `fast` | Same as fast |
| `cloud` | Anthropic API | Heavy reasoning, bypasses queue |

### Degradation Contract

Every client method returns safe defaults on failure:

| Situation | `query()` returns | `health()` returns |
|-----------|-------------------|--------------------|
| Server down | `""` | `{"status": "down"}` |
| Server warming | `""` | `{"status": "warming"}` |
| Queue full | `""` | — |
| Backend crashed | `""` | `{"status": "degraded"}` |

Your service never crashes because sluice is unavailable.

## Works With

Any OpenAI-compatible inference server:

- **[mlx-lm](https://github.com/ml-explore/mlx-examples)** — Apple Silicon native (what sluice was built for)
- **[ollama](https://ollama.com)** — set `SLUICE_REASONING_URL=http://localhost:11434/v1`
- **[vLLM](https://github.com/vllm-project/vllm)** — high-throughput serving
- **[TGI](https://github.com/huggingface/text-generation-inference)** — HuggingFace serving

Cloud backend (optional): Anthropic Claude via SDK or CLI fallback.

## vs. Alternatives

| Feature | sluice | LiteLLM | Ollama | RouteLLM |
|---------|--------|---------|--------|----------|
| GPU OOM prevention | Yes (serialized queue) | No | No | No |
| Priority queue | Yes (5 levels) | No | No | No |
| Dependencies | 0 (stdlib only) | FastAPI+Redis+Docker | Go binary | torch+transformers |
| Apple Silicon focus | Yes | No | Partial | No |
| Cloud fallback | Optional | Yes | No | Yes |
| Lines of code | ~600 | ~50K | ~100K | ~5K |

## Origin

Built from 30 days of running dual MLX models 24/7 on a Mac Mini M4 Pro (24GB). The priority queue started as 100 lines to stop kernel panics. It grew into a router because every service needed the same thing: "query a model, get text back, don't crash."

More about the project and the infrastructure it came from: [ledatic.org](https://ledatic.org)

## License

MIT
