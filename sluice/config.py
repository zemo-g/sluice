"""sluice — configuration via environment variables.

All settings have sensible defaults for Apple Silicon with mlx-lm.
Override any value via env vars prefixed with SLUICE_.
"""

import os
from pathlib import Path

# ─── Server ──────────────────────────────────────────────────────────────────

PORT = int(os.environ.get("SLUICE_PORT", "5590"))
LOG_DIR = Path(os.environ.get("SLUICE_LOG_DIR", str(Path.home() / ".sluice" / "logs")))

# ─── Local Backends (OpenAI-compatible) ──────────────────────────────────────

REASONING_URL = os.environ.get("SLUICE_REASONING_URL", "http://localhost:8080/v1")
REASONING_MODEL = os.environ.get("SLUICE_REASONING_MODEL", "mlx-community/Qwen3-8B-4bit")

FAST_URL = os.environ.get("SLUICE_FAST_URL", "http://localhost:8081/v1")
FAST_MODEL = os.environ.get("SLUICE_FAST_MODEL", "mlx-community/Qwen3-1.7B-4bit")

# ─── Cloud Backend (optional — requires `pip install anthropic`) ─────────────

CLOUD_MODEL = os.environ.get("SLUICE_CLOUD_MODEL", "claude-sonnet-4-6")
CLOUD_TIMEOUT = int(os.environ.get("SLUICE_CLOUD_TIMEOUT", "180"))
CLAUDE_BIN = Path(os.environ.get("SLUICE_CLAUDE_BIN", str(Path.home() / ".local" / "bin" / "claude")))

# ─── Model Routing ───────────────────────────────────────────────────────────
# alias → (api_url, model_id, inject_think)
# "cloud" is a sentinel — handled by query_cloud(), bypasses GPU queue

MODELS = {
    "reasoning": (REASONING_URL, REASONING_MODEL, True),
    "fast": (FAST_URL, FAST_MODEL, "fast"),
    "tiny": (FAST_URL, FAST_MODEL, "fast"),
    "cloud": ("cloud", CLOUD_MODEL, None),
}

# ─── Token Budgets ───────────────────────────────────────────────────────────

MAX_TOKENS_LIMIT = int(os.environ.get("SLUICE_MAX_TOKENS", "8192"))
MAX_PROMPT_CHARS = int(os.environ.get("SLUICE_MAX_PROMPT_CHARS", "48000"))
THINK_MIN_TOKENS = 2048   # Room for <think> block + answer
FAST_MIN_TOKENS = 512     # Short CoT, ~8x cheaper than reasoning

# ─── Inference ───────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 120
MAX_RETRIES = 2

# ─── Priority Levels (lower = higher priority) ──────────────────────────────

P_CRITICAL = 0   # Blocking gates
P_HIGH = 1       # Important executors
P_MEDIUM = 2     # Default
P_LOW = 3        # Background work
P_BG = 4         # Best-effort

# ─── Queue ───────────────────────────────────────────────────────────────────

MAX_QUEUE_DEPTH = int(os.environ.get("SLUICE_MAX_QUEUE_DEPTH", "20"))
QUEUE_WAIT_TIMEOUT = int(os.environ.get("SLUICE_QUEUE_TIMEOUT", "180"))
