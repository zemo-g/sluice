#!/usr/bin/env python3
"""sluice — priority-queue LLM router for Apple Silicon.

Serializes GPU inference to prevent OOM crashes. Dual-model local backend
(reasoning + fast) with optional cloud fallback. Zero required dependencies.

Usage:
    sluice                                    # Start server (if installed)
    python3 -m sluice.server                  # Start server (from source)
    curl localhost:5590/v1/health             # Health check
    curl -X POST localhost:5590/v1/query -d '{"model":"fast","prompt":"Say hi"}'
"""

import heapq
import itertools
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from sluice.config import (
    PORT, LOG_DIR,
    REASONING_URL, REASONING_MODEL, FAST_URL, FAST_MODEL,
    CLOUD_MODEL, CLOUD_TIMEOUT, CLAUDE_BIN,
    MODELS, MAX_TOKENS_LIMIT, MAX_PROMPT_CHARS,
    THINK_MIN_TOKENS, FAST_MIN_TOKENS,
    DEFAULT_TIMEOUT, MAX_RETRIES,
    P_CRITICAL, P_HIGH, P_MEDIUM, P_LOW, P_BG,
    MAX_QUEUE_DEPTH, QUEUE_WAIT_TIMEOUT,
)

try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None


# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("sluice")
log.setLevel(logging.INFO)

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "sluice.log", maxBytes=10_000_000, backupCount=3,
)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_fh)

_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
log.addHandler(_sh)


# ─── Priority Queue ─────────────────────────────────────────────────────────

class QueueFullError(Exception):
    pass

class QueueTimeoutError(Exception):
    pass


@dataclass
class _Req:
    priority: int
    seq: int
    fn: Callable
    args: tuple
    kwargs: dict
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[Exception] = None
    submitted_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        return (self.priority, self.seq) < (other.priority, other.seq)


class LLMQueue:
    """Serialized priority queue — one request at a time, prevents GPU OOM."""

    def __init__(self):
        self._heap: list[_Req] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._seq = itertools.count()
        self._running = True
        self._active: Optional[_Req] = None

        # Stats
        self._served = 0
        self._rejected = 0
        self._timeouts = 0
        self._errors = 0
        self._latency_ms = 0
        self._started_at = time.time()

        threading.Thread(target=self._run, daemon=True, name="llm-q").start()
        log.info("LLM queue started (depth=%d)", MAX_QUEUE_DEPTH)

    def submit(self, priority: int, fn: Callable, *args,
               queue_timeout: float = QUEUE_WAIT_TIMEOUT, **kwargs) -> Any:
        """Submit and block until done. Raises QueueFullError or QueueTimeoutError."""
        with self._lock:
            if len(self._heap) >= MAX_QUEUE_DEPTH:
                self._rejected += 1
                raise QueueFullError("queue full")
            req = _Req(priority, next(self._seq), fn, args, kwargs)
            heapq.heappush(self._heap, req)
            self._cv.notify()

        if not req.event.wait(timeout=queue_timeout):
            self._timeouts += 1
            raise QueueTimeoutError(f"timeout after {queue_timeout:.0f}s")
        if req.error:
            raise req.error
        return req.result

    def status(self) -> dict:
        with self._lock:
            pending = len(self._heap)
            active = self._active is not None
            elapsed = (time.time() - self._active.submitted_at) if self._active else 0
        return {
            "pending": pending,
            "active": active,
            "active_elapsed_s": round(elapsed, 1),
            "total_served": self._served,
            "total_rejected": self._rejected,
            "total_timeouts": self._timeouts,
            "total_errors": self._errors,
            "avg_latency_ms": round(self._latency_ms / max(self._served, 1)),
            "uptime_s": round(time.time() - self._started_at),
        }

    def _run(self):
        while self._running:
            with self._lock:
                while not self._heap and self._running:
                    self._cv.wait(5)
                if not self._running:
                    break
                req = heapq.heappop(self._heap)
                self._active = req
            try:
                req.result = req.fn(*req.args, **req.kwargs)
            except Exception as e:
                req.error = e
                self._errors += 1
                log.warning("Queue request failed: %s", e)
            finally:
                self._latency_ms += int((time.time() - req.submitted_at) * 1000)
                self._served += 1
                with self._lock:
                    self._active = None
                req.event.set()

    def shutdown(self):
        self._running = False
        with self._lock:
            self._cv.notify_all()


# ─── MLX Interface ───────────────────────────────────────────────────────────

def _mlx_available(api_url: str, timeout: int = 5) -> bool:
    """Check if an OpenAI-compatible server is reachable."""
    try:
        req = Request(f"{api_url}/models")
        resp = urlopen(req, timeout=timeout)
        return resp.status == 200
    except (URLError, OSError):
        return False


def _mlx_post(api_url: str, payload: dict, timeout: int = 120) -> dict:
    """POST to an OpenAI-compatible server. Returns parsed JSON response."""
    data = json.dumps(payload).encode()
    req = Request(
        f"{api_url}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks from R1/Qwen3 thinking output.
    If think block is truncated (no closing tag), extract last paragraph as best-effort."""
    # Complete think blocks
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Truncated think block - model hit max_tokens mid-thought
    if "<think>" in cleaned and "</think>" not in cleaned:
        lines = [ln.strip() for ln in cleaned.splitlines()
                 if ln.strip() and not ln.strip().startswith("<think")]
        return lines[-1] if lines else ""
    return cleaned.strip()


def _resolve(alias: str) -> tuple:
    """Resolve alias -> (api_url, model_id, inject_think). Falls back to reasoning."""
    key = alias.lower().strip()
    if key in MODELS:
        return MODELS[key]
    return MODELS["reasoning"]


# Track backend status (updated during warmup + periodic health)
_backend_status = {REASONING_URL: False, FAST_URL: False}


def query_local(
    model: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.3,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES,
    json_mode: bool = False,
) -> str:
    """Query a local model. Returns response text or 'ERROR: ...'."""
    api_url, model_id, inject_think = _resolve(model)

    # Thinking mode: bump min tokens so model has room for <think> + answer
    if inject_think is True and max_tokens < THINK_MIN_TOKENS:
        max_tokens = THINK_MIN_TOKENS
    elif inject_think == "fast" and max_tokens < FAST_MIN_TOKENS:
        max_tokens = FAST_MIN_TOKENS

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Disable Qwen3 native thinking — faster, no <think> token waste
    payload["chat_template_kwargs"] = {"enable_thinking": False}

    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            data = _mlx_post(api_url, payload, timeout=timeout)
            elapsed = time.time() - t0

            content = data["choices"][0]["message"]["content"]
            content = _strip_think_tags(content)

            tokens_out = (data.get("usage") or {}).get("completion_tokens", "?")
            log.info("[%s] %s -> %s tok in %.1fs", model, model_id, tokens_out, elapsed)
            return content

        except URLError:
            if attempt < retries:
                time.sleep(2)
                continue
            return f"ERROR: Server unreachable at {api_url}"
        except OSError as e:
            if "timed out" in str(e):
                if attempt < retries:
                    continue
                return f"ERROR: {model_id} timed out after {timeout}s"
            if attempt < retries:
                time.sleep(1)
                continue
            return f"ERROR: {e}"
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return f"ERROR: Bad response — {e}"
        except Exception as e:
            return f"ERROR: {e}"


# ─── Cloud Interface ─────────────────────────────────────────────────────────

_cloud_client = None
_cloud_client_lock = threading.Lock()


def _get_cloud_client():
    """Lazy singleton for Anthropic SDK client."""
    global _cloud_client
    if _cloud_client is not None:
        return _cloud_client
    with _cloud_client_lock:
        if _cloud_client is not None:
            return _cloud_client
        if _anthropic_mod is None:
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        _cloud_client = _anthropic_mod.Anthropic(api_key=api_key)
        log.info("Anthropic SDK client initialized")
        return _cloud_client


def query_cloud(
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    timeout: int = CLOUD_TIMEOUT,
    tools: list = None,
    tool_choice: dict = None,
    cache_system: bool = False,
) -> str:
    """Query Claude. Uses SDK if available, else CLI fallback.

    Returns response text, or JSON envelope with tool_calls if tools are used.
    """
    client = _get_cloud_client()
    if client:
        return _query_cloud_sdk(
            client, prompt, system, max_tokens, timeout,
            tools=tools, tool_choice=tool_choice, cache_system=cache_system,
        )
    return _query_cloud_cli(prompt, system, max_tokens, timeout)


def _query_cloud_sdk(
    client,
    prompt: str,
    system: str,
    max_tokens: int,
    timeout: int,
    tools: list = None,
    tool_choice: dict = None,
    cache_system: bool = False,
) -> str:
    """Query Claude via Anthropic SDK. Returns response text or 'ERROR: ...'."""
    kwargs: dict = {
        "model": CLOUD_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        if cache_system:
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    try:
        t0 = time.time()
        response = client.messages.create(**kwargs)
        elapsed = time.time() - t0

        # Extract text and tool_use blocks
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Log cache stats if available
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        cache_info = ""
        if cache_read or cache_create:
            cache_info = f" cache_read={cache_read} cache_create={cache_create}"

        text = "\n".join(text_parts)

        if tool_calls:
            log.info(
                "[cloud/sdk] %s -> %d chars + %d tool_calls in %.1fs (in=%d out=%d%s)",
                CLOUD_MODEL, len(text), len(tool_calls), elapsed,
                input_tok, output_tok, cache_info,
            )
            return json.dumps({"text": text, "tool_calls": tool_calls})
        else:
            log.info(
                "[cloud/sdk] %s -> %d chars in %.1fs (in=%d out=%d%s)",
                CLOUD_MODEL, len(text), elapsed,
                input_tok, output_tok, cache_info,
            )
            return text

    except _anthropic_mod.APIStatusError as e:
        return f"ERROR: API {e.status_code}: {e.message}"
    except _anthropic_mod.APIConnectionError as e:
        return f"ERROR: Cloud API connection failed: {e}"
    except _anthropic_mod.APITimeoutError:
        log.warning("Cloud SDK timed out after %ds", timeout)
        return f"ERROR: Cloud query timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: Cloud SDK failed: {e}"


def _query_cloud_cli(
    prompt: str,
    system: str,
    max_tokens: int,
    timeout: int,
) -> str:
    """Query Claude via CLI subprocess (fallback). Returns response text or 'ERROR: ...'."""
    if not CLAUDE_BIN.exists():
        return f"ERROR: Claude CLI not found at {CLAUDE_BIN}"

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    try:
        t0 = time.time()
        result = subprocess.run(
            [str(CLAUDE_BIN), "-p", "--model", CLOUD_MODEL, full_prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:200]
            return f"ERROR: Claude CLI exit code {result.returncode}: {stderr}"

        content = result.stdout.strip()
        if not content:
            return "ERROR: Claude CLI returned empty response"

        log.info("[cloud/cli] %s -> %d chars in %.1fs", CLOUD_MODEL, len(content), elapsed)
        return content

    except subprocess.TimeoutExpired:
        log.warning("Cloud query timed out after %ds", timeout)
        return f"ERROR: Cloud query timed out after {timeout}s"
    except FileNotFoundError:
        return f"ERROR: Claude CLI not found at {CLAUDE_BIN}"
    except OSError as e:
        return f"ERROR: Cloud query failed: {e}"


def query_json(model: str, prompt: str, system: str = "", **kwargs):
    """Query and parse JSON. Returns dict/list or None on failure."""
    raw = query_local(model, prompt, system, json_mode=True, temperature=0.1, **kwargs)
    if raw.startswith("ERROR:"):
        log.warning(raw)
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("JSON parse failed: %s", raw[:300])
        return None


# ─── Warmup ──────────────────────────────────────────────────────────────────

_queue: Optional[LLMQueue] = None

_warmup = {
    "warming": True,
    "backends": {REASONING_URL: "pending", FAST_URL: "pending"},
    "ready": 0,
    "total": 2,
}
_warmup_lock = threading.Lock()


def _wait_for_backend(api_url: str, max_retries: int = 15, backoff: float = 2.0) -> bool:
    """Wait for a backend to come online."""
    for attempt in range(max_retries):
        if _mlx_available(api_url):
            return True
        if attempt < max_retries - 1:
            wait = backoff * (2 ** min(attempt, 4))
            log.info("Waiting for %s (attempt %d/%d, %.0fs)...",
                     api_url, attempt + 1, max_retries, wait)
            time.sleep(wait)
    return False


def _warmup_one(api_url: str, model_id: str, alias: str) -> bool:
    """Wait for one backend and mark it ready. Returns True if reachable."""
    with _warmup_lock:
        _warmup["backends"][api_url] = "loading"

    connected = _wait_for_backend(api_url, max_retries=15, backoff=2)
    _backend_status[api_url] = connected

    if not connected:
        log.warning("Warmup: %s unreachable", api_url)
        with _warmup_lock:
            _warmup["backends"][api_url] = "failed"
        return False

    with _warmup_lock:
        _warmup["backends"][api_url] = "ready"
        _warmup["ready"] += 1
    log.info("Warmup: %s backend ready", model_id)
    return True


def _warmup_thread():
    """Background: connect to backends, start queue."""
    global _queue
    log.info("Warmup: connecting to backends...")

    ok_reasoning = _warmup_one(REASONING_URL, REASONING_MODEL, "reasoning")
    _warmup_one(FAST_URL, FAST_MODEL, "fast")

    with _warmup_lock:
        _warmup["warming"] = False

    # Start queue if ANY local backend is up
    if ok_reasoning or _backend_status.get(FAST_URL):
        _queue = LLMQueue()
        log.info("Warmup complete: %d/%d backends ready, queue active",
                 _warmup["ready"], _warmup["total"])
    else:
        log.error("Warmup failed: no local backends available")


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        log.debug("HTTP: %s", fmt % args)

    def _json(self, data: dict, status: int = 200):
        try:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _queued(self, priority, fn, *args, queue_timeout=180, **kwargs):
        """Execute through LLM queue. Falls back to direct call during warmup."""
        if _queue is None:
            return fn(*args, **kwargs)
        return _queue.submit(priority, fn, *args, queue_timeout=queue_timeout, **kwargs)

    def _safe(self, fn):
        try:
            return fn()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            log.error("Request error: %s\n%s", e, traceback.format_exc())
            self._json({"status": "degraded", "error": str(e)}, 500)

    # ─── Routing ─────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/v1/health":
            self._handle_health()
        elif self.path == "/v1/queue/status":
            self._handle_queue_status()
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        routes = {
            "/v1/query": self._handle_query,
        }
        handler = routes.get(self.path)
        if handler:
            self._safe(handler)
        else:
            self._json({"error": "Not found"}, 404)

    # ─── Endpoints ───────────────────────────────────────────────────────

    def _handle_health(self):
        with _warmup_lock:
            w = dict(_warmup)
        status = "warming" if w["warming"] else ("ok" if w["ready"] > 0 else "degraded")
        resp = {
            "status": status,
            "backends": w["backends"],
            "models_ready": w["ready"],
            "models_total": w["total"],
            "model_status": {
                "reasoning": w["backends"].get(REASONING_URL, "pending"),
                "fast": w["backends"].get(FAST_URL, "pending"),
                "cloud": "api" if os.environ.get("ANTHROPIC_API_KEY") else ("cli" if CLAUDE_BIN.exists() else "unavailable"),
            },
        }
        if _queue is not None:
            resp["queue"] = _queue.status()
        self._json(resp)

    def _handle_queue_status(self):
        if _queue is None:
            self._json({"status": "not_initialized", "pending": 0})
            return
        self._json(_queue.status())

    def _handle_query(self):
        body = self._body()
        model = body.get("model", "fast")
        prompt = body.get("prompt", "")
        system = body.get("system", "")
        max_tokens = min(body.get("max_tokens", 2048), MAX_TOKENS_LIMIT)
        temperature = body.get("temperature", 0.3)
        query_timeout = min(body.get("timeout", 120), 300)
        priority = body.get("priority", P_MEDIUM)

        if not prompt:
            self._json({"error": "Missing 'prompt'"}, 400)
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._json({"error": f"Prompt exceeds {MAX_PROMPT_CHARS} char limit"}, 400)
            return

        # Cloud queries bypass GPU queue entirely
        if model.lower().strip() == "cloud":
            tools = body.get("tools")
            tool_choice = body.get("tool_choice")
            cache_system = body.get("cache_system", False)

            result = query_cloud(
                prompt, system, max_tokens=max_tokens,
                timeout=query_timeout,
                tools=tools, tool_choice=tool_choice,
                cache_system=cache_system,
            )
            if result.startswith("ERROR:"):
                self._json({"status": "degraded", "source": "cloud", "error": result})
            else:
                self._json({"status": "ok", "source": "cloud", "result": result})
            return

        with _warmup_lock:
            if _warmup["warming"]:
                self._json({"status": "warming", "source": "unavailable"})
                return

        try:
            result = self._queued(
                priority, query_local, model, prompt, system,
                max_tokens=max_tokens, temperature=temperature,
                timeout=query_timeout,
                queue_timeout=query_timeout + 60,
            )
        except QueueFullError:
            self._json({"status": "overloaded", "error": "queue full"}, 503)
            return
        except QueueTimeoutError:
            self._json({"status": "timeout", "error": "timed out"}, 504)
            return

        if result.startswith("ERROR:"):
            self._json({"status": "degraded", "source": "fallback", "error": result})
        else:
            self._json({"status": "ok", "source": "local", "result": result})


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("sluice starting on port %d", PORT)
    log.info("Reasoning: %s (%s) | Fast: %s (%s)",
             REASONING_URL, REASONING_MODEL, FAST_URL, FAST_MODEL)

    threading.Thread(target=_warmup_thread, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log.info("Listening on http://0.0.0.0:%d", PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        if _queue is not None:
            _queue.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
