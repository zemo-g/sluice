"""sluice client — thin HTTP client with safe degradation.

Every method returns safe defaults on any failure (connection refused, timeout,
warming, backend down). Callers never crash due to sluice being unavailable.

Usage:
    from sluice import sluice

    result = sluice.query("reasoning", "Explain quicksort")
    health = sluice.health()
"""

import json
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT = 10   # seconds for non-inference calls
QUERY_TIMEOUT = 130    # seconds for inference calls


class SluiceClient:
    """HTTP client with degradation contract. Every method returns safe defaults on failure."""

    def __init__(self, base_url: str = "http://localhost:5590"):
        self.base_url = base_url

    def _post(self, path: str, body: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
        """POST JSON to sluice server. Returns response dict or error dict."""
        try:
            data = json.dumps(body).encode()
            req = Request(
                f"{self.base_url}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError, ValueError):
            return {"status": "down"}

    def _get(self, path: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
        """GET from sluice server. Returns response dict or error dict."""
        try:
            req = Request(f"{self.base_url}{path}")
            resp = urlopen(req, timeout=timeout)
            return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError, ValueError):
            return {"status": "down"}

    def _is_available(self, resp: dict) -> bool:
        """Check if response indicates the service is ready."""
        status = resp.get("status", "")
        return status not in ("down", "warming", "degraded", "unavailable")

    def health(self) -> dict:
        """Check sluice server health. Returns dict, never raises."""
        return self._get("/v1/health", timeout=DEFAULT_TIMEOUT)

    def query(self, model: str, prompt: str, system: str = "",
              max_tokens: int = 2048, temperature: float = 0.3,
              priority: int = 2, tools: list = None,
              tool_choice: dict = None, cache_system: bool = False,
              timeout: Optional[int] = None) -> str:
        """General model query. Returns response text or ''.

        Args:
            model: Model alias — "reasoning", "fast", "tiny", or "cloud".
            prompt: The user prompt.
            system: Optional system prompt.
            max_tokens: Max output tokens.
            temperature: Sampling temperature.
            priority: Queue priority (0=critical, 1=high, 2=medium, 3=low, 4=background).
            tools: Tool definitions for Claude tool_use (cloud model only).
            tool_choice: Tool choice constraint (cloud model only).
            cache_system: Enable prompt caching on system prompt (cloud model only).
            timeout: Override HTTP timeout in seconds.
        """
        body = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "priority": priority,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        if cache_system:
            body["cache_system"] = True
        resp = self._post("/v1/query", body, timeout=timeout or QUERY_TIMEOUT)
        if not self._is_available(resp):
            return ""
        return resp.get("result", "")

    def queue_status(self) -> dict:
        """Get LLM queue status. Returns dict, never raises."""
        return self._get("/v1/queue/status", timeout=5)
