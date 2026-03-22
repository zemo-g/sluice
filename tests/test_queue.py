"""Tests for sluice priority queue and core logic."""

import threading
import time
import unittest

from sluice.server import LLMQueue, QueueFullError, QueueTimeoutError, _strip_think_tags
from sluice.config import P_CRITICAL, P_HIGH, P_MEDIUM, P_LOW, P_BG


class TestPriorityQueue(unittest.TestCase):
    """Test the core LLMQueue serialization and priority ordering."""

    def setUp(self):
        self.queue = LLMQueue()

    def tearDown(self):
        self.queue.shutdown()

    def test_basic_submit(self):
        """Queue executes a simple function and returns the result."""
        result = self.queue.submit(P_MEDIUM, lambda: 42)
        self.assertEqual(result, 42)

    def test_serialization(self):
        """Concurrent submits execute one at a time (no overlap)."""
        running = []
        lock = threading.Lock()

        def task(name):
            with lock:
                self.assertEqual(len(running), 0, f"Overlap detected: {running}")
                running.append(name)
            time.sleep(0.05)
            with lock:
                running.remove(name)
            return name

        results = [None, None, None]

        def submit(idx, name, priority):
            results[idx] = self.queue.submit(priority, task, name)

        threads = [
            threading.Thread(target=submit, args=(0, "A", P_MEDIUM)),
            threading.Thread(target=submit, args=(1, "B", P_MEDIUM)),
            threading.Thread(target=submit, args=(2, "C", P_MEDIUM)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(sorted(results), ["A", "B", "C"])

    def test_priority_ordering(self):
        """Higher priority requests execute before lower priority ones."""
        order = []

        # Block the queue with a slow task
        blocker = threading.Event()

        def slow():
            blocker.wait(timeout=5)
            return "slow"

        def record(name):
            order.append(name)
            return name

        # Submit blocker first
        t0 = threading.Thread(
            target=lambda: self.queue.submit(P_MEDIUM, slow)
        )
        t0.start()
        time.sleep(0.05)  # Let blocker start executing

        # Now submit low, then high — high should execute first after blocker
        t_low = threading.Thread(
            target=lambda: self.queue.submit(P_LOW, record, "low")
        )
        t_high = threading.Thread(
            target=lambda: self.queue.submit(P_HIGH, record, "high")
        )
        t_low.start()
        time.sleep(0.02)
        t_high.start()
        time.sleep(0.02)

        # Release blocker
        blocker.set()
        t0.join(timeout=5)
        t_high.join(timeout=5)
        t_low.join(timeout=5)

        self.assertEqual(order, ["high", "low"])

    def test_error_propagation(self):
        """Exceptions in queued functions propagate to the caller."""
        with self.assertRaises(ValueError):
            self.queue.submit(P_MEDIUM, lambda: (_ for _ in ()).throw(ValueError("boom")))

    def test_stats(self):
        """Queue tracks served count and uptime."""
        self.queue.submit(P_MEDIUM, lambda: "ok")
        self.queue.submit(P_MEDIUM, lambda: "ok")
        status = self.queue.status()
        self.assertEqual(status["total_served"], 2)
        self.assertGreaterEqual(status["uptime_s"], 0)
        self.assertEqual(status["total_rejected"], 0)


class TestThinkTagStripping(unittest.TestCase):
    """Test the <think> tag removal for Qwen3/R1 models."""

    def test_complete_think_block(self):
        text = "<think>Let me reason about this...</think>\nThe answer is 42."
        self.assertEqual(_strip_think_tags(text), "The answer is 42.")

    def test_no_think_tags(self):
        text = "Just a normal response."
        self.assertEqual(_strip_think_tags(text), "Just a normal response.")

    def test_truncated_think_block(self):
        text = "<think>This got cut off mid-thought\nsome reasoning\nThe answer is 42"
        result = _strip_think_tags(text)
        self.assertEqual(result, "The answer is 42")

    def test_empty_after_strip(self):
        text = "<think>All thinking, no answer.</think>"
        self.assertEqual(_strip_think_tags(text), "")

    def test_multiline_think(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>\nFinal answer."
        self.assertEqual(_strip_think_tags(text), "Final answer.")


class TestClientDegradation(unittest.TestCase):
    """Test that the client returns safe defaults when server is down."""

    def test_query_returns_empty_on_failure(self):
        from sluice.client import SluiceClient
        client = SluiceClient(base_url="http://localhost:1")  # nothing listening
        result = client.query("fast", "test")
        self.assertEqual(result, "")

    def test_health_returns_down_on_failure(self):
        from sluice.client import SluiceClient
        client = SluiceClient(base_url="http://localhost:1")
        health = client.health()
        self.assertEqual(health["status"], "down")


if __name__ == "__main__":
    unittest.main()
