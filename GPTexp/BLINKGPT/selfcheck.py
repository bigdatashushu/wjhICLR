"""Meaningful offline checks for retries, exact replay and slot exclusion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import time
import unittest

from .common import parse_official
from .transport import APIError, Client, Slots, backoff

CONFIG = {"timeout_seconds": 36000, "attempts_per_request": 12,
          "backoff_min_seconds": 0.001, "backoff_max_seconds": 0.002}
MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Choose A or B."}]}]


def response(text="A"):
    return {"text": text, "usage": {}, "generation": {"finish_reason": "stop"},
            "response_identity": {"model": "gpt-5.6-terra", "model_reported": True},
            "reasoning": "", "native_response": {}}


class OfflineChecks(unittest.TestCase):
    def test_identical_retry_and_durable_replay(self):
        with tempfile.TemporaryDirectory() as root:
            slots = Slots(root + "/slots", {"one": "private-one", "two": "private-two"})
            calls = []
            def send(payload, key):
                calls.append((payload.copy(), key))
                if len(calls) == 1:
                    raise APIError("http_error", 503)
                return response()
            c = Client(CONFIG, root + "/requests", slots, "job", 1, send=send)
            self.assertEqual(c.chat(MESSAGES), "A")
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], calls[1][0])
            self.assertNotEqual(calls[0][1], calls[1][1])
            c = Client(CONFIG, root + "/requests", slots, "job", 1,
                       send=lambda *args: self.fail("replay sent a new model request"))
            self.assertEqual(c.chat(MESSAGES), "A")
            self.assertEqual(c.replayed, 1)
            c = Client(CONFIG, root + "/requests", slots, "job", 1)
            with self.assertRaisesRegex(APIError, "resume_payload_identity_changed"):
                c.chat([{"role": "user", "content": "changed"}])

    def test_attempt_cap_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            slots = Slots(root + "/slots", {"one": "private-one"})
            calls = []
            def send(*args):
                calls.append(1)
                raise APIError("http_error", 503)
            config = {**CONFIG, "attempts_per_request": 2}
            for _ in range(2):
                c = Client(config, root + "/requests", slots, "job", 1, send=send)
                with self.assertRaisesRegex(APIError, "transport_attempt_budget_exhausted"):
                    c.chat(MESSAGES)
            self.assertEqual(len(calls), 2)

    def test_slot_exclusion_under_concurrency(self):
        with tempfile.TemporaryDirectory() as root:
            slots = Slots(root, {"one": "private-one", "two": "private-two"})
            active, maximum, guard = {}, {}, threading.Lock()
            def task(i):
                with slots.acquire(i % 2) as (api, key, state):
                    with guard:
                        active[api] = active.get(api, 0) + 1
                        maximum[api] = max(maximum.get(api, 0), active[api])
                    time.sleep(0.005)
                    with guard:
                        active[api] -= 1
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(task, range(12)))
            self.assertTrue(maximum)
            self.assertTrue(all(n == 1 for n in maximum.values()))

    def test_backoff_and_label_parser(self):
        options = [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}]
        self.assertEqual(parse_official("A", options)[0], "A")
        self.assertEqual(parse_official("Some reasoning.\nFinal answer: (B) no.", options)[0], "B")
        self.assertIsNone(parse_official("A or B could both be possible.", options)[0])
        self.assertEqual(backoff(1, CONFIG, 20), 20)


if __name__ == "__main__":
    unittest.main()
