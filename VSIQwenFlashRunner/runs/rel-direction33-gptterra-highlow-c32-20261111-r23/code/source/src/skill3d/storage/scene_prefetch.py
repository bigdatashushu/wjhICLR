"""One eligible scheduler lookahead; transport only, with observable failures."""

from concurrent.futures import ThreadPoolExecutor


class ScenePrefetch:
    def __init__(self, runtime, events):
        self.runtime, self.events = runtime, events
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eligible-scene")
        self.future = None
        self.scene = None
        self.ready = set()
        self.attempt = None

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future, scene = self.future, self.scene
        self.future, self.scene = None, None
        try:
            receipt = future.result()
        except BaseException as exc:
            self.events.append({"kind": "prefetch_failed", "scene": scene,
                                "error_type": type(exc).__name__})
            raise
        self.events.append({"kind": "prefetch_completed", **receipt})
        if receipt["status"] == "ready":
            self.ready.add(scene)

    def consider(self, row, *, active_scenes, completed_count):
        self.poll()
        if row is None or self.future is not None:
            return
        scene = str(row["scene_id"])
        attempt = (scene, completed_count)
        if scene in active_scenes or scene in self.ready or attempt == self.attempt:
            return
        self.attempt = attempt
        self.scene = scene
        self.events.append({"kind": "prefetch_started", "scene": scene,
                            "variant_id": row["variant_id"]})
        self.future = self.pool.submit(self.runtime.prefetch, scene)

    def close(self):
        # A stop never abandons a transfer or silently discards its integrity error.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.poll()
