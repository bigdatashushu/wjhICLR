"""Serial owner processes with bounded, independent revision transactions."""

import copy
import signal
import threading
import time

from skill3d.evaluation.fitset_v11 import _require, fit_slot_key
from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution


def _owner_executor(runner, index, shard, jobs, messages, stop, semaphore):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _require(threading.current_thread() is threading.main_thread(),
             "reviser executor requires a main thread")
    runner._episode_semaphore = semaphore
    owner = None
    while not stop.is_set():
        job = jobs.get()
        if job is None:
            return
        operation, rows = job
        try:
            if owner is None:
                owner = FitSetWorkerEvolution(runner, index, shard)
            if operation == "recover":
                owner.recover()
            else:
                _require(operation == "ingest", "invalid owner operation")
                for variant, result in rows:
                    owner.ingest_result(variant, result)
        except BaseException as exc:
            messages.put({"kind": "reviser_error", "owner": index,
                          "error_type": type(exc).__name__, "at": time.time()})
            stop.set()
            return
        messages.put({"kind": "reviser_done", "owner": index,
                      "operation": operation, "at": time.time()})


class OwnerReviserPool:
    """The coordinator alone assigns work; no owner has two active writers."""

    def __init__(self, runner, shards, context, messages, stop, events):
        self.runner, self.messages, self.stop, self.events = runner, messages, stop, events
        self.shards = {index: shard for index, shard in enumerate(shards) if shard}
        self.cap = runner.reviser_concurrency
        self.jobs, self.processes = {}, []
        self.busy, self.recovered = {}, set()
        self.peak = 0
        for index, shard in self.shards.items():
            owner_runner = copy.copy(runner)
            slots = {fit_slot_key(row["question_type"], row["evidence_state"]) for row in shard}
            allowed = {row["qa_id"] for row in [
                *shard, *getattr(runner, "gate_variant_rows", {}).values()]
                if fit_slot_key(row["question_type"], row["evidence_state"]) in slots}
            # Offline information belongs only to this owner's learning/gate.
            owner_runner.label_metadata = {
                qa: value for qa, value in runner.label_metadata.items() if qa in allowed}
            owner_runner.gate_panels = {
                key: value for key, value in runner.gate_panels.items() if key in slots}
            jobs = context.Queue()
            process = context.Process(
                target=_owner_executor,
                args=(owner_runner, index, shard, jobs, messages, stop,
                      runner._episode_semaphore),
                name=f"fitset-reviser-{index:02d}", daemon=False)
            process.start()
            self.jobs[index] = jobs
            self.processes.append(process)

    def submit(self, index, operation, rows=()):
        _require(index not in self.busy and len(self.busy) < self.cap,
                 "owner revision concurrency exceeded")
        self.busy[index] = (operation, list(rows))
        self.events.append({"kind": "reviser_started", "owner": index,
                            "operation": operation, "case_count": len(rows),
                            "at": time.time()})
        self.peak = max(self.peak, len(self.busy))
        self.jobs[index].put((operation, list(rows)))
        self._status()

    def acknowledge(self, message):
        index = message["owner"]
        _require(index in self.busy, "unexpected reviser acknowledgement")
        operation, rows = self.busy.pop(index)
        _require(message["operation"] == operation, "reviser operation identity changed")
        if operation == "recover":
            self.recovered.add(index)
        self.events.append(dict(message))
        self._status()
        return index, rows

    def _status(self):
        self.runner._write_json_atomic(
            self.runner.root / "scheduler_receipts/reviser_pool_status.json", {
            "reviser_concurrency": self.cap,
            "executor_pids": [process.pid for process in self.processes],
            "active_owner_transactions": {
                str(index): {"operation": op, "case_count": len(rows)}
                for index, (op, rows) in self.busy.items()},
            "peak_active_owner_transactions": self.peak,
            "at": time.time(), "run_identity": self.runner.run_identity,
        })

    def close(self):
        for jobs in self.jobs.values():
            jobs.put(None)

    def finish(self):
        for process in self.processes:
            process.join()
        for jobs in self.jobs.values():
            jobs.cancel_join_thread()
            jobs.close()
