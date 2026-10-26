"""Spawned episode executors with coordinator-owned, version-bound streams."""

from __future__ import annotations

from collections import defaultdict
import copy
import fcntl
import json
import multiprocessing
import queue
import signal
import threading
import time

from skill3d.evaluation.fitset_v11 import (
    FitSetError, _require, _variant_key, _write_create_once, _write_json_atomic,
    content_sha256, fit_slot_key,
)
from skill3d.evolution.fitset_stream_v11 import (
    FitSetWorkerEvolution, FrozenSkillProvider,
)

EPISODE_POOL_SCHEDULER = "coordinator_owned_episode_spawn_pool_v1"


def _executor(runner, index, jobs, messages, stop, semaphore):
    """The sandbox and every episode execute on a child process main thread."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    runner._episode_semaphore = semaphore
    _require(threading.current_thread() is threading.main_thread(),
             "episode executor requires a main thread")
    while not stop.is_set():
        job = jobs.get()
        if job is None:
            return
        variant, skills, identity = job
        variant_id = str(variant["variant_id"])
        messages.put({"kind": "started", "executor": index,
                      "variant_id": variant_id, "at": time.time()})
        try:
            worker = copy.copy(runner)
            worker.skills_for_variant = FrozenSkillProvider(skills, identity)
            result = worker._run_one(variant)
        except BaseException as exc:
            # Never include exception text: a transport may contain secrets.
            messages.put({"kind": "error", "executor": index,
                          "variant_id": variant_id,
                          "error_type": type(exc).__name__})
            stop.set()
            return
        messages.put({"kind": "result", "executor": index,
                      "variant_id": variant_id, "result": result,
                      "at": time.time()})


def execute_episode_pool(runner, pending, accept):
    """Keep slot ownership fixed while assigning independent episodes freely.

    At most the remainder of a twelve-case window is dispatched for a lineage.
    Completed episodes are ingested in the frozen lineage order. A full window
    runs revision, paired admission, publication and normal loading before any
    further episode of that lineage is eligible.
    """
    with (runner.root / "episode_pool.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FitSetError("another episode coordinator owns this run") from exc
        _execute_locked(runner, pending, accept)


def _execute_locked(runner, pending, accept):
    context = multiprocessing.get_context("spawn")
    cap = runner.episode_concurrency or runner.workers
    runner._episode_semaphore = context.BoundedSemaphore(cap)
    owners = {}
    by_slot = defaultdict(list)
    for index, shard in enumerate(runner._planned_slot_shards()):
        if not shard:
            continue
        if runner.evolution_config is not None:
            owner = FitSetWorkerEvolution(runner, index, shard)
            owner.recover()
        else:
            owner = None
        for row in shard:
            key = fit_slot_key(row["question_type"], row["evidence_state"])
            owners[key] = owner
            by_slot[key].append(row)
    for rows in by_slot.values():
        rows.sort(key=lambda row: (str(row["scene_id"]), str(row["variant_id"])))
    existing = runner.completed()
    eligible = {row["variant_id"] for row in pending} | set(existing)
    if runner.evolution_config is not None:
        for rows in by_slot.values():
            excluded = False
            for row in rows:
                if row["variant_id"] in existing:
                    # A prior interrupted window can leave completed results
                    # beyond a missing episode. They stay buffered and do not
                    # force a bounded resume to execute every gap at once.
                    continue
                included = row["variant_id"] in eligible
                _require(not (excluded and included),
                         "learning selection must be a prefix of each frozen lineage")
                excluded |= not included
    cursor = {key: 0 for key in by_slot}
    inflight = {}
    events = []
    aborted = None
    peak = 0
    last_slot = ""

    def ingest_ready():
        for key in sorted(by_slot):
            rows = by_slot[key]
            while cursor[key] < len(rows):
                row = rows[cursor[key]]
                result = existing.get(row["variant_id"])
                if result is None:
                    break
                owner = owners[key]
                if owner is not None:
                    owner.ingest_result(row, result)
                events.append({"kind": "ingested", "variant_id": row["variant_id"],
                               "slot_key": key, "at": time.time()})
                cursor[key] += 1

    def candidates():
        found = {}
        for key in sorted(by_slot):
            rows = by_slot[key]
            owner = owners[key]
            remaining = len(rows)
            if owner is not None:
                slot = owner.stream.slots.get(key)
                remaining = owner.config.batch_size - (
                    slot.new_since_attempt if slot else 0)
            window = rows[cursor[key]:cursor[key] + remaining]
            for row in window:
                vid = row["variant_id"]
                if vid in eligible and vid not in existing and vid not in inflight:
                    found[key] = row
                    break
        return found

    ingest_ready()
    jobs = context.Queue()
    messages = context.Queue()
    stop = context.Event()
    processes = []
    # Strip offline information before sending a runner to an online executor.
    executor_runner = copy.copy(runner)
    executor_runner.evolution_config = None
    executor_runner.label_metadata = {}
    executor_runner.input_contract_identity = copy.deepcopy(runner.input_contract_identity)
    executor_runner.input_contract_identity.pop("offline_metadata", None)
    executor_runner.seed_skills = []
    executor_runner.gate_panels = {}
    executor_runner.revision_client_factory = None
    executor_runner.gate_client_factory = None
    executor_runner.skills_for_variant = None
    prior_handler = signal.getsignal(signal.SIGINT)

    def interrupt(_signum, _frame):
        nonlocal aborted
        aborted = KeyboardInterrupt()
        stop.set()

    signal.signal(signal.SIGINT, interrupt)
    try:
        if candidates():
            for index in range(min(runner.workers, len(pending))):
                process = context.Process(
                    target=_executor,
                    args=(executor_runner, index, jobs, messages, stop,
                          runner._episode_semaphore),
                    name=f"fitset-episode-{index:02d}", daemon=False,
                )
                process.start()
                processes.append(process)
        while True:
            if aborted is None and not stop.is_set():
                ingest_ready()
                while len(inflight) < len(processes):
                    choices = candidates()
                    if not choices:
                        break
                    ordered = sorted(choices)
                    key = next((value for value in ordered if value > last_slot), ordered[0])
                    last_slot = key
                    row = choices[key]
                    owner = owners[key]
                    skills, identity = (
                        owner.provider(row) if owner is not None
                        else runner.skills_for_variant(row))
                    dispatch_path = (
                        runner.root / "dispatch" / f"{_variant_key(row['variant_id'])}.json")
                    prior = (json.loads(dispatch_path.read_text())
                             if dispatch_path.exists() else None)
                    if prior is not None:
                        # Other lineages can advance the owner snapshot during
                        # an interrupted window. Preserve the originally
                        # delivered snapshot if this method is unchanged.
                        _require(
                            prior["skills_sha256"] == content_sha256([
                                skill.model_dump(mode="json") for skill in skills]),
                            "unfinished dispatch belongs to a different Skill method")
                        identity = prior["skill_identity"]
                    frozen = {
                        "variant_id": row["variant_id"], "slot_key": key,
                        "skill_identity": identity,
                        "skills_sha256": content_sha256([
                            skill.model_dump(mode="json") for skill in skills]),
                        "run_identity": runner.run_identity,
                    }
                    _write_create_once(
                        dispatch_path, frozen,
                    )
                    inflight[row["variant_id"]] = row
                    jobs.put((dict(row), skills, identity))
                    events.append({"kind": "dispatched", **frozen, "at": time.time()})
                    peak = max(peak, len(inflight))
            if not inflight:
                break
            try:
                message = messages.get(timeout=0.5)
            except queue.Empty:
                dead = [process for process in processes
                        if process.exitcode is not None]
                if dead and aborted is None:
                    aborted = FitSetError("episode executor exited before completing its jobs")
                    stop.set()
                if stop.is_set() and all(p.exitcode is not None for p in processes):
                    break
                continue
            vid = message["variant_id"]
            kind = message["kind"]
            if kind == "started":
                events.append({key: value for key, value in message.items()})
            elif kind == "result":
                _require(vid in inflight, "unexpected episode result")
                inflight.pop(vid)
                result = dict(message["result"])
                _require(result["variant_id"] == vid, "episode result identity mismatch")
                existing[vid] = result
                events.append({key: value for key, value in message.items()
                               if key != "result"})
                if aborted is None:
                    try:
                        accept(result)
                    except BaseException as exc:
                        aborted = exc
                        stop.set()
            elif kind == "error":
                inflight.pop(vid, None)
                aborted = FitSetError(
                    f"episode {vid} failed ({message['error_type']}); "
                    "inspect its isolated artifacts before resuming")
                stop.set()
            else:
                raise FitSetError("invalid executor message")
        if aborted is None:
            ingest_ready()
    except BaseException as exc:
        aborted = exc
    finally:
        # Drain current episodes; never terminate unrelated services or replay
        # a request whose completion is unknown.
        stop.set()
        for _ in processes:
            jobs.put(None)
        # Children may be waiting for their result-queue feeder. Keep draining
        # while they exit, even when admission or a progress callback raised.
        while any(process.is_alive() for process in processes):
            try:
                message = messages.get(timeout=0.2)
            except queue.Empty:
                continue
            events.append({key: value for key, value in message.items()
                           if key != "result"})
        for process in processes:
            process.join()
        signal.signal(signal.SIGINT, prior_handler)
        jobs.cancel_join_thread()
        jobs.close()
        messages.close()
        messages.join_thread()
        receipt = {
            "scheduler": EPISODE_POOL_SCHEDULER,
            "executor_count": len(processes), "episode_concurrency": cap,
            "skill_owners": runner.skill_owners, "peak_dispatched": peak,
            "events": events, "completed_variant_ids": sorted(runner.completed()),
            "status": "failed" if aborted is not None else "drained",
            "run_identity": runner.run_identity,
        }
        _write_create_once(
            runner.root / "scheduler_receipts" / f"{time.time_ns()}.json", receipt)
        _write_json_atomic(runner.root / "episode_pool_receipt.json", receipt)
    if aborted is not None:
        raise aborted
