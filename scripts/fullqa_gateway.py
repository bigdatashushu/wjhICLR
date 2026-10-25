#!/usr/bin/env python3
"""Authenticated custody/ACK endpoint; never exposes arbitrary shell execution."""
from __future__ import annotations

import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import fullqa_custody as c


class Gateway:
    def __init__(self, config):
        self.config = config
        self.root, self.home, self.cache = map(Path,
            (config["root"], config["home"], config["cache"]))
        c.pins(self.root)
        self.collector = c.load_pinned("fullqa_gateway_native_collector",
            self.home / "collect_fullqa_acceptance_v1.py", c.COLLECTOR)
        self.lock = threading.RLock()
        (self.home / "deliveries").mkdir(exist_ok=True, mode=0o700)
        for name in ("jobs", "acks"):
            (self.cache / name).mkdir(parents=True, exist_ok=True, mode=0o700)

    def legacy(self):
        path = Path(self.config["legacy_startup"])
        identity = c.read(path)["process"]
        return {"process": identity, "live": c.live(identity),
                "job_key": self.config["legacy_job_key"]}

    def deliveries(self):
        result = []
        for path in sorted((self.home / "deliveries").glob("*.json")):
            if path.name.endswith(".state.json"):
                continue
            value = c.read(path)
            state_path = path.with_suffix(".state.json")
            state = c.read(state_path) if state_path.exists() else {"phase": "delivered"}
            identity = state.get("process")
            result.append({"delivery_id": path.stem, "job_key": value["lease"]["spec"]["job_key"],
                           "phase": state["phase"], "at": state.get("at"),
                           "live": bool(identity and c.live(identity))})
        return result

    def resources(self):
        memory = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemAvailable:", "Shmem:"))}
        fs = os.statvfs(self.cache)
        return {**memory, "cache_free_bytes": fs.f_bavail * fs.f_frsize}

    def state(self):
        return {"schema": "harness3d-fullqa-custody-state/1", "at": time.time(),
                "process": c.process(os.getpid()), "legacy": self.legacy(),
                "deliveries": self.deliveries(), "resources": self.resources(),
                "method_sha256": c.METHOD, "worker_sha256": c.WORKER}

    def deliver(self, body):
        from skill3d.cluster.queue import Lease
        lease = Lease.from_dict(body["lease"])
        c.require(lease.worker.to_dict() == self.config["worker"]
                  and lease.spec.identity.identity_hash == c.IDENTITY, "wrong_delivery_scope")
        delivery_id = c.digest(c.canon({"job_key": lease.job_key, "token": lease.token}))
        c.require(body["delivery_id"] == delivery_id, "delivery_id_changed")
        path = self.home / "deliveries" / (delivery_id + ".json")
        with self.lock:
            if path.exists():
                stored = c.read(path)["lease"]
                # Renewals may differ; custody and initial model attempt cannot.
                c.require(all(stored[k] == body["lease"][k] for k in
                    ("spec", "worker", "token", "attempt", "phase", "output_metadata")),
                    "delivery_conflict")
                return {"status": "retained", "delivery_id": delivery_id}
            c.require(not self.legacy()["live"], "legacy_worker_still_running")
            c.require(all(row["phase"] == "complete" for row in self.deliveries()),
                      "slot_has_unresolved_custody")
            c.require(not list((self.cache / "jobs").iterdir()), "cache_not_released")
            resources = self.resources()
            c.require(resources["MemAvailable"] >= 172 * c.GIB
                      and resources["cache_free_bytes"] >= 92 * c.GIB, "memory_reserve")
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=True)
            identity = subprocess.run(["nvidia-smi", "-i", "3", "--query-gpu=uuid",
                                       "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=True)
            c.require(identity.stdout.strip().decode() not in gpu.stdout.decode(), "gpu_not_released")
            c.durable(path, {"schema": "harness3d-fullqa-custody-delivery/1",
                            "delivery_id": delivery_id, "lease": body["lease"]}, immutable=True)
            c.durable(path.with_suffix(".state.json"), {"phase": "delivered", "at": time.time()})
            self.launch(path)
        return {"status": "retained", "delivery_id": delivery_id}

    def launch(self, path):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=self.config["worker"]["gpu_id"],
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                          "--config", str(self.home / "gateway_config.json"),
                          "--child", str(path)],
                         env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)

    def reconcile(self):
        with self.lock:
            for row in self.deliveries():
                if row["phase"] in {"complete", "quarantined"} or row["live"]:
                    continue
                path = self.home / "deliveries" / (row["delivery_id"] + ".json")
                key = c.digest(row["job_key"].encode())
                ack_path = self.cache / "acks" / (key + ".json")
                pending = list((self.cache / "jobs" / key).glob("attempt-*/pending.json"))
                action = c.recovery_action(row["phase"], row["live"], len(pending) == 1,
                                           ack_path.exists())
                if action == "copy_ack":
                    if (self.cache / "jobs" / key).exists():
                        c.require(len(pending) == 1, "ack_cleanup_without_checkpoint")
                        self.launch(path)
                        continue
                    c.durable_bytes(self.root / "worker_cache/acks" / (key + ".json"),
                                    ack_path.read_bytes(), immutable=True)
                    c.durable(path.with_suffix(".state.json"),
                              {"phase": "complete", "job_key": row["job_key"], "at": time.time()})
                elif action in {"launch", "resume_publication"}:
                    # A child writes its identity before any request. The file
                    # lock below fences a crash between Popen and this write.
                    self.launch(path)
                else:
                    c.durable(path.with_suffix(".state.json"), {
                        "phase": "quarantined", "job_key": row["job_key"],
                        "error_type": "ComputeProcessLost", "at": time.time()})

    def probe(self, final):
        c.require(type(final) is bool, "invalid_final_flag")
        value = c.probe_source(self.collector.WORKER_PROBE,
                              [self.collector.RUN, c.METHOD, final])
        roots = [self.root / "worker_cache", self.cache,
                 Path(self.config["legacy_cache"]),
                 Path(self.config["network_cache"])]
        extra = {str(root): [path.name for path in (root / "jobs").iterdir()]
                 for root in roots if (root / "jobs").exists()}
        if final:
            c.require(all(not names for names in extra.values()), "extra_staging_not_released")
            c.require(not self.legacy()["live"], "legacy_not_exited")
            c.require(all(row["phase"] == "complete" for row in self.deliveries()),
                      "custody_not_complete")
        return {"native": value, "cache_staging": extra,
                "probe_source_sha256": c.digest(self.collector.WORKER_PROBE.encode())}


def serve(config):
    gateway = Gateway(config)
    token = os.environ["HARNESS3D_CLUSTER_TOKEN"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return
            try:
                size = int(self.headers["Content-Length"])
                c.require(0 < size <= 4 * 1024 * 1024, "request_size")
                body = json.loads(self.rfile.read(size))
                if self.path == "/state":
                    value = gateway.state()
                elif self.path == "/deliver":
                    value = gateway.deliver(body)
                elif self.path == "/probe":
                    value = gateway.probe(body["final"])
                else:
                    self.send_error(404)
                    return
                raw = c.canon(value)
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                raw = c.canon({"status": "refused", "error_type": type(exc).__name__})
                self.send_response(409)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

    server = ThreadingHTTPServer((config["host"], config["port"]), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    c.durable(Path(config["home"]) / "gateway_startup.json",
              {"at": time.time(), "process": c.process(os.getpid()), "port": config["port"]})
    while True:
        try:
            gateway.reconcile()
            c.durable(Path(config["home"]) / "gateway_current.json", gateway.state())
        except Exception as exc:
            c.durable(Path(config["home"]) / "gateway_error.json",
                      {"at": time.time(), "error_type": type(exc).__name__})
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    config = c.read(args.config)
    os.umask(0o077)
    sys.path[:0] = [str(Path(config["root"]) / "source_frozen/src"),
                   str(Path(config["root"]) / "source_frozen/scripts")]
    lock_path = (args.child.with_suffix(".lock") if args.child else
                 Path(config["home"]) / "gateway.lock")
    with lock_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.child:
            c.run_child(config, args.child)
        else:
            serve(config)


if __name__ == "__main__":
    main()
