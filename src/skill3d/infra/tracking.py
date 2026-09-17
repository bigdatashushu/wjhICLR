"""M21 离线实验追踪：优先 MLflow（lazy import，本地文件后端），不可用降级为 JSONL。"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


def _import_mlflow():
    try:
        import mlflow  # lazy import：未安装时降级
        return mlflow
    except ImportError:
        return None


class Tracker:
    """统一追踪接口：start_run / log_param / log_metric / end_run。"""

    def __init__(self, tracking_dir: str, experiment: str = "skill3d") -> None:
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiment = experiment
        self._mlflow = _import_mlflow()
        self._run_id: str | None = None
        self._jsonl_path: Path | None = None
        self._mlflow_active = False

    @property
    def backend(self) -> str:
        return "mlflow" if self._mlflow is not None else "jsonl"

    def start_run(self, run_name: str | None = None) -> str:
        if self._mlflow is not None:
            self._mlflow.set_tracking_uri(str(self.tracking_dir / "mlruns"))
            self._mlflow.set_experiment(self.experiment)
            run = self._mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id
            self._mlflow_active = True
        else:
            self._run_id = f"local-{uuid.uuid4().hex[:12]}"
            self._jsonl_path = self.tracking_dir / f"run_{self._run_id}.jsonl"
            self._write_jsonl({"event": "start_run", "run_name": run_name,
                               "ts": time.time()})
        return self._run_id

    def log_param(self, key: str, value) -> None:
        if self._mlflow_active:
            self._mlflow.log_param(key, value)
        else:
            self._write_jsonl({"event": "param", "key": key, "value": str(value),
                               "ts": time.time()})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._mlflow_active:
            self._mlflow.log_metric(key, value, step=step)
        else:
            self._write_jsonl({"event": "metric", "key": key, "value": value,
                               "step": step, "ts": time.time()})

    def end_run(self) -> None:
        if self._mlflow_active:
            self._mlflow.end_run()
            self._mlflow_active = False
        else:
            self._write_jsonl({"event": "end_run", "ts": time.time()})

    def _write_jsonl(self, record: dict) -> None:
        assert self._jsonl_path is not None
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
