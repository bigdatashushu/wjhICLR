"""Print the existing MMSI experiment's live progress to the terminal and log."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

SOURCE = Path("/tmp/h3d-mmsi-live-stream-resume08.py")
LOG = Path("/Users/bytedance/wangjh/outputs/harness3d_four_datasets_20261031/MMSI实时进度_20261101.log")
LOCK = Path("/tmp/h3d-mmsi-live-follow-20261101.lock")
STARTUP = Path("/tmp/h3d-mmsi-live-follow-20261101.started.json")
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"


def stop(_signum, _frame):
    raise KeyboardInterrupt


def emit_snapshot(value, emit):
    if value.get("phase") == "reader_failed":
        emit(f'[{value["at"][11:19]}] 只读监控失败：{value["error_type"]}；实验保持原有状态。')
        return
    if value["initial"]:
        emit("MMSI 实时打印已连接。准确率按已评分题目计算，失败计零分；以下回显最近 8 条记录。")
    for row in value["records"]:
        arm = "Skill" if row["arm"] == "skill" else "Qwen"
        score = "待评分" if row["score"] is None else "对" if row["score"] else "错"
        if row["status"] != "answered":
            score = row["status"] + "/" + score
        change = "评分更新" if row.get("score_update") else "结果"
        prediction = row["prediction"] if row["prediction"] is not None else "—"
        emit(
            f'{row["at"][11:19]} [{arm}/{change}] 第{row["ordinal"]}题 '
            f'{row["task"]} 预测={prediction} {score} | '
            f'请求{row["requests"]} 工具{row["successful_tools"]}/{row["tools"]} '
            f'用时{row["seconds"]:.1f}s')
    parts = []
    for arm, label in (("skill", "Skill"), ("direct", "Qwen")):
        row = value["summary"][arm]
        accuracy = f'{row["accuracy_graded"] * 100:.2f}%' if row["accuracy_graded"] is not None else "—"
        text = (f'{label} {row["recorded"]}/{row["expected"]}，'
                f'答对{row["correct"]}/{row["graded"]}（{accuracy}）')
        if row["pending_grade"]:
            text += f'，待评分{row["pending_grade"]}'
        parts.append(text)
    names = {
        "running": "运行中", "handling_timeout": "核验单题服务错误", "draining_after_fatal_error": "停止派发，等待在途请求结束", "new_service_error": "新增服务错误",
        "awaiting_acceptance": "等待最终验收", "complete_verified": "完整验收通过",
        "stopped": "已停止", "acceptance_failed": "最终验收失败", "runner_exited": "进程已退出",
    }
    statuses = value["request_statuses"]
    emit(f'{value["at"][11:19]} [{names[value["phase"]]}] ' + "；".join(parts)
         + f'；模型请求 {value["requests_used"]}/{value["requests_limit"]}'
         + f'，在途{statuses.get("started", 0)}，失败{statuses.get("error", 0)}')
    if value["phase"] == "complete_verified":
        emit(f'MMSI 已完成并通过独立验收：两组各365题，保留{value["total_service_errors"]}次服务失败（计零分）。')


def main():
    assert SOURCE.is_file() and LOG.parent.is_dir()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with LOCK.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STARTUP.write_text(json.dumps({
            "pid": os.getpid(), "at": datetime.now().astimezone().isoformat(),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "log": str(LOG), "model_requests_issued_by_monitor": 0,
        }) + "\n")
        environment = {
            k: v for k, v in os.environ.items()
            if not any(t in k.upper() for t in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY"))
        }
        environment.update(
            H3D_CREDENTIAL_CHOICE="1",
            SSH_ASKPASS="/private/tmp/harness3d_credentials_askpass.py",
            SSH_ASKPASS_REQUIRE="force", DISPLAY="harness3d-ssh-auth:0")
        command = [
            "ssh", "-T", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
            "-o", "ControlMaster=auto", "-o", "ControlPersist=900",
            "-o", "ControlPath=/private/tmp/h3d-other-pilots-%r-%h-%p",
            "-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password",
            "-o", "NumberOfPasswordPrompts=1", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2", "wangy@10.10.0.34", PYTHON + " -u -B -",
        ]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=environment, start_new_session=True)
        try:
            process.stdin.write(SOURCE.read_text())
            process.stdin.close()
            with LOG.open("a", encoding="utf-8", buffering=1) as log:
                def emit(line):
                    print(line, flush=True)
                    print(line, file=log, flush=True)
                emit("\n" + datetime.now().astimezone().isoformat() + " MMSI 实时打印启动")
                for line in process.stdout:
                    emit_snapshot(json.loads(line), emit)
                code = process.wait()
                if code:
                    emit("实时打印已停止；请检查连接或实验停止回执。没有自动重连或重试模型。")
                return code
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("实时打印已停止，服务器实验继续保持原有状态。", flush=True)
        sys.exit(0)
