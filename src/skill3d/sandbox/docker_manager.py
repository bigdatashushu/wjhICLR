"""per-episode 容器生命周期管理（§4 M10）。

docker SDK lazy import；docker 不可用时抛 SandboxUnavailableError 明确错误。
容器参数按 §13.2：只读根 FS / tmpfs 可写层 / --network none / recon 只读挂载；
final-test 目录绝不挂载（硬约束 9）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .receipt import ReceiptChain, SandboxReceipt

DEFAULT_IMAGE = "skill3d-sandbox:latest"  # TODO_USER_INPUT 镜像 digest 锁定（§13.6）
DEFAULT_CPUS = 4
DEFAULT_MEMORY = "8g"
DEFAULT_PIDS_LIMIT = 256


class SandboxUnavailableError(RuntimeError):
    """docker SDK 或 daemon 不可用。"""


def build_container_kwargs(
    recon_volume: str,
    recon_mount: str = "/data",
    image: str = DEFAULT_IMAGE,
) -> dict:
    """docker SDK containers.run 参数（§13.2；纯函数可测）。"""
    return {
        "image": image,
        "read_only": True,
        "tmpfs": {"/tmp": ""},
        "network_mode": "none",
        "volumes": {recon_volume: {"bind": recon_mount, "mode": "ro"}},
        "nano_cpus": DEFAULT_CPUS * 10**9,
        "mem_limit": DEFAULT_MEMORY,
        "pids_limit": DEFAULT_PIDS_LIMIT,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "detach": True,
        # 注意：final-test 目录绝不出现在 volumes 中（硬约束 9）
    }


@dataclass
class ContainerHandle:
    episode_id: str
    container_id: str
    receipts: ReceiptChain


class DockerManager:
    """per-episode 容器：start → 运行 → destroy。"""

    def __init__(self, recon_volume: str, image: str = DEFAULT_IMAGE) -> None:
        self.recon_volume = recon_volume
        self.image = image
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            try:
                import docker  # lazy import：未安装时给明确错误
            except ImportError as exc:
                raise SandboxUnavailableError(
                    "docker SDK 未安装（pip install docker）；沙箱不可用"
                ) from exc
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise SandboxUnavailableError(f"docker daemon 不可用: {exc}") from exc
        return self._client

    def start(self, episode_id: str, recon_artifact_ref: Optional[str] = None) -> ContainerHandle:
        client = self._get_client()
        kwargs = build_container_kwargs(self.recon_volume, image=self.image)
        kwargs["name"] = f"skill3d-sandbox-{episode_id}"
        container = client.containers.run(**kwargs)
        chain = ReceiptChain(episode_id)
        chain.append(
            "container_start",
            {"container_id": container.id, "recon_artifact_ref": recon_artifact_ref},
        )
        return ContainerHandle(episode_id=episode_id, container_id=container.id, receipts=chain)

    def destroy(self, handle: ContainerHandle) -> SandboxReceipt:
        client = self._get_client()
        try:
            container = client.containers.get(handle.container_id)
            container.remove(force=True)
        finally:
            return handle.receipts.append(
                "container_destroy", {"container_id": handle.container_id}
            )
