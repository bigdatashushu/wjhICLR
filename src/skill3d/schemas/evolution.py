"""Identity binding shared by the v11 parent/candidate solver."""
from typing import Literal
from . import Spec


class SkillEvaluationBinding(Spec):
    """§8.2：固定注入的显式标记（只能用于候选效果评测）。

    规范原文（§8.2）："该模式只能用于候选效果评测，不能用于 learning 经验采集、
    发布后运行或最终系统成绩。" 因此这个对象的存在本身就是"这一跑不是正常检索"的
    证据 —— 它必须落进 trace，且不允许被改写成正常检索命中。
    """

    mode: Literal["fixed_skill_evaluation"]
    arm: Literal["parent", "candidate"]
    skill_id: str
    skill_version: str
    content_sha256: str
    bypassed_component: Literal["retrieval_selection"]

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.skill_version != f"{self.skill_id}@{self.skill_version.split('@')[-1]}":
            raise ValueError(f"skill_version 必须是 skill_id@version：{self.skill_version!r}")
        if not self.content_sha256:
            raise ValueError("固定注入必须带正文 hash（§16.2：核对两臂正文 hash）")


class GPUJob(Spec):
    """M20 调度单元。"""

    gpu_rank: int
    role: Literal["reconstruct", "vllm", "eval"]
    paired_unit_id: str
