"""M5 对象绑定/分割：SAM2 视频 mask 传播 + mask×深度反投影绑定世界点云（§4 M5）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
sam2/torch 未安装，lazy import；mask→世界点云绑定与 G7/G9 统计为 numpy 真实实现。
（G8 包围盒覆盖已按附录 A 删除，本模块不再计算它。）

G-19 已核验配置（§7.1）：`sam2==1.1.0` + checkpoint `facebook/sam2.1_hiera-tiny`
（~33.5M encoder，适合在线），启动时预拉取到本地 cache。

产物去向：
- `ObjectRecord`（v6 §5.6：点云/bbox/质心 + `track_id`/`duplicate_suspect`/
  `pointconf_world`/`grounding_status`）→ `SceneHandle`（硬约束 17：Tool 只经句柄访问）；
- `track_ious`（G9）→ M4 质量门禁；
- `track_stable_ratio`（§7.1 `track_consensus` 的占比输入）→ M4 证据画像；
- `dynamic_masks`（G7）→ 重建污染检测（`dynamic_masks_from_rigidity`）。

v6 证据字段（本模块的落点）：

- `track_id`：**跨帧 track 身份** = 产生该对象的 SAM2 传播序号（确定性字符串）。
  去重合并后，一组里只有幸存代表留在清单中，故 `count_objects`（§9.2
  "track 共识计数，不数清单长度"）按 `track_id` 去重即得 `n_distinct_tracks`；
- `duplicate_suspect`：去重**吸收**了 ≥2 条候选（清单长度被高估的直接证据），
  或去重后仍有两个同类对象的世界质心落在同一容差内（时序不重叠 → 没合并，
  但"同地同类不同时"仍有重复嫌疑）→ 计数必须降级输出（§7.1 track_consensus）；
- `pointconf_world`：与 `pointcloud_world` **逐点 1:1 对齐**的 VGGT 逐点置信度 ref。
  调用方不给 `point_conf` 时写空串（**绝不伪造**置信度，§12.2 软权重来源）；
- `grounding_status`：场景清单 = `base_list`，逐题补漏 = `question_targeted_fill`。
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.coords import vlm_box_to_pixels
from skill3d.schemas.reconstruction import ObjectRecord

# 场景清单缓存版本（改 M5 清单构造逻辑时递增 → 旧缓存自动失效，不会混用口径）
# v2：`ObjectInstance` 增 `visible_frames`（逐帧可见性题型）→ 旧缓存缺该字段，必须失效
# v3：v6 §5.6 字段改名（obj_id/category_name/det_conf）+ 新增 track_id /
#     duplicate_suspect / pointconf_world / grounding_status → 旧缓存是 v5 口径
#     （无 track，计数无法做 track 共识），必须失效重算。
_INVENTORY_VERSION = "m5-inventory-4"

# `track_id` 命名空间（v6 §5.6）：不同 pass 的传播序号会重名（都从 0 起），
# 前缀把它隔开 —— 否则 `count_objects` 的 track 去重会把两条不同对象算成一条。
TRACK_PREFIX_BASE_LIST = "trk"      # 场景基础清单（scene 级，可缓存）
TRACK_PREFIX_QUESTION = "qtrk"      # 逐题补漏（题级，不复用基础清单的 track）
# VLM **通用清单**问题（与具体题目无关 → 清单可跨 episode 复用/缓存）
_GENERIC_INVENTORY_QUESTION = (
    "请列出这张室内照片里最主要的物体（家具、门窗、家电、常见物品），"
    "不要只挑与某道题相关的物体。")

# G-19 档位（本机实测，2026-09-18）：可用权重为 SAM2.1 Hiera-Large 与 Hiera-Base+，
# **Tiny 未下载**。实测加载峰值显存：Large 0.98 GiB / Base+ 0.44 GiB（权重部分）。
# 默认取 Large（质量优先，仍可与 VGGT 同卡共存）；低显存场景改 Base+：
#   SKILL3D_SAM2_CHECKPOINT=<base+ .pt> SKILL3D_SAM2_CONFIG=configs/sam2/sam2_hiera_b+.yaml
# 注意 `SAM2_CONFIG` 必须用官方文档的**相对形式**（hydra 以 sam2/configs 为搜索根；
# 传绝对路径会 MissingConfigException）。
SAM2_CHECKPOINT: Optional[str] = (
    "/home/cvailab/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/"
    "snapshots/227a114a2f535cd147f82442e7d2038cdd2e5d68/sam2.1_hiera_large.pt")
SAM2_CONFIG: Optional[str] = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_LOCAL_CACHE: str = "data/checkpoints"                      # 预拉取目录（§7.1 G-19）
_ENV_CHECKPOINT = "SKILL3D_SAM2_CHECKPOINT"
_ENV_CONFIG = "SKILL3D_SAM2_CONFIG"
_ENV_ALLOW_DOWNLOAD = "SKILL3D_ALLOW_HF_DOWNLOAD"

# VSI-Bench 问题文本中的常见物体名词（确定性提示词抽取，非 VLM；VLM 路径见 docstring）
OBJECT_VOCABULARY: tuple[str, ...] = (
    "sofa", "couch", "chair", "table", "desk", "bed", "door", "doorway",
    "cabinet", "counter", "countertop", "stool", "shelf", "lamp", "plant",
    "tv", "monitor", "fridge", "refrigerator", "microwave", "oven", "sink",
    "toilet", "bathtub", "mirror", "painting", "window", "rug", "pillow",
    "box", "bottle", "cup", "bowl", "book", "bag", "basket", "trash",
    "nightstand", "dresser", "wardrobe", "bench",
)

# G7 刚性残差阈值（px）：观测光流与自运动预测光流之差超此值记为动态（TODO_CALIBRATE）
TH_RIGIDITY_RESIDUAL_PX: float = 3.0
# G7 自适应阈值的稳健系数：thr = max(下界, median + k×1.4826×MAD)（TODO_CALIBRATE）
RIGIDITY_MAD_K: float = 3.0
# 动态占比上限：超此值记重建污染（§10 G7 触发动作，TODO_CALIBRATE）
TH_DYNAMIC_RATIO: float = 0.3


def resolve_sam2_config(checkpoint: Optional[str] = None,
                        config: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """解析 checkpoint/config：显式参数 > 环境变量 > 模块默认。"""
    ckpt = checkpoint or os.environ.get(_ENV_CHECKPOINT) or SAM2_CHECKPOINT
    cfg = config or os.environ.get(_ENV_CONFIG) or SAM2_CONFIG
    return ckpt, cfg


def ensure_checkpoint(checkpoint: Optional[str] = None,
                      cache_dir: str = SAM2_LOCAL_CACHE) -> Optional[str]:
    """把 checkpoint 解析为本地路径（HF id 时预拉取到 cache，§7.1 G-19）。

    已是本地文件则原样返回；HF id 走 `huggingface_hub.hf_hub_download`。
    自动下载需显式开启 `SKILL3D_ALLOW_HF_DOWNLOAD=1`：离线/受限网络环境下
    HF 连接会长时间悬挂，默认不隐式发起网络请求（返回 None → 调用方报降级）。
    """
    ckpt, _ = resolve_sam2_config(checkpoint=checkpoint)
    if ckpt is None:
        return None
    p = Path(ckpt)
    if p.is_file():
        return str(p)
    target = Path(cache_dir) / str(ckpt).replace("/", "__")
    if target.is_file():
        return str(target)
    if not os.environ.get(_ENV_ALLOW_DOWNLOAD):
        return None
    try:
        from huggingface_hub import hf_hub_download  # lazy import
    except Exception:  # noqa: BLE001
        return None
    try:
        got = hf_hub_download(repo_id=str(ckpt), filename="model.safetensors",
                              local_dir=str(target.parent / target.name))
        return str(got)
    except Exception:  # noqa: BLE001 - 无网络/无权限 → 交由调用方报降级
        return None


def build_video_predictor(checkpoint: Optional[str] = None,
                          config: Optional[str] = None):
    """构建 SAM2 视频预测器（lazy import）。

    TODO: `build_sam2_video_predictor` 具体签名以官方 repo 为准（§4 M5 字段 5）。
    配置缺失 / sam2 未安装 / 权重不可得，统一抛 RuntimeError 并保留根因，
    由 M5 降级分支接住（§4 M5 字段 9）。
    """
    ckpt, cfg = resolve_sam2_config(checkpoint, config)
    if ckpt is None or cfg is None:
        raise RuntimeError("SAM2 checkpoint/config 未配置（TODO_USER_INPUT：档位见 §7.1 G-19）")
    try:
        from sam2.build_sam import build_sam2_video_predictor  # lazy import
    except ImportError as exc:
        raise RuntimeError(f"sam2 未安装（pip install sam2==1.1.0）：{exc}") from exc
    local = ensure_checkpoint(ckpt)
    if local is None:
        raise RuntimeError(
            f"SAM2 权重不可得（{ckpt}）：预拉取失败或未配置本地路径；"
            f"可设 {_ENV_CHECKPOINT}=<本地 .pt 路径>，或先下载到 {SAM2_LOCAL_CACHE}"
            f"（自动下载需 {_ENV_ALLOW_DOWNLOAD}=1）"
        )
    return build_sam2_video_predictor(cfg, local)


# ------------------------------------------------------------------ 提示词来源 ----

_GENERIC_NOUN_PATTERNS = (
    r"how many\s+(.+?)\s+(?:are|is|do|does|can)\b",
    r"what is the (?:size|length|width|height|distance) of (?:the\s+)?(.+?)[\?\.,]",
    r"(?:is|are)\s+(?:there|the)\s+(.+?)\s+(?:in|on|near|next to|closest|farther|farthest)\b",
    r"the\s+(.+?)\s+(?:that|which)\s+(?:is|are|appears?)\b",
)


def _singularize(noun: str) -> str:
    """轻量单数化（只处理规则复数；不做词形还原，避免引入不确定性）。"""
    w = str(noun).strip()
    if len(w) > 3 and w.lower().endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.lower().endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 2 and w.lower().endswith("s") and not w.lower().endswith("ss"):
        return w[:-1]
    return w


def object_prompts_from_question_generic(question: str) -> list[str]:
    """通用名词短语抽取（不依赖固定词表；覆盖 "How many X are..." 等官方句式）。

    为什么要它：固定词表遇到 `computer tower` 这类词直接抽空 → M5 拿不到任何提示 →
    **整个 scene 的对象工具被收回**（fail-closed 的爆炸半径过大）。这里只做轻量清洗，
    不做语义推断；抽不出来时返回空列表（由调用方走检测器/整帧兜底）。
    """
    import re as _re

    q = str(question).strip()
    out: list[str] = []
    for pat in _GENERIC_NOUN_PATTERNS:
        m = _re.search(pat, q, _re.IGNORECASE)
        if not m:
            continue
        phrase = m.group(1).strip(" ?.,\"\'")
        phrase = _re.sub(r"\(s\)$", "", phrase)          # "tower(s)" → "tower"
        phrase = _re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=_re.IGNORECASE)
        if 0 < len(phrase) <= 60 and phrase.lower() not in {"room", "scene", "image"}:
            out.append(phrase)
    return list(dict.fromkeys(out))


def object_prompts_from_question(question: str) -> list[str]:
    """从问题文本抽取候选物体名词（确定性，VLM 无关）。

    在线正解由 Qwen3-VL-8B 给出"对象名 + bbox"提示（§4 M5 字段 12，非 GPT-6）；
    本函数是无需模型的确定性兜底，覆盖 VSI-Bench 问题的常见名词。
    """
    text = str(question).lower()
    hits: list[str] = []
    for word in OBJECT_VOCABULARY:
        if word in text and word not in hits:
            # 避免 "chair" 命中 "chairman" 类误报：要求词边界
            idx = text.find(word)
            left_ok = idx == 0 or not text[idx - 1].isalpha()
            right_ok = idx + len(word) >= len(text) or not text[idx + len(word)].isalpha()
            if left_ok and right_ok:
                hits.append(word)
    return hits


def box_prompts_from_vlm(frames: Sequence[np.ndarray], question: str, client,
                         *, frame_idx: int = 0,
                         max_objects: int = 6,
                         seed: Optional[int] = None) -> tuple[list[str], list[list[float]]]:
    """用**在线 Qwen3-VL-8B**给出对象名 + 像素 bbox 提示（§4 M5 字段 12，非 GPT-6）。

    这是 SAM2 提示的正解来源：文本抽名词只能给类别、给不出框，整帧当框会让 SAM2
    退化成"分割整幅图"。返回 `(class_hints, boxes)`；模型不可用/解析失败返回 `([], [])`
    由调用方走降级路径（不臆造框）。

    bbox 约定 `[x0, y0, x1, y1]`（像素，左上原点），与 SAM2 box prompt 一致。
    """
    import base64
    import json
    import re

    import cv2

    if client is None or not len(frames):
        return [], []
    img = np.asarray(frames[min(frame_idx, len(frames) - 1)])
    h, w = int(img.shape[0]), int(img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return [], []
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    prompt = (
        f"这是一张室内照片。问题：{question}\n"
        f"请找出不超过 {max_objects} 个与问题相关的物体，给出名称与包围盒。\n"
        '严格只输出 JSON 数组：[{"name": "table", "bbox": [x0, y0, x1, y1]}]\n'
        "**坐标必须用 0-1000 归一化整数**（左上为 (0,0)，右下为 (1000,1000)），"
        "x0<x1, y0<y1。若图里确实没有相关物体，输出 []。不要解释。"
    )
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    try:
        # 请求级 seed：M5 的框提示也参与"同配置重复运行逐题一致"的复现要求
        text = client.chat(messages, max_tokens=512, seed=seed)
    except Exception as exc:  # noqa: BLE001 - 模型不可用 → 降级
        box_prompts_from_vlm.last_error = f"{type(exc).__name__}: {exc}"
        return [], []
    box_prompts_from_vlm.last_raw = text

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        box_prompts_from_vlm.last_error = f"未找到 JSON 数组；raw={text[:200]!r}"
        return [], []
    try:
        items = json.loads(m.group(0))
    except Exception as exc:  # noqa: BLE001
        box_prompts_from_vlm.last_error = f"JSON 解析失败: {exc}; raw={m.group(0)[:200]!r}"
        return [], []
    hints: list[str] = []
    boxes: list[list[float]] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        box = it.get("bbox") or it.get("box")
        name = str(it.get("name") or it.get("label") or "").strip()
        if not name or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        # §9 坐标适配层：VLM-normalized-1000 → original 像素（C-6/实测 999 量级）
        x0, y0, x1, y1 = vlm_box_to_pixels([x0, y0, x1, y1], w, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        hints.append(normalize_hint(name))
        boxes.append([x0, y0, x1, y1])
    return hints, boxes


def normalize_hint(name: str) -> str:
    """对象名归一（小写、去空白），与 G-11 的类先验查表口径一致。"""
    return str(name).strip().lower().replace(" ", "_")


def object_prompts_from_handle(handle) -> tuple[list[str], list[list[float]]]:
    """由 SceneHandle 现有对象给出 (class_hints, box_prompts)（重跑/增量场景）。"""
    hints: list[str] = []
    boxes: list[list[float]] = []
    if handle is None or not hasattr(handle, "list_objects"):
        return hints, boxes
    for oid in handle.list_objects():
        obj = handle.get_object(oid)
        if len(obj.bbox) == 6:
            hints.append(str(obj.category_name))
            boxes.append([float(v) for v in obj.bbox])
    return hints, boxes


def track_objects(
    frames: Sequence[np.ndarray],
    box_prompts: Optional[Sequence[Sequence[float]]] = None,
    predictor=None,
    *,
    prompts: Optional[Sequence[tuple[int, int, Sequence[float]]]] = None,
    class_hints: Optional[Sequence[str]] = None,
    handle=None,
    question: str = "",
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
) -> list[dict[int, np.ndarray]]:
    """SAM2 视频 mask 传播（§4 M5 伪代码）。

    `box_prompts`：每个对象一个 [x0,y0,x1,y1] 提示框。为 None 时按以下顺序取提示：
    ① `handle` 中已有对象（增量场景）；② 由 `question` 抽取的对象名词（无框提示）。
    两者都不可得时抛 RuntimeError（SAM2 需要框/点提示，不臆造分割结果）。

    返回 per-object `{frame_idx: mask(H,W) bool}`。
    """
    hints = list(class_hints or [])
    boxes = list(box_prompts) if box_prompts is not None else []
    if not boxes and handle is not None:
        h_hints, h_boxes = object_prompts_from_handle(handle)
        hints, boxes = h_hints, h_boxes
    if not boxes and question:
        hints = object_prompts_from_question(question)
        if not hints:
            raise RuntimeError(
                f"SAM2 提示不可得：问题中未识别到候选物体（{question[:40]!r}），"
                "且无既有对象/框提示（在线正解由 Qwen3-VL-8B 给对象名+bbox 提示）"
            )
        # 无数值提示框：以全帧为框（退化为整帧分割，仅用于拿到候选 mask）
        h = int(np.asarray(frames[0]).shape[0]) if len(frames) else 1
        w = int(np.asarray(frames[0]).shape[1]) if len(frames) else 1
        boxes = [[0.0, 0.0, float(w), float(h)] for _ in hints]
    if not boxes:
        raise RuntimeError("SAM2 需要 box_prompts 或可解析的既有对象（§4 M5 伪代码）")

    if predictor is None:
        predictor = build_video_predictor(checkpoint, config)
    # 已核验的官方 API：init_state 接受 **MP4 或 JPEG 文件夹路径**（不接受 ndarray），
    # 故把帧写到临时 JPEG 目录；帧名需为连续数字（load_video_frames 的约定）。
    frames_dir = _write_jpeg_sequence(frames)
    # 官方 demo 会在调用前**全局进入 bf16 autocast**；不进入会出现
    # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"。
    with _autocast_ctx():
        state = predictor.init_state(
            frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
    # 提示形式：`prompts=[(frame_idx, obj_id, box)]`（多帧各自给框，SAM2 从该帧起传播）
    # 或退化形式 `boxes=[box,...]`（统一在第 0 帧给框）
    prompt_list = ([(int(fi), int(oid), box) for fi, oid, box in prompts]
                   if prompts else [(0, i, b) for i, b in enumerate(boxes)])
    n_objects = max([oid for _fi, oid, _b in prompt_list], default=-1) + 1
    for fi, oid, box in prompt_list:
        # 已核验签名：add_new_points_or_box(state, frame_idx, obj_id, box=...)
        # dtype 必须与模型参数一致（SAM2 在 bf16 下运行；传 float32 会
        # "mat1 and mat2 must have the same dtype"）
        with _autocast_ctx():
            predictor.add_new_points_or_box(state, frame_idx=fi, obj_id=oid,
                                            box=_prompt_box(box, predictor))
    # propagate_in_video 产出 (frame_idx, obj_ids, video_res_masks)
    per_object: list[dict[int, np.ndarray]] = [dict() for _ in range(max(n_objects, 1))]
    for frame_idx, obj_ids, mask_logits in _propagate_bidirectional(predictor, state):
        for oid, logits in zip(obj_ids, mask_logits):
            mask = _to_bool_mask(logits)
            if mask.ndim == 3:        # (1,H,W) → (H,W)
                mask = mask[0]
            per_object[int(oid)][int(frame_idx)] = mask
    return per_object


def _propagate_bidirectional(predictor, state):
    """SAM2 **双向**传播（§4 M5 字段 7 / D-1）：先正向，再逆向补齐提示帧之前的帧。

    只正向传播时，在第 n-1 帧给框的对象拿不到之前的 mask（单帧漏检 = 对象不存在），
    这正是 C-9 要避免的失败模式。逆向 pass 不可用（旧版 API / 测试替身）时静默跳过：
    反向缺口由 3D 质心去重与多探测帧兜底，不影响正确性。
    """
    yielded = False
    # C-4：mask decoder 在 bf16 下计算，传播**也必须**在 autocast 内，否则
    # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"。
    with _autocast_ctx():
        for item in predictor.propagate_in_video(state):
            yielded = True
            yield item
        try:
            for item in predictor.propagate_in_video(state, reverse=True):
                yield item
        except TypeError:
            # 替身/旧版签名不支持 reverse → 只保留正向结果
            return
        except Exception:  # noqa: BLE001 - 逆向失败不阻断（正向结果已产出）
            if not yielded:
                raise
            return


def _autocast_ctx():
    """官方 SAM2 demo 的全局 bf16 autocast 上下文（CUDA 可用时）。

    SAM2 的 mask decoder 内部按 autocast 计算，而 prompt/权重为 fp32；
    不在外层进入 autocast 会触发 dtype 不一致（见 sam2 README 的推理示例）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", dtype=torch.bfloat16)
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)  # pragma: no cover
        return contextlib.nullcontext()
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


def _prompt_box(box: Sequence[float], predictor):
    """构造与模型 dtype 一致的 box 提示张量（SAM2 常以 bf16 运行）。"""
    dtype = None
    try:
        params = list(predictor.parameters())      # type: ignore[attr-defined]
        if params:
            dtype = params[0].dtype
        elif getattr(predictor, "device", None) is None:
            dtype = None
    except Exception:  # noqa: BLE001 - fake predictor 无 parameters()
        dtype = None
    try:
        import torch

        return torch.as_tensor(np.asarray(box, dtype=np.float32),
                               dtype=dtype or torch.float32)
    except Exception:  # noqa: BLE001 - 无 torch 时退回 ndarray
        return np.asarray(box, dtype=np.float32)


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int],
                    transform: Optional[dict] = None) -> np.ndarray:
    """把掩码最近邻缩放到目标 (H, W)（§9 坐标层：SAM2-mask → VGGT-depth-grid）。

    v6 §20 已废止 BA / 正方形中心 pad 路线 → **唯一正确的映射是纯等比缩放**。
    `transform` 形参保留只为签名兼容（`bind_masks_for_scene` / runner 仍按旧签名
    传值），**不再应用 pad 仿射**：pad 形态的深度网格在 v6 不存在 ——
    `ReconstructionArtifact.grid_transform` 已是 legacy-only 字段（出现即 hard fail），
    故迁移后调用方只会传 `None`，本函数行为与旧实现的 `None` 分支逐位一致。
    """
    from skill3d.coords import mask_to_grid

    del transform          # v6 §20：square-pad 已废止，纯缩放是唯一正确映射
    return mask_to_grid(mask, None, shape=(int(shape[0]), int(shape[1])))


def _to_bool_mask(logits) -> np.ndarray:
    """SAM2 输出（可能是 CUDA tensor）→ 布尔 mask ndarray。

    已核验：`propagate_in_video` 的 `video_res_masks` 是 **GPU 上的 torch.Tensor**，
    直接 `np.asarray` 会抛 "can't convert cuda:0 device type tensor to numpy"。
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    return np.asarray(logits) > 0.0


def _write_jpeg_sequence(frames: Sequence[np.ndarray],
                         quality: int = 92) -> str:
    """把帧写成连续编号的 JPEG 目录（SAM2 `init_state` 的唯一非-MP4 输入形式）。

    返回目录路径。放在临时目录下（M10 沙箱纪律：不往产物目录写中间态）。
    """
    import tempfile

    import cv2

    d = Path(tempfile.mkdtemp(prefix="skill3d_sam2_"))
    for i, fr in enumerate(frames):
        img = np.asarray(fr)
        cv2.imwrite(str(d / f"{i:05d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return str(d)


# ------------------------------------------------------------------ G7 动态 mask ----

def dynamic_mask_from_rigidity(
    frames: Sequence[np.ndarray],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    *,
    th_residual_px: float = TH_RIGIDITY_RESIDUAL_PX,
    mad_k: float = RIGIDITY_MAD_K,
) -> tuple[np.ndarray, list[str]]:
    """G7 动态 mask：由**自运动刚性残差**判定动态像素（确定性，无额外模型）。

    对相邻帧 (t, t+1)：用 depth+位姿把帧 t 的像素投影到帧 t+1 得"自运动预测光流"，
    与 Farneback 观测光流比较；残差超阈的像素记为动态（相机静止/纯自运动场景该
    残差为 0）。

    **阈值是自适应 + 绝对下界**：光流估计本身有噪声地板（实测在平滑纹理上
    ~3–5px），固定阈值会大面积误报。故取
    `thr = max(th_residual_px, median(res) + mad_k × 1.4826 × MAD(res))`，
    以场景自身的残差分布标定噪声地板（对"多数像素静止"的常见情形成立）。

    TODO_CALIBRATE：`th_residual_px` 下界与 `mad_k`；真实手持视频的噪声地板更高，
    需在自有数据上重新标定（§15.2）。
    """
    import cv2

    depth = np.asarray(depth_maps, dtype=np.float64)
    c2w = np.asarray(c2w_list, dtype=np.float64)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    n = min(len(frames), len(depth), len(c2w))
    notes: list[str] = []
    if n < 2:
        return np.zeros((0, 0, 0), dtype=bool), ["帧数不足 2，无法估计动态 mask"]

    h, w = depth.shape[1], depth.shape[2]
    dyn = np.zeros((n, h, w), dtype=bool)
    thr_used: list[float] = []
    # 光流必须与深度在同一像素网格：VGGT 会改动帧分辨率（如 640x480 竖屏视频 →
    # 518x392 预处理网格），直接在原始帧上算光流会导致坐标越界/错位。
    gray = [cv2.resize(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY), (w, h),
                       interpolation=cv2.INTER_AREA)
            for f in frames[:n]]
    for t in range(n - 1):
        k = k_all[t] if k_all.ndim == 3 else k_all
        z = depth[t]
        try:
            obs = cv2.calcOpticalFlowFarneback(
                gray[t], gray[t + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        except Exception as exc:  # noqa: BLE001 - 分辨率/类型异常 → 跳过该帧对
            notes.append(f"帧对 {t}->{t+1} 光流失败（{type(exc).__name__}）")
            continue
        # 自运动预测：X_{t+1} = T_{t+1<-t} X_t，T = inv(c2w_{t+1}) @ c2w_t
        t_rel = np.linalg.inv(c2w[t + 1]) @ c2w[t]
        vv, uu = np.nonzero(np.isfinite(z) & (z > 1e-6))
        if vv.size == 0:
            continue
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=0).astype(np.float64)
        k_inv = np.linalg.inv(k)
        xyz = (k_inv @ pix) * z[vv, uu][None, :]
        moved = t_rel[:3, :3] @ xyz + t_rel[:3, 3:4]
        depth_next = moved[2]
        valid = depth_next > 1e-6
        if not valid.any():
            continue
        proj = k @ (moved[:, valid] / depth_next[valid])
        u2 = np.clip(np.round(proj[0]).astype(int), 0, w - 1)
        v2 = np.clip(np.round(proj[1]).astype(int), 0, h - 1)
        pred_u = proj[0] - uu[valid]
        pred_v = proj[1] - vv[valid]
        res = np.hypot(obs[v2, u2, 0] - pred_u, obs[v2, u2, 1] - pred_v)
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        thr = max(float(th_residual_px), med + float(mad_k) * 1.4826 * mad)
        thr_used.append(thr)
        dyn[t][vv[valid], uu[valid]] = res > thr
    ratio = float(dyn[: n - 1].mean()) if n > 1 else 0.0
    if thr_used:
        notes.append(f"G7 刚性残差动态占比={ratio:.4f}（自适应阈值 "
                     f"min={min(thr_used):.2f}px max={max(thr_used):.2f}px，下界 "
                     f"{th_residual_px}px/U+2B k={mad_k}，TODO_CALIBRATE）")
    if ratio > TH_DYNAMIC_RATIO:
        notes.append(f"G7 动态占比超阈 {TH_DYNAMIC_RATIO} → 记重建污染（§10 G7）")
    return dyn, notes


# ------------------------------------------------------------------ 绑定世界点云 ----

def _as_point_conf(point_conf, depth_maps: np.ndarray) -> Optional[np.ndarray]:
    """规整逐点置信度到与 `depth_maps` 同形的 (N,H,W)；不可用 → `None`。

    **不伪造**：形状对不上（旧产物/别的网格上的 conf）一律返回 None，
    由调用方写空 ref（§12.2：conf 不可用时不得当作 1.0 参与加权）。
    """
    if point_conf is None:
        return None
    arr = np.asarray(point_conf, dtype=np.float64)
    d = np.asarray(depth_maps)
    if d.ndim != 3:
        return None
    if arr.ndim == 2 and arr.shape == d.shape[1:]:
        arr = np.broadcast_to(arr, (d.shape[0], arr.shape[0], arr.shape[1]))
    if arr.ndim != 3 or arr.shape != d.shape:
        return None
    return arr


def bind_masks_to_world(
    masks_per_object: Sequence[dict[int, np.ndarray]],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    class_hints: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    max_points: int = 20000,
    out_dir: Optional[str | Path] = None,
    scene_name: str = "scene",
    grid_transform: Optional[dict] = None,
    point_conf: Optional[np.ndarray] = None,
    track_prefix: str = TRACK_PREFIX_BASE_LIST,
    grounding_status: str = "base_list",
) -> list[ObjectRecord]:
    """mask × 深度反投影 → 世界坐标点云绑定（numpy 真实实现，§4 M5）。

    - masks_per_object: per-object {frame_idx: mask(H,W)}
    - depth_maps: (N,H,W) 逐帧深度（与相机坐标系对齐，正值）
    - c2w_list: (N,4,4) 世界坐标 SE(3)
    - intrinsics: (N,3,3) 或 (3,3) 相机内参
    - out_dir: 给出时把逐帧 mask 与对象点云落盘并填 `ObjectRecord` 的 ref 字段
      （M3 尺度锚定与 M4 覆盖率都需要从 ref 回读对象点云）
    - point_conf: (N,H,W) 或 (H,W) **逐点** VGGT 置信度（与 `depth_maps` 同网格）。
      给出时按与点云**同一套有效掩码/同一套下采样下标**取点，落盘 `pointconf_world`
      并与 `pointcloud_world` 逐点 1:1 对齐；**不给则写空 ref**（不伪造置信度）。
    - track_prefix: `track_id` 命名空间（基础清单 `trk*` / 逐题补漏 `qtrk*`）；
      `track_id = f"{track_prefix}{传播序号}"` —— 确定性、可复现，合并组内统一为
      幸存代表的 track（见 `_dedupe_by_world_centroid`）。
    - grounding_status: §5.6 的 `base_list` / `question_targeted_fill`。

    `centroid_ref` 留空串：质心已内联在 `centroid_world`（同一批数字不写第二份文件，
    避免两处事实源漂移），**不写假路径**。
    """
    depth_maps = np.asarray(depth_maps, dtype=np.float64)
    c2w_list = np.asarray(c2w_list, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics, (len(c2w_list), 3, 3))
    pconf = _as_point_conf(point_conf, depth_maps)

    instances: list[ObjectRecord] = []
    for prop_idx, masks in enumerate(masks_per_object):
        # 有效支撑帧（掩码非空）→ 供"逐帧可见性"类题型程序化作答
        vis_frames = sorted(int(fi) for fi, m in masks.items()
                            if np.asarray(m).any())
        hint = class_hints[prop_idx] if class_hints and prop_idx < len(class_hints) else "unknown"
        pts_world_all = []
        conf_all = []
        for frame_idx, mask in sorted(masks.items()):
            if frame_idx >= len(depth_maps):
                continue
            depth = depth_maps[frame_idx]
            K = intrinsics[min(frame_idx, len(intrinsics) - 1)]
            c2w = c2w_list[min(frame_idx, len(c2w_list) - 1)]
            # SAM2 掩码在原始视频分辨率，VGGT 深度在预处理分辨率（518×392；v6 §20
            # 已废止 BA 的 518×518 正方形 pad 网格）→ 按 §9 纯等比最近邻映射到深度
            # 网格（硬约束：以重建产物为几何参考）。**无条件调用**：尺寸相同时
            # `mask_to_grid` 原样返回。
            m = _resize_mask_to(np.asarray(mask, dtype=bool), depth.shape, grid_transform)
            vs, us = np.nonzero(m)
            z = depth[vs, us]
            valid = np.isfinite(z) & (z > 0)
            vs, us, z = vs[valid], us[valid], z[valid]
            if z.size == 0:
                continue
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # 相机坐标系反投影
            x = (us - cx) * z / fx
            y = (vs - cy) * z / fy
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4,M)
            pts_world = (c2w @ pts_cam)[:3].T                        # (M,3)
            pts_world_all.append(pts_world)
            if pconf is not None:
                conf_all.append(pconf[frame_idx][vs, us])            # 与 pts 同序同长

        iid = f"obj_{prop_idx}"
        # v6 §5.6：跨帧 track 身份 = 产生该对象的 SAM2 传播序号（确定性）
        track_id = f"{track_prefix}{prop_idx}"
        if not pts_world_all:
            # 分割失败 → 该对象标记 unverified（§4 M5 字段 9）
            instances.append(ObjectRecord(
                obj_id=iid,
                category_name=hint if class_hints else "unverified",
                track_id=track_id,
                grounding_status=grounding_status,
                mask_per_frame="",
                pointcloud_world="",
                pointconf_world="",          # 无点云 → 无逐点 conf（不伪造）
                centroid_ref="",
                centroid_world=[0.0, 0.0, 0.0],
                bbox=[0.0] * 6,
                det_conf=0.0,
                visible_frames=vis_frames,
            ))
            continue

        pts = np.concatenate(pts_world_all, axis=0)
        conf_pts = np.concatenate(conf_all, axis=0) if conf_all else None
        if len(pts) > max_points:
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[sel]
            if conf_pts is not None:
                conf_pts = conf_pts[sel]     # 同一套下标 → 下采样后仍逐点对齐
        centroid = pts.mean(axis=0)
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        # 置信度：优先用外部（SAM2 打分），否则按有效帧占比估计
        if confidences is not None and prop_idx < len(confidences):
            conf = float(confidences[prop_idx])
        else:
            conf = min(1.0, len(masks) / max(1, len(depth_maps)))

        mask_ref, cloud_ref, conf_ref = "", "", ""
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            mask_arr = _stack_masks(masks, len(depth_maps), depth_maps.shape[1:])
            mask_path = d / f"{scene_name}_{iid}_mask.npy"
            cloud_path = d / f"{scene_name}_{iid}_points.npy"
            np.save(mask_path, mask_arr)
            np.save(cloud_path, pts)
            mask_ref, cloud_ref = str(mask_path), str(cloud_path)
            if conf_pts is not None:
                conf_path = d / f"{scene_name}_{iid}_pointconf.npy"
                np.save(conf_path, conf_pts)     # 与 pts 同长（逐点对齐）
                conf_ref = str(conf_path)

        instances.append(ObjectRecord(
            obj_id=iid,
            category_name=hint,
            track_id=track_id,
            grounding_status=grounding_status,
            mask_per_frame=mask_ref,
            pointcloud_world=cloud_ref,
            pointconf_world=conf_ref,
            centroid_ref="",
            centroid_world=centroid.tolist(),
            bbox=(list(pmin) + list(pmax)),
            det_conf=conf,
            visible_frames=vis_frames,
        ))
    return instances


def _stack_masks(masks: dict[int, np.ndarray], n_frames: int,
                 shape: Sequence[int]) -> np.ndarray:
    """把 per-frame mask 堆成 (N,H,W) uint8（缺帧置 0），便于落盘与 G9 统计。

    **必须按 `_resize_mask_to` 映射到深度网格后再堆**（2026-09-22 实测缺陷）：
    SAM2 掩码在**原始视频分辨率**，本函数的 `shape` 来自 `depth_maps.shape[1:]`
    （VGGT 深度网格 518×392）。旧实现写的是 `if np.asarray(m).shape == out.shape[1:]`
    —— 尺寸不等就**静默跳过**，于是每个对象的 `*_mask.npy` 落盘成**整片 0**
    （实测 7 个 scene、全部 54 个对象的 mask 数组 sum 均为 0），而清单里的
    `visible_frames` 由**未缩放的** mask 算出、依然非空 →
    落盘产物与清单自相矛盾，审计与回读（G9/掩码类复算）全部失真。
    纯缩放是 v6 §20 的唯一正确映射（square-pad 仿射已废止）。
    """
    out = np.zeros((n_frames, int(shape[0]), int(shape[1])), dtype=np.uint8)
    for idx, m in masks.items():
        if not (0 <= idx < n_frames):
            continue
        mm = _resize_mask_to(np.asarray(m, dtype=bool), out.shape[1:])
        if mm.shape == out.shape[1:]:
            out[idx] = np.asarray(mm, dtype=np.uint8)
    return out


def _inventory_key(scene_name: str, frame_set_hash: str, depth_maps, n_frames: int) -> str:
    """场景清单缓存键（确定性）：scene + frame_set_hash + 产物形状。

    `frame_set_hash` 由调用方传入（M1 统一 FrameSet 的哈希）；缺失时用形状兜底，
    仍然确定性，只是粒度更粗（换帧集但形状相同的场景会误命中 → 调用方应尽量传）。
    """
    import hashlib

    payload = json.dumps({
        "v": _INVENTORY_VERSION,
        "scene": str(scene_name),
        "fsh": str(frame_set_hash or ""),
        "n_frames": int(n_frames),
        "depth_shape": list(np.asarray(depth_maps).shape) if depth_maps is not None else None,
        "vocab": sorted(OBJECT_VOCABULARY),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_inventory(out_dir: Optional[str | Path], scene_name: str,
                    key: str) -> Optional[tuple[list[ObjectRecord], dict, str]]:
    """读场景清单缓存；返回 `(objects, stats_scalars, dynamic_masks_ref)` 或 None。"""
    if not out_dir:
        return None
    p = Path(out_dir) / f"{scene_name}_inventory_{key}.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        objects = [ObjectRecord.model_validate(o) for o in payload["objects"]]
    except Exception:  # noqa: BLE001 - 缓存损坏 → 当作未命中重算（不抛错阻断 episode）
        return None
    # mask/点云/逐点 conf ref 必须仍然存在，否则缓存不可用（防止半写入/清理过的目录）
    for o in objects:
        if o.mask_per_frame and not Path(o.mask_per_frame).exists():
            return None
        if o.pointcloud_world and not Path(o.pointcloud_world).exists():
            return None
        if o.pointconf_world and not Path(o.pointconf_world).exists():
            return None
    stats = dict(payload.get("stats") or {})
    g7_ref = str(payload.get("dynamic_masks_ref") or "")
    if g7_ref:
        if not Path(g7_ref).exists():
            g7_ref = ""
        else:
            stats["dynamic_masks_ref"] = g7_ref
    return objects, stats, g7_ref


def _save_inventory(out_dir: Optional[str | Path], scene_name: str, key: str,
                    objects: Sequence[ObjectRecord], stats: dict,
                    dynamic_masks: Optional[np.ndarray]) -> Optional[str]:
    """写场景清单缓存（原子写）；返回 dynamic_masks ref（未给则空串）。"""
    if not out_dir:
        return ""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    g7_ref = ""
    if dynamic_masks is not None and np.asarray(dynamic_masks).size:
        g7_ref = str(d / f"{scene_name}_inventory_{key}_dyndyn.npy")
        np.save(g7_ref, np.asarray(dynamic_masks))
    payload = {
        "inventory_version": _INVENTORY_VERSION,
        "scene_name": str(scene_name),
        "objects": [o.model_dump() for o in objects],
        "stats": {k: v for k, v in stats.items() if k != "dynamic_masks"},
        "dynamic_masks_ref": g7_ref,
    }
    p = d / f"{scene_name}_inventory_{key}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    tmp.replace(p)
    return g7_ref


def _detector_boxes(frames: Sequence[np.ndarray], probe: Sequence[int],
                    text_prompt: str, notes: list[str]) -> list[tuple[int, str, list[float]]]:
    """检测器探测（**失败要出声**）：返回 `(frame_idx, label, box)` 列表。

    历史缺陷（2026-09-21 实测）：`ovd.detect` 在服务不可用/超时时**静默返回 `[]`**，
    调用方只在"新增了框"时才记 note → 检测器整段不可用时对象清单悄悄退化成
    "只有 VLM 框"（outer_holdout 实测：同一 scene 一题只剩 2 个对象，程序路径直接不可答），
    而日志里看不出任何异常。这里改成：整轮零检测 → 重试一次；仍为零则按
    `ovd.detect.last_error` 如实区分"服务故障降级"与"本轮确实没检出"。
    """
    from skill3d.segmentation import open_vocab_detector as ovd

    if not ovd.available() or not text_prompt:
        return []
    out: list[tuple[int, str, list[float]]] = []
    for attempt in (1, 2):
        out = []
        for fi in probe:
            for d in ovd.detect(frames[fi], text_prompt):
                x0, y0, x1, y1 = d.bbox_xyxy
                out.append((int(fi), d.label or "object", [x0, y0, x1, y1]))
        if out:
            if attempt > 1:
                notes.append("M5 检测器首次整轮零检测 → 重试后恢复")
            return out
    err = str(getattr(ovd.detect, "last_error", "") or "")
    if err:
        notes.append(
            f"M5 检测器（{ovd.detector_endpoint()}）连续 2 轮探测均返回 0 个框"
            f"（last_error={err}）→ **检测器侧降级**，对象清单仅来自 VLM 框；"
            "对象类 Tool 的可用性随之下降（不得据此判定场景中不存在某物体）")
    else:
        notes.append(f"M5 检测器（{ovd.detector_endpoint()}）连续 2 轮探测均无检出"
                     "（服务正常，本轮确实没检出）")
    return out


def bind_objects_for_scene(
    frames: Sequence[np.ndarray],
    handle,
    question: str = "",
    *,
    depth_maps: Optional[np.ndarray] = None,
    c2w_list: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    grid_transform: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
    scene_name: str = "scene",
    predictor=None,
    vlm_client=None,
    frame_set_hash: str = "",
    use_inventory_cache: bool = True,
    seed: Optional[int] = None,
    point_conf: Optional[np.ndarray] = None,
    depth_conf: Optional[np.ndarray] = None,
) -> tuple[list[ObjectRecord], dict, list[str]]:
    """M5 端到端（在线）：提示 → SAM2 传播 → 世界点云绑定 → G7/G9 统计。

    返回 `(objects, stats, notes)`；`stats` 键：`track_ious`(G9)、
    `dynamic_ratio`(G7)、`dynamic_masks`(G7)，
    直接喂给 M4 的 `quality_gate`（G-18 数据源接线）。

    **场景清单（v5.1 修订，2026-09-21）**：对象清单是 **scene 级产物**，由
    ① 确定性检测器（完整室内词表，**与问题无关**）与 ② VLM 通用清单提示共同产生，
    按 `(scene, frame_set_hash)` 缓存复用；③ 问题里的目标物再走一次**逐题补漏**
    （只补清单里没有的），从而：
    - 同一 scene 的多个 episode 看到**同一份对象清单**（id 稳定、可复现）；
    - 检测器一旦不可用不再悄悄让清单塌缩（见 `_detector_boxes`）；
    - 重复跑同一配置（噪声底/多 seed）不必重付 M5 的 GPU 成本。

    v6 证据输入：
    - `point_conf` / `depth_conf`：与 `depth_maps` 同网格的逐点 VGGT 置信度
      （`point_conf` 优先；都缺省 → `pointconf_world` 写空串，不伪造），
      随对象点云逐点落盘供 §12.2 的距离原语做软权重；
    - 基础清单对象 `grounding_status="base_list"`、`track_id` 前缀 `trk`；
      逐题补漏对象 `grounding_status="question_targeted_fill"`、前缀 `qtrk`；
    - `stats["track_stable_ratio"]`：跨帧 track 稳定占比（每条 track 可见帧数/帧集帧数
      的均值）→ §7.1 `track_consensus`；`count_objects` 依赖它是 available/degraded
      而不是 unavailable（§7.2：required=unavailable 会直接隐藏该 Tool）。

    注意：`point_conf` 只在**首次**（未命中缓存）计算时落盘 —— 命中缓存直接复用
    清单与 ref，不会重算/改写（缓存键已含产物形状，换网格即失效重算）。
    """
    from skill3d.reconstruction_gate.confidence_map import track_ious_from_masks

    notes: list[str] = []
    stats: dict = {}
    if depth_maps is None or c2w_list is None:
        return [], {}, ["M5 跳过：无深度/位姿产物，无法把 mask 反投影到世界系"]
    if point_conf is None:
        point_conf = depth_conf          # 兼容两种命名；都缺省 → 不落 conf

    n_frames = len(frames)
    # 4 个时间均匀探测帧（D-1）：单帧常常看不到目标物（实测同一视频帧 8/31 有桌子、
    # 帧 0/16/24 没有）。
    probe = sorted({0, n_frames // 3, 2 * n_frames // 3, n_frames - 1}) if n_frames else []
    key = _inventory_key(scene_name, frame_set_hash, depth_maps, n_frames)

    objects: list[ObjectRecord] = []
    cached_stats: dict = {}
    if use_inventory_cache:
        cached = _load_inventory(out_dir, scene_name, key)
        if cached is not None:
            objects, cached_stats, g7_ref = cached
            stats.update({k: v for k, v in cached_stats.items() if k != "dynamic_masks_ref"})
            if g7_ref:
                stats["dynamic_masks"] = np.load(g7_ref)
            notes.append(
                f"M5 场景清单缓存命中（{scene_name}/{key}）：直接复用 {len(objects)} 个对象"
                f"与 G7/G9 统计（零重算；对象是 scene 级产物）")

    if not objects:
        # ---- ① + ② 场景清单（与问题无关）：检测器全词表 ∪ VLM 通用清单 ----
        hints: list[str] = []
        boxes: list[list[float]] = []
        prompt_list: list[tuple[int, int, list[float]]] = []
        if vlm_client is not None and n_frames:
            inv_hints, inv_boxes, used_frames = _vlm_inventory_pass(frames, probe,
                                                                   vlm_client, seed=seed)
            for (fi, h, b) in zip(used_frames, inv_hints, inv_boxes):
                hints.append(h)
                boxes.append(b)
                prompt_list.append((fi, len(prompt_list), b))
            if inv_hints:
                notes.append(f"M5 VLM 通用清单：{len(inv_hints)} 个框"
                             f"（{sorted(set(inv_hints))[:8]}）")
        text_prompt = ovd_prompt_full_vocabulary()
        det = _detector_boxes(frames, probe, text_prompt, notes)
        for (fi, label, b) in det:
            hints.append(label)
            boxes.append(b)
            prompt_list.append((fi, len(prompt_list), b))
        if det:
            notes.append(f"M5 检测器（全词表 {len(OBJECT_VOCABULARY)} 类）在 {len(probe)} 个"
                         f"探测帧给出 {len(det)} 个框（{sorted(set(h for _, h, _ in det))[:8]}）")
        if not boxes:
            h_hints, h_boxes = object_prompts_from_handle(handle)
            if h_boxes:
                hints, boxes = h_hints, h_boxes
                prompt_list = [(0, i, b) for i, b in enumerate(boxes)]
        if not boxes and n_frames:
            notes.append("M5 场景清单为空（VLM 未给框且检测器不可用）→ 对象类 Tool 将 "
                         "fail-closed（不伪造'场景里没有'，硬约束 23）")

        masks_per_object = track_objects(frames, boxes or None, predictor,
                                         prompts=prompt_list or None,
                                         handle=handle, question=question)
        objects = bind_masks_to_world(masks_per_object, depth_maps, c2w_list, intrinsics,
                                      class_hints=hints or None, out_dir=out_dir,
                                      scene_name=scene_name, grid_transform=grid_transform,
                                      point_conf=point_conf,
                                      track_prefix=TRACK_PREFIX_BASE_LIST,
                                      grounding_status="base_list")
        # 去重（§3 M5 字段 5）：类别 + 世界质心距离 + 时序重叠三判据
        objects, n_dup, groups = _dedupe_by_world_centroid(
            objects, depth_maps, masks_per_object=masks_per_object)
        kept_masks = _merge_mask_groups(masks_per_object, groups)
        n_suspect = sum(1 for o in objects if o.duplicate_suspect)
        notes.append(f"M5 场景清单绑定 {len(objects)} 个对象"
                     f"（SAM2 传播 {len(masks_per_object)} 条，3D 去重 {n_dup} 个，"
                     f"重复嫌疑 {n_suspect} 个 → count_objects 按 track 共识计数）")
        ious = track_ious_from_masks(kept_masks)
        if ious:
            stats["track_ious"] = ious
        ratio = track_stable_ratio(objects, n_frames)
        if ratio is not None:
            stats["track_stable_ratio"] = ratio
        tcm = track_consensus_metrics(objects, n_propagations=len(masks_per_object),
                                      n_frames=n_frames)
        if tcm is not None:
            stats["track_consensus"] = tcm
        dyn, dnotes = dynamic_mask_from_rigidity(frames, depth_maps, c2w_list, intrinsics)
        notes.extend(dnotes)
        if dyn.size:
            stats["dynamic_ratio"] = float(dyn.mean())
            stats["dynamic_masks"] = dyn
        if ratio is not None:
            notes.append(f"M5 track 稳定占比={ratio:.3f}"
                         f"（每条 track 可见帧数/帧集帧数的均值 → §7.1 track_consensus）")
        if use_inventory_cache:
            _save_inventory(out_dir, scene_name, key, objects,
                            {k: v for k, v in stats.items() if k != "dynamic_masks"},
                            stats.get("dynamic_masks"))
    else:
        # 缓存命中：G9/G7 是 scene 级统计，直接复用缓存值（不重算、不回读掩码）——
        # 与首次计算口径完全一致；回读掩码重算会把"全零帧"算进 IoU 对，反而偏。
        if "track_consensus" not in stats:
            tcm = track_consensus_metrics(objects, n_propagations=len(objects),
                                          n_frames=n_frames)
            if tcm is not None:
                stats["track_consensus"] = tcm
        if "track_stable_ratio" not in stats:
            # 老缓存缺该键（或调用方手写 stats）→ 由清单现算，保证证据不缺项
            ratio = track_stable_ratio(objects, n_frames)
            if ratio is not None:
                stats["track_stable_ratio"] = ratio
        notes.append(f"M5 复用缓存 G7/G9 统计（track_ious="
                     f"{'有' if stats.get('track_ious') else '无'}，"
                     f"dynamic_ratio={'有' if 'dynamic_ratio' in stats else '无'}，"
                     f"dynamic_masks={'有' if 'dynamic_masks' in stats else '无'}）")

    # ---- ③ 逐题补漏：问题里的目标物若不在清单中，补绑一次 ----
    if vlm_client is not None and n_frames and question:
        sup_objs, sup_stats, sup_note = _bind_question_supplement(
            frames, probe, question, objects, depth_maps, c2w_list, intrinsics,
            grid_transform=grid_transform, out_dir=out_dir, scene_name=scene_name,
            predictor=predictor, vlm_client=vlm_client, handle=handle, seed=seed,
            point_conf=point_conf)
        if sup_note:
            notes.append(sup_note)
        if sup_stats.get("grounding"):
            stats["grounding"] = dict(sup_stats["grounding"])
        if sup_objs:
            objects = list(objects) + sup_objs
            extra = [float(v) for v in (sup_stats.get("track_ious") or [])]
            if extra:
                prev = list(stats.get("track_ious") or [])
                stats["track_ious"] = prev + extra

    if stats.get("track_ious"):
        notes.append(f"G9 跟踪 IoU 均值={float(np.mean(stats['track_ious'])):.3f}"
                     "（去重后对象集）")
    return objects, stats, notes


def ovd_prompt_from_nouns(nouns: Sequence[str]) -> str:
    """共用入口：名词 → 检测器文本 prompt（单数化 + 去重 + 保序）。"""
    from skill3d.segmentation import open_vocab_detector as ovd

    sing = [_singularize(n) for n in nouns if str(n).strip()]
    return ovd.prompt_from_nouns(list(dict.fromkeys(sing))) if sing else ""


def ovd_prompt_full_vocabulary() -> str:
    """场景清单用的**完整**室内词表 prompt（与问题无关 → 可缓存、可复现）。

    历史缺陷（2026-09-21 实测）：旧实现只把 `OBJECT_VOCABULARY[:12]` 补进 prompt，
    且以问题名词为主 → 与问题无关的类别（chair/table/counter…）大量缺失，
    计数题的 `count_objects` 直接失真。这里改用完整词表。
    """
    from skill3d.segmentation import open_vocab_detector as ovd

    return ovd.prompt_from_nouns(list(OBJECT_VOCABULARY))


def _vlm_inventory_pass(frames: Sequence[np.ndarray], probe: Sequence[int], vlm_client,
                        *, seed: Optional[int] = None
                        ) -> tuple[list[str], list[list[float]], list[int]]:
    """VLM **通用清单**（与问题无关）：在 4 个探测帧上问"有哪些主要物体"。

    与逐题补漏（`_bind_question_supplement`）配合：通用清单保证 scene 级清单完整、
    可缓存；补漏保证题目点名的物体一定在清单里。
    """
    hints: list[str] = []
    boxes: list[list[float]] = []
    used: list[int] = []
    for fi in probe:
        h, b = box_prompts_from_vlm(frames, _GENERIC_INVENTORY_QUESTION, vlm_client,
                                    frame_idx=fi, max_objects=12, seed=seed)
        for hi, bi in zip(h, b):
            hints.append(hi)
            boxes.append(bi)
            used.append(int(fi))
    return hints, boxes, used


def question_object_names_from_vlm(question: str, client, *,
                                   seed: Optional[int] = None,
                                   max_names: int = 6) -> list[str]:
    """用在线 VLM 从题面抽出**被点名的物体名**（不依赖正则句式，覆盖 VSI-Bench 各种问法）。

    为什么不让正则抽：官方问法里 "If I am standing by the telephone and facing the cup,
    is the trash can to my left…"、"You are a robot beginning at the door with the clock
    above it…" 都不落在既有 `_GENERIC_NOUN_PATTERNS` 里 → 正则给空 → 补漏框为空 →
    题目点名的物体没进清单 → 程序只能 abstain（inner_validation 1501–1504 实测）。
    这里只做"名词抽取"，不涉及几何、不涉及答案。
    """
    import json as _json
    import re as _re

    if client is None or not str(question).strip():
        return []
    prompt = (
        f"题目：{question}\n"
        f"请列出题目里**提到或指向的物体**（用于在室内照片中定位它们），"
        f"不超过 {max_names} 个，用常见的英文物体名，单数形式（如 telephone、cup、"
        'trash can）。严格只输出 JSON 数组，例如 ["telephone", "cup", "trash can"]。'
        "不要解释。"
    )
    try:
        text = client.chat([{"role": "user", "content": prompt}],
                           max_tokens=128, seed=seed)
    except Exception:  # noqa: BLE001 - 模型不可用 → 退回空表（调用方走通用提示）
        return []
    m = _re.search(r"\[.*\]", str(text), _re.DOTALL)
    if not m:
        return []
    try:
        items = _json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for it in items if isinstance(items, list) else []:
        name = str(it).strip().strip('"').lower()
        if 0 < len(name) <= 40 and name not in out:
            out.append(name)
    return out[:max_names]


def _bind_question_supplement(frames, probe, question, existing, depth_maps, c2w_list,
                              intrinsics, *, grid_transform, out_dir, scene_name,
                              predictor, vlm_client, handle, seed: Optional[int] = None,
                              point_conf=None
                              ) -> tuple[list[ObjectRecord], dict, str]:
    """逐题补漏：只把"清单里还没有"的问题目标物绑进来（不重算已有对象）。

    返回 `(新增对象, stats, note)`。判定"已有"用**类别 + 世界质心距离**（与 3D
    去重同源口径），不再对全量对象重跑 SAM2 —— 这是 M5 复用（handoff 优先级 2）
    的关键：同一 scene 的后续 episode 只需为补漏框付 GPU 成本。

    v6：新增对象标 `grounding_status="question_targeted_fill"`（§5.6 / §7.1
    `object_grounding` 的题级证据）；`track_id` 用 `qtrk*` 命名空间 —— 与基础清单的
    `trk*` 不重名，否则 `count_objects` 的 track 去重会把两条不同对象并成一条。
    """
    hints: list[str] = []
    boxes: list[list[float]] = []
    prompt_list: list[tuple[int, int, list[float]]] = []
    # ① 定向：先把题面点名的物体抽出来，再逐探测帧"只找这些物体"
    #    （通用问法提示实测会漏掉题目要用的目标物 → 程序只能 abstain）
    named = question_object_names_from_vlm(question, vlm_client, seed=seed)
    if named:
        ask = ("请在这张室内照片中定位以下物体（能看到几个就报几个，看不到的不要报）："
               + "、".join(named)
               + '。严格只输出 JSON 数组：[{"name": "物体名", "bbox": [x0,y0,x1,y1]}]，'
                 "坐标用 0-1000 归一化整数；若一个都看不到，输出 []。不要解释。")
        for fi in probe:
            f_hints, f_boxes = box_prompts_from_vlm(frames, ask, vlm_client,
                                                    frame_idx=fi, max_objects=8, seed=seed)
            for h, b in zip(f_hints, f_boxes):
                hints.append(h)
                boxes.append(b)
                prompt_list.append((int(fi), len(prompt_list), b))
    # ② 兜底：定向提示没抽到名字/没给框时，退回原来的"与问题相关"通用提示
    if not boxes:
        for fi in probe:
            f_hints, f_boxes = box_prompts_from_vlm(frames, question, vlm_client,
                                                    frame_idx=fi, max_objects=8, seed=seed)
            for h, b in zip(f_hints, f_boxes):
                hints.append(h)
                boxes.append(b)
                prompt_list.append((int(fi), len(prompt_list), b))
    if not boxes:
        # v6 证据：题面点名了物体但检测/VLM 一个框都没给出 → **明确未命中**。
        # 必须报给 EvidenceProfile 的 `object_grounding=unavailable`，
        # 不能让"没找到"静默退化成"没查"（§7.1 三值判定 + fail-closed）。
        return [], {"grounding": {"attempted": True, "n_boxes": 0,
                                  "n_new": 0, "all_present": False,
                                  "miss": True}}, (
            "M5 逐题补漏：题面点名物未检出（grounding_recall_miss）")
    masks = track_objects(frames, boxes, predictor, prompts=prompt_list,
                          handle=handle, question=question)
    cand = bind_masks_to_world(masks, depth_maps, c2w_list, intrinsics,
                               class_hints=hints or None, out_dir=out_dir,
                               scene_name=scene_name, grid_transform=grid_transform,
                               point_conf=point_conf,
                               track_prefix=TRACK_PREFIX_QUESTION,
                               grounding_status="question_targeted_fill")
    cand, n_dup, groups = _dedupe_by_world_centroid(cand, depth_maps,
                                                    masks_per_object=masks)
    d = np.asarray(depth_maps, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    tol = 0.15 * (float(np.median(d)) if d.size else 1.0)   # TODO_CALIBRATE：同 3D 去重口径

    def _hint(o) -> str:
        h = str(getattr(o, "category_name", "") or "").strip().lower()
        return "" if h in ("unverified", "unknown") else h

    kept: list[ObjectRecord] = []
    kept_masks: list[dict] = []
    base_ids = {o.obj_id for o in existing}
    for o, g in zip(cand, groups):
        co = np.asarray(o.centroid_world, dtype=np.float64)
        dup = False
        for b in existing:
            if np.linalg.norm(co - np.asarray(b.centroid_world, dtype=np.float64)) >= tol:
                continue
            ho, hb = _hint(o), _hint(b)
            if ho and hb and ho != hb:
                continue
            dup = True
            break
        if not dup:
            kept.append(o)
            kept_masks.append(_merge_mask_groups(masks, [g])[0])

    if not kept:
        # v6 证据：题面点名物**已在清单中确认存在**（框都能对上已有对象）→ 命中
        return [], {"grounding": {"attempted": True, "n_boxes": len(boxes),
                                  "n_new": 0, "all_present": True}}, (
            f"M5 逐题补漏：{len(boxes)} 个问题目标框均已在场景清单中"
            f"（复用，不重算）")
    out: list[ObjectRecord] = []
    for i, o in enumerate(kept):
        new_id = f"obj_{len(base_ids) + i}"
        # obj_id 换成 scene 内唯一的稳定 id；track_id / grounding_status /
        # duplicate_suspect 原样保留（track 由本次传播产生，`qtrk*` 命名空间）
        out.append(o.model_copy(update={"obj_id": new_id}))
    stats: dict = {"grounding": {"attempted": True, "n_boxes": len(boxes),
                                 "n_new": len(out), "all_present": False,
                                 "miss": False}}
    from skill3d.reconstruction_gate.confidence_map import track_ious_from_masks

    ious = track_ious_from_masks(kept_masks)
    if ious:
        stats["track_ious"] = ious
    return out, stats, (f"M5 逐题补漏：问题目标框 {len(boxes)} 个 → 新绑 {len(out)} 个"
                        f"（{sorted({o.category_name for o in out})}）")


def track_stable_ratio(objects: Sequence[ObjectRecord],
                       n_frames: int) -> Optional[float]:
    """可见帧率均值（**诊断量**，不再是 `track_consensus` 的判据；[TODO_CALIBRATE]）。

    定义：每条 track 的存活率 = 该对象在帧集里可见的帧数 / 帧集帧数，再对清单取均值。

    **2026-09-21 真实数据修正**：本函数曾是 `track_consensus` 的判据，但实测
    （scene 41069043）该值只有 0.14–0.19 —— 手持扫描里单个物体本来只在少数帧出现，
    这个量回答的是"物体可见多久"，**不是**"同一实例是否被跨帧一致地跟成一条 track"。
    用它当门会让 `count_objects` 结构性不可用（counting 题的程序路径直接没了），
    而 v5 实测的计数失败根因是**重复实例**（清单 12 个重复）与**碎片化**，
    不是"可见帧少"。故改为记录为诊断量，判据见 `track_consensus_metrics`。
    """
    if n_frames <= 0 or not objects:
        return None
    vals = [min(1.0, len(set(int(i) for i in (o.visible_frames or []))) / float(n_frames))
            for o in objects]
    return float(sum(vals) / len(vals))


def track_consensus_metrics(objects: Sequence[ObjectRecord], *,
                            n_propagations: int,
                            n_frames: int) -> Optional[dict]:
    """`track_consensus` 的判据输入（§7.1；阈值 `[TODO_CALIBRATE]`）。

    两个信号，直接对应 v5 [已实测] 的计数失败模式：

    - `track_fragmentation_ratio` = `n_dup / n_propagations`：SAM2 传播了多少条
      被 3D 去重合并掉的**碎片**。越高说明同一实例被切成多条 track 越严重；
    - `duplicate_suspect_ratio` = `n_suspect / n_objects`：去重后仍留在清单里的
      重复嫌疑对象占比。越高说明"清单长度"越不能当计数答案（v5 实测 12 个重复）。

    同时保留 `track_stable_ratio`（可见帧率）作**诊断**，便于事后比较两种口径。

    `n_propagations <= 0` 或无对象 → `None`（证据侧按 unavailable，**不伪造 0**）。
    """
    if n_propagations <= 0 or not objects:
        return None
    n_objects = len(objects)
    n_suspect = sum(1 for o in objects if bool(getattr(o, "duplicate_suspect", False)))
    n_dup = max(0, int(n_propagations) - n_objects)
    return {
        "track_fragmentation_ratio": float(n_dup) / float(n_propagations),
        "duplicate_suspect_ratio": float(n_suspect) / float(n_objects),
        "n_propagations": int(n_propagations),
        "n_objects": int(n_objects),
        "n_suspect": int(n_suspect),
        "track_stable_ratio": track_stable_ratio(list(objects), n_frames),
    }


def _with_dedup_verdict(o, *, suspect: bool, track_id: str):
    """把 v6 去重结论（`duplicate_suspect` / `track_id`）写回对象。

    优先 `model_copy`（不改动调用方传入的原对象）；鸭子类型替身退回 `setattr`。
    """
    updates = {"duplicate_suspect": bool(suspect), "track_id": track_id}
    if hasattr(o, "model_copy"):
        return o.model_copy(update=updates)
    for k, v in updates.items():                        # pragma: no cover - 替身
        setattr(o, k, v)
    return o


def _dedupe_by_world_centroid(objects, depth_maps, *, masks_per_object=None,
                              rel_tol: float = 0.15):
    """3D 去重（§3 M5 字段 5）：**类别 + 世界质心距离 + 时序重叠**三判据。

    - 类别：`category_name` 大小写不敏感相等（任一侧为空/unknown 视为通配，不否决）；
    - 质心：距离 < `场景深度中位 × rel_tol`（[TODO_CALIBRATE] 起始参考 0.15）；
    - 时序重叠：两份候选的支撑帧集合有交集（同一时刻同位置才判同一实例；
      支撑帧未知时退化为只看前两条）。

    三判据**同时**满足才合并 —— 只按质心合并会误合并"先后出现在同一位置"的不同物体
    （C-9：单帧常看不到目标，靠多帧提示补齐，故必须区分"同物多帧"与"多物同地"）。

    v6 证据（§5.6 / §7.1 track_consensus）：

    - `track_id`：保留对象的跨帧 track 身份 = **幸存代表的传播身份**（已有则原样保留，
      缺省时按输入下标补 `trk<idx>`，确定性）。被合并的候选只是代表的同一条 track
      的重复观测，不构成新 track —— `count_objects` 靠这一点"不数清单长度"；
    - `duplicate_suspect`：① 该组吸收了 ≥2 条候选（清单长度确定被高估）；
      ② 去重后仍有两对象**同类且世界质心落在同一容差内**（时序不重叠 → 按纪律
      不合并，但"同地同类不同时"仍有重复绑定嫌疑）→ 两条都标嫌疑。
      两条规则的阈值口径与合并判据同源（`rel_tol`），[TODO_CALIBRATE]。

    返回 `(kept_objects, n_dup, groups)`；`groups[k]` 为保留对象 k 吸收的原始下标列表
    （含代表自身，故 `len(groups[k]) >= 2` ⇔ 发生过合并），
    供调用方在去重后的对象集上重算 G9。
    """
    if not objects:
        return [], 0, []
    d = np.asarray(depth_maps, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    med = float(np.median(d)) if d.size else 1.0
    tol = rel_tol * med

    def _frames_of(idx: int) -> set[int]:
        if not masks_per_object or idx >= len(masks_per_object):
            return set()
        m = masks_per_object[idx]
        return set(m.keys()) if hasattr(m, "keys") else set()

    def _hint_of(o) -> str:
        h = str(getattr(o, "category_name", "") or "").strip().lower()
        return "" if h in ("unverified", "unknown") else h

    def _repr_track(o, idx: int) -> str:
        tid = getattr(o, "track_id", None)
        return str(tid) if tid else f"{TRACK_PREFIX_BASE_LIST}{idx}"

    kept: list = []
    groups: list[list[int]] = []
    group_frames: list[set[int]] = []      # 每个保留组的支撑帧并集（时序重叠判据用）
    group_track: list[str] = []            # 每个保留组的 track 身份（幸存代表）
    n_dup = 0
    for idx, o in sorted(enumerate(objects), key=lambda kv: -float(kv[1].det_conf)):
        c = np.asarray(o.centroid_world, dtype=np.float64)
        frames = _frames_of(idx)
        hit = -1
        for k, ko in enumerate(kept):
            kc = np.asarray(ko.centroid_world, dtype=np.float64)
            if np.linalg.norm(c - kc) >= tol:
                continue
            ho, hk = _hint_of(o), _hint_of(ko)
            if ho and hk and ho != hk:
                continue                       # 类别不同 → 不是同一实例
            kframes = group_frames[k]
            if frames and kframes and not (frames & kframes):
                continue                       # 时序不重叠 → 不是同一实例
            hit = k
            break
        if hit >= 0:
            groups[hit].append(idx)
            group_frames[hit] |= frames
            n_dup += 1
            continue
        kept.append(o)
        groups.append([idx])
        group_frames.append(set(frames))
        group_track.append(_repr_track(o, idx))

    # ---- v6：把去重结论写成证据字段（track_id 统一为组内幸存代表 + 重复嫌疑）----
    suspect = {k for k, g in enumerate(groups) if len(g) >= 2}
    # 去重后仍"同类 + 质心过近"的两条（时序不重叠故未合并）→ 都标嫌疑。
    # 类别为空（unverified/unknown）不参与该判据：那类对象的质心是占位 [0,0,0]。
    for a in range(len(kept)):
        ha = _hint_of(kept[a])
        if not ha:
            continue
        ca = np.asarray(kept[a].centroid_world, dtype=np.float64)
        for b in range(a + 1, len(kept)):
            if _hint_of(kept[b]) != ha:
                continue
            cb = np.asarray(kept[b].centroid_world, dtype=np.float64)
            if np.linalg.norm(ca - cb) < tol:
                suspect.add(a)
                suspect.add(b)
    kept = [_with_dedup_verdict(o, suspect=k in suspect, track_id=group_track[k])
            for k, o in enumerate(kept)]
    return kept, n_dup, groups


def _merge_mask_groups(masks_per_object, groups) -> list[dict]:
    """把去重分组还原成"每个保留对象一份掩码字典"（帧 → 该组掩码的并集）。

    用于在**去重后**的对象集上计算 G9，使统计口径与 `SceneState.objects` 一致。
    """
    merged: list[dict] = []
    for g in groups:
        frames: dict[int, np.ndarray] = {}
        for idx in g:
            if idx >= len(masks_per_object):
                continue
            for fi, m in masks_per_object[idx].items():
                m = np.asarray(m, dtype=bool)
                prev = frames.get(fi)
                if prev is None:
                    frames[fi] = m
                elif prev.shape == m.shape:
                    frames[fi] = np.logical_or(prev, m)
                # 形状不一致（同帧两条候选分辨率不同）：保留先到者，不伪造并集
        merged.append(frames)
    return merged


def _bbox_of(pts: np.ndarray) -> list[float]:
    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    return [float(v) for v in list(pmin) + list(pmax)]
