# pySpatial 如何评测 Omni3D-Bench

核查日期：2026-11-01。范围：论文协议与官方公开源码静态阅读；未安装、导入或运行 pySpatial，未调用模型、GPU 或评分器。

## 来源与版本

- [项目主页](https://pyspatial.github.io/)。
- [论文 arXiv:2603.00905v1](https://arxiv.org/html/2603.00905)：2026-03-01，重点阅读 §3、§4.1、§4.2 与附录 B/C。
- [官方源码](https://github.com/Zhanpeng1202/pySpatial/tree/c6b08f3beef4fb7b33699bd408806f6ace02ae45)：本次固定提交 `c6b08f3beef4fb7b33699bd408806f6ace02ae45`，提交日期为 2026-06-02。
- Omini 当前数据仍固定为 `dmarsili/Omni3D-Bench@fdba2ef00255e74c353da59475eafec47ec974a5`，不依据 pySpatial 的浮动分支重新下载或替换。

## 论文中明确说明的 Omni3D 协议

1. Omni3D-Bench 被用作单视角空间推理评测，测试多视角方法向单图的迁移。
2. §4.1 声明在 MindCube 与 Omni3D-Bench 上使用 VGGT；默认代码与答案 agent 使用 GPT-4o。全文 MindCube 的 GPT-4.1-mini 预算选择不能自动套用到 Omni3D。
3. §4.2 的 Table 3 沿用 VADAR：`numeric (other)` 使用 MRA；`numeric (ct)`、yes/no 和 multi-choice 使用 Accuracy。
4. 方法流程是生成 Python 程序，执行空间工具得到文本／图像证据，再由模型根据程序和证据回答。
5. 方法不进行梯度微调；附录 C 提供了程序生成的示例提示。这里的 zero-shot 不意味着提示中没有示例。

论文不足以确定实际 Omni3D 的逐题加载器、答案类型适配、精确阈值实现、输入图像预处理和逐题请求预算。这些需要专用源码或更完整的执行材料，不能从 MindCube 脚本推定。

## 当前公开实现的主流程

源码索引：

| 文件 | 已核对位置 | 行为 |
|---|---|---|
| [README.md](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/README.md#L35-L64) | L35–64 | 仅介绍 MindCube 数据与预处理／评测命令 |
| [mindcube.py](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/mindcube.py#L240-L336) | L240–336 | 构造 Scene → 生成程序 → 提取代码 → 执行 → 回读证据作答 → 宿主比较 |
| [query.py](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/agent/codeAgent/query.py#L10-L55) | L10–55 | 代码 agent 使用 `gpt-5`，`reasoning_effort="low"`，输出上限 4000；此请求只发送文本提示及题文 |
| [execute.py](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/agent/codeAgent/execute.py#L10-L58) | L10–58 | 提取首个 Python 代码块，通过普通 `exec` 定义 `program` |
| [pySpatial_Interface.py](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/pySpatial_Interface.py#L258-L300) | L258–300 | 调用 `program(scene)`，保存返回的 visual clue，再交给答案 agent |
| [anwer.py](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/agent/anwer.py#L36-L124) | L36–124 | 答案 agent 使用 `gpt-5`，输出上限 2000；答案限制为 A/B/C/D；发送题文、程序及 visual clue |

注意官方文件名确为 `anwer.py`。

`mindcube.py` 的输入包含 `id`、`question`、`images` 和宿主侧 `gt_answer`。构造给 agent 的 Scene 时只传图片、题文和 scene_id。公开入口统计 among／around／rotation，并按 A–D 字母归一化匹配答案；这属于 MindCube 的合同。

正常成功路径包含生成代码与最终作答两个显式模型调用。该入口还包含 API backoff 重试和 basic QA fallback，所以不能把实际调用量固定写成两次。各进程限速也不等于本项目的全局请求账本。

## 单图与缓存接口

- [Scene._load_images](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/pySpatial_Interface.py#L26-L52) 支持单个图像文件；传入目录则枚举目录内的图片。
- [pySpatial.reconstruct](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/pySpatial_Interface.py#L118-L147) 优先装载显式目录或 `PROCESSED_BASE_DIR/scene_id` 的缓存；未命中时进入 `reconstruct_3d`。
- [ReconstructionTool.reconstruct_3d](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/tool/recontruct.py#L161-L202) 的预计算模式缺失时抛错；关闭预计算后，常规 VGGT 路径明确要求至少两张图片。
- 同文件的 [reconstruct_single_view](https://github.com/Zhanpeng1202/pySpatial/blob/c6b08f3beef4fb7b33699bd408806f6ace02ae45/tool/recontruct.py#L233-L256) 定义了单图函数，但所检查的常规 Agent/API 路径没有自动选择该函数。
- `tool/estimate_depth.py` 定义 DepthAnythingV2；当前 `pySpatial_Interface.py` 对它的导入被注释。其文档将输出称为米制，但这不足以证明尺度经过标定，不能作为 Harness3D 开启米制权限的依据。

因此，“Scene 能接收一个图片文件”不能单独证明当前公开评测入口能完成 Omni3D 单图评测。

## 已确认的公开实现缺口

本次检查固定提交的完整文件树，未发现 Omni3D 专用 loader、runner 或评分入口；README 也没有 Omni3D 执行命令。

当前公开主路径与论文 Omni3D 协议至少存在这些差异：

- 模型：公开代码为 GPT-5，论文默认 GPT-4o。
- 答案：公开 `SpatialAnswer` 为 `Literal["A","B","C","D"]`，不能承载 Omni 的整数、浮点及自由文本。
- 评分：MindCube 的字母匹配不是 Omni 的严格浮点 MRA、整数匹配和小写文本匹配。
- 重建：缓存未命中且关闭预计算时，常规 VGGT 路径要求多图；单图函数尚未接到该入口。
- 原图交付：代码生成请求为纯文本；带 visual clue 的答案请求未同时附加原始图片。basic QA fallback 才显式读取原图。Harness3D 的逐题原图合同需要独立实现。

这些说明所检查提交的公开复现材料不完整，不能据此判断作者论文中的实际 Omni3D 实验过程或有效性。

## Ominirunner 的适配方向

采用其“程序 → 工具证据 → 答案”的流程组织，使用 Harness3D 已发布的 MoGe2 单图相对几何缓存和受限沙箱。

本目录后续独立实现：

1. 官方原图与完整题文输入；每题只绑定一图，禁止目录枚举扩成同场景多图。
2. 按原图哈希验证缓存，按当前题文过滤类别；不以同场景缓存代替逐题图像身份。
3. 整数／浮点／文本终态协议，不建立 A–D 包装。
4. 隔离宿主评分；浮点阈值严格使用 VADAR 的 `<`，文本只做官方小写比较。
5. 在每次求解上下文中保留当前原图及允许的观察，不把渲染图当成额外独立观测。
6. 固定请求上限和全局账本，不复制公开入口的自动请求重试、额外 fallback 调用或普通 `exec`。

没有复制或运行 pySpatial 代码，没有建立模型客户端，没有启动预计算，也没有产生 Omini 实验分数。
