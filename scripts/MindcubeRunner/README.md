# MindcubeRunner

MindCube tinybench 的初步适配，代码全部放在本目录。参考 `CV3Drunner` 的
五次程序请求加一次最终作答、测量后下一轮复核，以及 `Ominirunner` 的
题内工具过滤和认证网关。共享 `src/skill3d` 只作为依赖使用。
无需导入其他 runner 或 `tests` 下的业务代码。

## 输入与工具

- 只选 `source_split=tinybench`，默认使用已核验的 1,050 题、428 张原图，
  配置中的 `selection` 可指定一个公开 sample_id 列表及顺序。
- 保留原始完整题文，包括嵌入的 A./B./C./D. 选项。仅从题文校验当前字母，
  不改写问题或选项。`task_type` 保留绑定文件中的字符串，未把源分类映射成
  统一题型。
- 按每题 2–4 张图的原顺序组合逐图缓存。每张图调用当前 `load_prepared`，
  核验 manifest 公共输入、数组/产物内容哈希、原始 RGB、几何网格及方法身份。
  旧版产物的方法绑定沿用固定哈希的 publication catalog 与准确的产物正文；
  若正文另有 `method_identity`，还要求它一致。缓存必须声明 `labels_used=false`。
  不修改、重发布或重新计算缓存。
- 四个工具：`view_candidates`、`view_depth_summary`、`point_depth`、
  `pair_point_distance`。后两者沿用同图 5×5 局部中位数计算。
  候选必须满足 `category_confirmed=true`，再匹配当前题干或选项中的类别；
  保留仅在选项中出现且复核通过的参照物，不使用缓存聚合题文。
  返回的候选 ID 在题内重新编号，不提供 mask 或原始审核正文。
  测量额外检查原像素四周±3像素的3×3取点网格；局部深度相对跨度超过10%时，
  `data.status=unreliable`、`reason_code=edge_sampling`，并省去深度或距离数值。
  两点距离检查两个端点；内部稳定性探针不单独计为工具调用。有效深度不足的邻域返回
  `insufficient_local_depth`。这项检查不验证目标身份，稳定数值仍须结合原图核对对象。
- 几何单位为 `per_view_relative`。各图独立，不提供跨图配准、身份关联、
  世界方向、米制尺度或 mask 测量。跨图与旋转关系由模型结合原始 RGB 判断。
- Skill 臂从首轮起，每轮按原图顺序发送 `RGB 0、Depth 0、RGB 1、Depth 1…`，
  共 4–8 张图；RGB 保留原始字节。深度 PNG 直接从已核验缓存渲染，
  与对应 RGB 保持相同宽高及像素网格，不缩放、裁剪或增加边框。
  每对图片附 `view_index`、宽高、左上原点、x/y 坐标范围及深度尺度说明。
  各图分别按有效、有限、正深度的 2%–98% 分位数线性着色，
  由近到远为黄、绿、青、蓝、紫，超出显示范围的值截断，无效像素为黑。
  分位区间退化时有效像素为固定青色并说明无深度对比；
  完全无有效深度时附全黑图及明确的不可用说明。
  色彩和原始 Z 不可跨图比较，数值工具仍读取未截断的相对几何。
  输入回执记录实际全部图片的顺序哈希，以及 RGB/深度数量和逐图渲染参数。
  每请求只附一份共享深度图例；逐图标题保留图索引、宽高、坐标范围、有效像素数、
  显示范围和可用性状态。图片说明只描述数据，不携带工具调用建议。
  测量复核时返回全部成功结果、result_id、view_index 和准确采样像素。
  当前版本没有生成采样标注图；表面绑定仍由同一模型判断。

## 作答流程

Skill 臂最多 5 个程序槽和 1 个最终 JSON 槽。程序为一个完整的
`python` 代码块，唯一顶层入口是 `def solve(ctx):`。
使用 `ctx.tools` 调证据工具，使用全局 `ReturnAnswer("A")` /
`YieldObservations([...])` 终结片段。答案必须是当前题的合法大写字母。

静态 Skill 按需要指导调用顺序：先读原题、RGB 与深度预览，确定参照系和
待比较对象；同图深度或距离是决定因素且存在可用测点时，先取得相关比较证据。
定位需要候选框时先查询并 yield，下一轮对照 RGB 绑定对象与内部
表面像素，再测量，避免同程序直接取新候选框中心。`view_depth_summary` 为按需
诊断，不是测量前置步骤；已有摘要、候选和数值从 prompt 复用。
同图深度顺序用 `point_depth` 比较相关表面，表面点距离用
`pair_point_distance`；独立比较可在一个程序中完成，随后一次 yield 所有新
测量 ID。下一轮复核像素、目标与 `data.status`，必要时只修正有问题的点或
端点；视觉证据足够时可直接回答。旋转、跨图关系和方向判断仍依据原图与题目
参照系。此流程是模型指导，不额外增加候选查询的宿主强制门。

每轮先声明当前阶段和任务。程序阶段可以同时要求 `repair`、
`review_measurements` 和 `inspect_hints`；没有待办时使用 `plan_or_measure`。
全局科学约束保存在 `static_skill.txt`，本轮操作及唯一输出格式由
`context_stage.py` 选择。最终轮只要求 JSON，保留旧错误代码但转换其修复提示，
不携带程序格式或新增测量指令。解析、控制接口或普通运行错误不会清除待复核 ID；
有效完成的程序消费已交付输入，再携带本程序产生的新结果进入下一请求。

`TOOL RESULTS` 使用 `mindcube-observation-context/1` 无损表格，分成
`pending` 和 `history`。每组包含工具、外层执行状态、`shared_data`、
`data_keys` 和 `rows`；行结构为 `[sequence_index, result_id, *values]`，
`values` 对应 `data_keys`，再与 `shared_data` 合成原始 `data`。
序号保留全部观测的原始顺序。按字段结构分别分组，因此缺失值和显式 null、
可靠与不可靠结果均不会混淆；不截断条数、不舍入数值、不丢失坐标或稳定性信息。
`restore_observations()` 供宿主离线核验完整恢复；求解程序仍从 prompt 读取证据，
没有新增观测读取工具。回执保存记录数、待复核/历史 ID 和原始/紧凑数据哈希。
压缩后超过 `max_context_chars` 时明确返回 `context_limit`，不静默丢弃待复核证据。
未分页，不增加既有模型请求槽位。执行日志中的原始工具结果完整保留。

程序阶段的工具调用和测量次数没有上限，同一程序及后续程序都可继续调用。
调用统计仅记录 `calls` 和 `measurements`，包含已进入工具的失败调用；
不会因为次数增加自动 yield 或提前进入最终 JSON。
程序执行仍受 `cell_timeout_s` 和正常工具契约约束，下一请求仍受上下文长度限制。
有新测量时，同程序提交的答案记为暂定，下一请求必须带着新结果再复核。
不可靠测量的原因也会交付下一请求，可改选内表面重新测量；
仅有不可靠结果时，题级依据仍记为视觉估计。
没有新测量时允许提前作答。格式、控制接口误用和普通 Python 错误的有限修复
占用原有程序槽。安全违规、工具契约失败、超时终止本题；服务或预算偏离停止
后续调度，不自动重试。沙箱运行在独立子进程主线程。

可加 `arms: ["skill", "direct"]` 作成对比较。Direct 臂使用相同原题和原始
2–4 张 RGB，不附深度预览，
一次 JSON 字母作答。两臂使用相同模型、temperature、seed、thinking 和 token
设置；不进行训练或 Skill 学习。

## 零请求预检

在仓库根运行，配置和输出路径使用绝对路径：

```bash
.venv/bin/python -B scripts/MindcubeRunner/run.py \
  --config /absolute/path/to/mindcube.config.json
```

本目录 `example.config.json` 给出 Titan 的缓存路径。复制到自己的配置文件，
将 `output` 指向仓库和缓存目录外的新目录。输出的父目录须已存在。
预检不读密钥或答案、不联网、不调用模型/GPU，逐图检查所有选定输入，
保存 `preflight.json` 的输入、源码、协议、配置和首轮请求身份。
重复预检保留首次文件；身份变化则拒绝继续。

默认全 tinybench 的 Skill 上界是 6,300 次请求；成对比较是 7,350 次。
小集合可以这样设置（ID 必须来自公开 tinybench 绑定）：

```json
{
  "cache_root": "/absolute/path/to/mindcube-perview-cache",
  "output": "/absolute/path/to/new-output",
  "selection": ["public-sample-id-1", "public-sample-id-2"],
  "arms": ["skill", "direct"],
  "max_requests": 14
}
```

`max_requests` 省略时按选题和臂数计算；显式配置必须等于完整请求上界。
`runner` 可覆盖 `common.Config` 的超时、上下文长度等字段；程序槽固定为五次，
模型固定为 `qwen3.8-flash`、temperature 固定为 0。
`input_pins` 仅用于显式核验过的另一份 manifest/bindings/catalog 快照或离线
合成测试，需同时提供三个文件的 SHA-256，不绕过正文及数组校验。

## 显式运行与评分

真实调用需另外提供 `gateway`、`auth_module_root`、`budget_base` 并加
`--execute`。沿用现有 `transport_auth.load_transport_token` 和认证 evaluation
RPC；密钥不写入配置。支持 HTTPS、本机 loopback HTTP 和当前已授权的集群网关
`http://10.10.0.198:27441`。`budget_base` 是包含 `smallimage/requests` 与已审计
19 份历史账本的目录。
所有历史账本必须存在。执行前核验网关源码/上游身份与跨账本余额，每个响应
还要求服务显式报告正确模型。不得为了适配此入口改变既有预算策略。

历史账本按已核验的两组镜像投影去重，`started` 记录计入已用请求。
每次新调用同时记入本次请求账本和原主账本；历史账本列表应随新增外部
campaign 再核验。默认调度串行；设置 `concurrency: 2` 时，同题两臂同时在
独立子进程执行。聚合检查锁只协调本 runner 的请求准入，不覆盖实际推理时段，
不替代跨 campaign 的全局资源调度。执行必须在完整上界能被当前余额覆盖时
才开始。

同一输出目录不能再次执行或自动续跑。运行输出有 sanitized 请求意图、
执行状态、题级结果和完整分母汇总。每轮 `traces/NN/trace.json` 为完整 JSON：
原始请求与网关 envelope 字节以 base64 保存；完整有序消息内含原始 RGB
和深度 PNG；收到的响应在解析前保存，另记 assistant、实际 reasoning 字段、
finish_reason、model、完整 usage 和字段存在性。成功提取的 Python 全文、
工具位置/关键字参数、完整调用/结果、可靠性、错误和复核 ID 前后状态均保留。
所有事件独立原子写入、fsync，并用连续序号和前一事件哈希连接；进程中断
保留已提交事件和 orphan 状态。存储失败停止后续请求。密钥与隐藏标签不落盘；
预算账本只保存准许身份及数值 token 用量。未经调度、非法、运行失败的题目
都保留在分母。

轨迹的响应边界是 runner 实际收到的网关正文。当前网关若将上游失败转换为
502，仅能保存该错误 envelope；未转发的上游正文/推理标记为未观察到，
不视为 provider 未返回。空字符串、null 与字段缺失分别保存。
只读回放核验消息、图片顺序/字节、wire、程序和事件链，不执行程序：

```bash
.venv/bin/python -B scripts/MindcubeRunner/replay.py /absolute/path/to/output
```

评分是后续独立的主机操作；答案不会传进求解子进程。提供仅覆盖选定题目的
`{"sample_id":"A", ...}` JSON 文件和它的 SHA-256：

```bash
.venv/bin/python -B scripts/MindcubeRunner/score.py \
  --output /absolute/path/to/output \
  --answers /absolute/path/to/host-only-answers.json \
  --answers-sha256 ACTUAL_SHA256
```

结果及配对身份再次核验后保存 `scored_summary.json`，失败和未运行题仍计入
准确率分母。当前入口不自动从官方含标签的 JSONL 提取答案。

## 离线验证

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider scripts/MindcubeRunner/tests
```

测试使用临时原图、真实 NPZ 缓存保存/加载及假模型；不会访问服务或 GPU。
覆盖 RGB/深度配对、像素尺寸与有效性、逐图尺度和确定性、多图顺序与数值、
缓存篡改、题内候选范围、五轮修复和最终 JSON、
测量后复核、多次调用及跨轮测量、无损表格恢复、候选复核、修复与测量复核并存、
最终轮格式隔离、上下文超限、安全失败、服务不重试、预检冻结和评分分母。
