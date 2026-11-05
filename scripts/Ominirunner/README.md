# Ominirunner

Harness3D 的 Omni3D-Bench 独立评测目录。目录名按用户指定使用 `Ominirunner`；Omini、ominibench 在此均指 Omni3D-Bench。

当前协议：最多五个程序轮＋一个 JSON 终结轮，工具调用与数值测量不设次数上限，详见 [实验设计](experiment_design.md)。此前两程序轮的 12 题实验和已停止的 20 题先导产物保持冻结；新版实现使用独立冻结身份。

当前 `/12` 版本包含公开答案标签、错点替换、工具比较、图像坐标变换、任务量与单位约束、RGB 实体范围、可观察进展和被拒提交回传。实现、依据和验证见 [实现说明](implementation_notes.md)。程序轮文本不设硬上限并保留全部测量事实；超过 10000 字符时记录长度和超出量，不裁剪或中止。终结轮以 20000 字符、最多八条测量装配；直接组仍为 10000 字符。局部图最多两页，调用和测量次数不设上限。完整历史保存在宿主，遗漏数量、可见 ID、数值与历史哈希进入交付回执。

公开长度尺度支持 meters/m、feet/foot/ft、inches/inch/in、yards/yard/yd 和 miles/mile/mi。题面值保留原单位，任务契约识别输出单位；跨单位的公开尺度另给出 `answer_unit_value`。计算应将尺度声明为输出单位，宿主按物理长度核对它与题面尺度的等值关系，并保持原始算术精度与单位校验。

`point_depth` 的 `depth_samples` 返回 5×5 区域内每个像素的相机 Z，不计算均值或中位数。`origin_xy` 是矩阵左上角的原图坐标，`values` 按行排列，无效点为 `null`；图像边界处按实际范围裁剪。`pixel_xy` 是取整并限制到图内的实际中心坐标，`depth` 是该中心的原始 Z。计算绑定可以加 `sample_xy=[x,y]` 明确选取矩阵中某个有效值，省略时使用中心。边缘附近可能测到其他表面而出现较大差异，模型自行辨认目标值。`pair_point_distance` 使用实际中心像素的原始三维点计算距离，并返回 `a_depth_samples`／`b_depth_samples`；中心无效时距离为 `null`，更换端点须重新测量。

采样可使用默认原像素，或 `image_id="rgb-original", coordinate_space="normalized"`；归一化坐标取值为 [0,1]，宿主映射到 [0,width−1]／[0,height−1]。局部图使用交付的 `crop-M...` ID 和局部像素／归一化坐标。图像回执保存源区域、局部尺寸、在拼图中的偏移及拼图哈希；工具保存请求坐标、变换和最终原像素。未知 ID、越出局部图内容或非法归一化坐标会拒绝。

采样用 `targets=[{"target":"chair","surface":"top"}, ...]` 声明计划测量的表面。点位复核检查 RGB 标点、局部图和原始样本，另用 `observed_surfaces` 记录实际目标、表面、判断与可见性；pair 按 a/b 顺序分别记录。具体数值任务的计算和比较必须有这两组记录。测错目标则 `resample`，看不清则 `uncertain`；`wrong`／`wrong target` 兼容映射为 resample 并保留原值，其他非法枚举仍拒绝。边缘不会自动被拒绝。

```python
def solve(ctx):
    measurement_reviews = [{"result_id": "FULL_ID", "decision": "resample",
        "observed_surfaces": [{"target": "wall", "surface": "center",
                              "decision": "resample", "visibility": "visible"}]}]
    YieldObservations([])
```

复核只引用当前可见的成功测量。一条格式错误不会丢弃其他合法记录；反馈指出条目与字段。thinking 中发现的错点需要正式写入 `measurement_reviews`。每轮及最终结果保存完整状态，运行或格式错误后继续保留。替换工具调用加 `replaces="OLD_ID"`；成功返回后旧记录持续排除，保留原值与替换 ID，新点下一请求单独复核。被替换旧点退出待办，不再消耗重复取证轮次。其他旧记录可改判；计算只用当前 usable、尚未被替换的记录。已声明依赖优先保留，再交付新测量／Yield 指定点、可用测量和旧待办。受预算限制未展示的依赖会明确标出。

有测量之后，`ReturnAnswer(value, evidence=...)` 或终结 JSON 的 `evidence` 必须声明 `visual_estimate` 与简短依据，或 `calculation` 与表达式／测量绑定。计算只允许当前可见且已判为 usable 的测量，宿主核对有限算术和答案数值。完整高度／宽度／长度的 `geometry` 移入对应计算绑定，例如 `{"result_id":"FULL_ID","quantity":"height","geometry":{"coverage":"complete","axis":"object_vertical","anchors":["top","base"],"objects":["chair","chair"]}}`。要求完整范围、对应轴向和极值端点，按 a/b 顺序声明同一对象，名称各限 80 字符。对象绑定、完整范围与尺度常数的真实性仍由同一个千问模型判断；宿主检查声明一致性，不认证真实几何。未使用的待复核测量保留在档案，不阻止已有依据的答案。

程序轮的计算示例要求先保存可见工具回执中的完整原始值，再在 Python 中求值，将 `answer` 变量传给 `ReturnAnswer`；不手算或提前舍入为数值字面量。宿主继续用 `rel_tol=1e-9, abs_tol=1e-9` 检查表达式与答案的一致性，不自动修正提交值。终结 JSON 保留足够精度。`chord` 必须省略 `geometry`，只有完整 `height/width/length` 才要求相应几何；point 绑定不能附 geometry，pair 绑定不能选择 sample_xy。

工具支持的文本答案使用 `tool_comparison`，绑定两侧表达式、`< <= > >= == != three_way` 和公开选项映射。宿主核对可见性、usable 状态、任务量、共同参照和两侧单位，再重算关系及答案标签。数值 calculation 同样检查表达式的单位；米制尺寸需要题内公开跨度和匹配的完整尺度锚，使用如 `s*h/w` 的公式。camera Z 不支持步行距离、碰撞或物体间距。完整极值须与两端实际复核一致且完全可见。分层回执将算术、任务单位合同和同模型表面声明分别记录，`independent_visual_geometry=false`。

公开题内选项和结构化 options 生成 `O1/O2/...`。程序组选择 ID 或精确标签，由宿主输出原选项文本并保留 raw_answer；非法冠词或标签不会模糊匹配。没有公开选项时提示直接提交所问数值或文本。官方评分及原 RGB 直答输出保持原规则。计数 visual_estimate 即使没有工具也必须声明 `object_scope={include,exclude,entities}`，将候选 ID 对应到 RGB 实体；孔等所问子特征也逐项列为实体。仅凭框重叠或类别确认不能合并实体，新 RGB 证据允许修改清单与计数。体积估计必须使用 `assumptions={"object_scope":"对象范围","volume_model":"形状模型"}`，恰含两个非空、各不超过 200 字符的字符串；这里的 object_scope 不是计数实体字典。

各程序轮和终结轮交付完整的当前题型模板，包括计数实体、体积字符串假设或普通 RGB 依据。basis 建议不超过 120 字符，硬上限仍为 200；程序例用 `len` 自检。尺寸计算示例按所问维度和题内公开尺度生成，使用已注册的 `s*t/a`；尺度锚可以是不同维度，必须声明各自真实测量量、完整端点与回答单位。例中的答案、测量值、对象名和 ID 需换为实际证据。

候选保留类别审核与通过状态、审核回执摘要及实际身份；缺少 NMS 记录时明确显示 `not_recorded_in_cache`。重叠候选的关系保持待定，不重新运行 NMS 或修改旧缓存。类别审核不能认证实例同一性，检测不到也不证明对象不存在。

下一轮显示未解决点位和固定长度进展计数；完整新增 ID、内容新增 ID、复核变化及替换关系留在宿主。重复调用得到相同数据不算新增证据，模型便笺标为未验证假设。终结阶段用已有 usable 依据或 RGB 作答。`evidence_activity`、审计、报告和轨迹分别汇总输入／输出 token、替换测量、内容新增且 usable 的替换、复核变化和无进展轮次。

`evidence.py` 的反馈按错误返回具体 `field`、缺失／多余字段、几何轴与端点要求或样本选择要求。算术不匹配同时返回 `submitted`、`calculated`、`abs_error`、`allowed_error`、`rel_tol` 和 `abs_tol`，这些值只由已交付的工具值与声明表达式得到。执行记录保存完整诊断，下一轮交付整条反馈；审计重建这些诊断并拒绝字段或数值篡改。多绑定错误按绑定名排序选择，保证保存后的 JSON 可重放。此前 `/8` 实验与源码快照保持原样。

`submission.py` 回传最近一次成功执行但被答案合同拒绝的声明，包含原答案、evidence 和当时的拒绝反馈；它明确标为未验证模型假设，其他字段未被认证。失败 cell 的答案不会进入此状态；状态可跨 yield、格式错误和可恢复输入错误保留，提交通过后清空。宿主快照最多 2400 个转义 JSON 字符，超长声明整字段省略并记录哈希；最终先装配工具事实和实际裁图说明，再用剩余空间交付该快照，避免挤掉测量依赖。回执记录源声明、状态及实际交付哈希，worker 与 audit 用同一函数重建。依赖优先顺序统一按绑定名排序，避免 JSON 保存时排序导致重放差异，包括动态构造的声明。

## 修改范围

- Omini 专用的入口、配置、输入适配、提示、工具适配、答案协议、宿主评分和离线检查全部放在本目录。
- 不修改其他数据集的 runner、`src/skill3d` 公共实现、公共配置、依赖文件或 `source_frozen`。
- 公共组件只能通过本目录的包装器复用；若公共接口不适配 Omini，在本目录实现独立适配，不添加全局 monkey patch 或跨任务状态。
- 导入模块不能启动模型、创建在线客户端、读取凭据、发送请求或初始化 GPU。
- 20 题先导测试已停止；修改代码不自动重启测试、扩展全量或修订 Skill。

## 固定输入与缓存

- 数据：`dmarsili/Omni3D-Bench@fdba2ef00255e74c353da59475eafec47ec974a5`。
- 501 题、201 张唯一图片，每题严格使用官方绑定的一张原图和完整题文。
- 首轮输入为原始 RGB＋该图缓存生成的相对深度图。后续当前可见的测量位置同时标在 RGB 和深度显示副本上，待复核点的 RGB 局部图按每页八个唯一位置分组；这些图均来自同一原图。标点编号沿用宿主历史编号，证据精简不改变坐标或数值。
- 保留题内选项、数值假设、单位和方向；不转换为 VSI 或 MindCube 的字母选择题。
- 原图 SHA256、逐题输入身份和图像缓存身份分别验证。同场景的其他图片不能加入输入。
- 现有 MoGe2、检测、SAM2 和 Intern 类别审核缓存只读复用；保留实际的两种审核版本。
- 米制、世界坐标、跨视角身份和对象掩码测量权限维持关闭。
- 合并题文形成的类别缓存，需要先按当前题文构建允许范围，再交付题内候选视图。
- 参考答案仅由隔离的宿主评分读取，不能进入在线求解、工具观察或 Skill 修订上下文。

## 入口与模块

`run.py preflight` 固定源码、20 个公开题目绑定、缓存、首轮消息与服务身份，不调用模型。`run.py run` 同时启动每题两路子进程；两组均 `thinking=true`，直接作答最多 1 个请求，静态方案最多 6 个请求。20 对题的本次上限由配置计算为 `20 × (1 + 6) = 140`，同时计入既有总账本；新协议使用新冻结身份。保留原 12 题，另以公开题号哈希排序加入 4 float、2 int、2 str。首次请求即正式测试，不重复通信探针或自动恢复。

`rerun.py` 用于 20261103-r03 的同题复测：16 个 episode 并发，程序组 20 题重跑，直答 15 题复用、5 题补跑，新增请求上限 125。预检核对历史直答的原图、完整提示词、采样参数、模型、冻结回执及本地／共享收费记录。复用的 `result.json` 与原始消息按字节复制，保留原 run identity；来源存入 `reuse.json`。评分表计入完整两组的历史与新增用量，审计单独报告本轮收费。共享限流切换必须持有状态锁及全部槽位锁；全部在线任务结束后恢复 4，并保留每条历史请求及切换回执。

当前入口支持 `--expected-reused-direct` 指定预检必须核验的历史直答数量；同一 20 题全部复用时，程序组新增请求上限为 120。跨轮复用按原来源核对，不将复制件当作新的收费记录。最新复测保留逐轮 `*.response.json` 的 output／thinking、最终 result.json 和汇总 trajectories.json；请求回执、episode 汇总和轨迹汇总均记录超一万字符情况。

复测并发由 `--concurrency` 指定，写入冻结合同；线程池、local/shared 请求账本和网关健康回执必须一致。支持 1—40 路，40 对应本题集的全部 episode 数量；不再固定为 16。直答全部复用时只有 20 个新 episode，使用 20 路可同时调度全部程序题。共享策略只在持有状态锁及全部相关槽位锁时切换，收尾恢复原先 4 路并保留历史记录。

`traces.py` 在审计完成后导出 `trajectories.json`，包含 20 题两组的完整输出、thinking、程序、工具回执、复核、提示词文本和图像哈希。原始图像仍保存在逐轮消息文件中。`python -B scripts/Ominirunner/rerun_check.py` 离线验证实际 16／20 路准入、活跃槽位拒绝切换、历史未知记录保留与复用费用扣除。

程序响应必须为一个完整的 Python 代码块，唯一顶层入口为 `def solve(ctx):`。四个证据工具在 `ctx.tools` 下，`ReturnAnswer` 与 `YieldObservations` 是全局控制函数。程序轮只展示 Python 输出方式，终结轮只展示 JSON 输出方式；完整复核和答案证据协议仍从首轮开始交付。短 Skill 的查工具／补采样步骤以工具开放为前提。每轮用简短阶段行展示剩余程序轮和累计调用次数。程序中的工具调用按顺序执行，不设每轮或整题次数上限，次数仅用于记录。调用次数不触发程序结束或提前切换 JSON；第五个程序轮之后进入终结轮，也可在程序轮完成测量复核后提前提交答案。执行超时、参数／权限检查和现有文本上限继续生效。

| 模块 | 职责 |
|---|---|
| `run.py` | 独立命令入口；零请求预检与正式执行显式分开 |
| `common.py` | 冻结模型、采样、上下文上限及身份配置 |
| `data.py` | 无参考答案的逐题输入和图像缓存双层验证 |
| `budget.py` / `transport.py` | 聚合历史独立账本、既有主账本收费和单次代理请求 |
| `protocol.py` | 完整题文、观察和整数／浮点／文本终态合同 |
| `tools.py` / `sampling.py` | 单图工具包装、原始逐点深度与三维点距、权限检查及证据身份 |
| `native.py` / `worker.py` | 原生标量、隔离进程程序求解和下一轮测量复核 |
| `review.py` / `evidence.py` | 简短复核及状态、计算绑定的用途声明、受限算术和答案核验 |
| `answer_contract.py` / `coordinates.py` | 公开标签与任务规格、宿主图像坐标变换 |
| `instances.py` / `progress.py` | 候选来历与 RGB 实体范围、可观察进展和替换统计 |
| `submission.py` | 有界被拒声明快照、整字段遗漏身份和上下文交付 |
| `task_evidence.py` / `provenance.py` | 有序任务操作数、符号表达式检查与答案来源/放弃依赖记录 |
| `trace_replay.py` | 读取保存的公开提交和工具回执，零模型/零工具执行的契约回放 |
| `score.py` | 完成后隔离宿主读取标签，保留官方严格评分 |
| `check.py` | 离线合成样例：六轮阶段、格式修正、无次数上限、真实工具数值、证据交付及审计重建 |
| `test_contracts.py` | 七项专项回归；公开标签、错点、比较、坐标、单位、范围和实际 worker 重放 |
| `static_skill.txt` | 各题、各轮使用同一份短 Skill |

执行产物使用启动合同指定的独立目录，不放进其他数据集的任务目录。

离线检查命令：`.venv/bin/python -B scripts/Ominirunner/check.py` 和 `.venv/bin/python -B -m pytest scripts/Ominirunner/test_*.py -q`。使用合成输入和模拟客户端运行真实工具／worker／审计路径；专项测试禁止在线 transport。当前 411 项 pytest 回归和原有 64 项离线 episode 检查通过。可追加固定 VADAR `engine.py` 路径检查官方评分函数一致性。

协议 `/11` 已增加局部输入恢复：工具签名及有明确本地输入归因的坐标/targets/replaces 错误可进入下一请求改参，失败结果和同 cell 答案仍不可信。质量门禁、安全、超时及未知契约故障继续终止；新冻结源码必须包含 `recovery.py`。最终轮保留连续 usable 的完整表面复核，计算依赖与错误指向端点优先裁图。详见 `implementation_notes.md` 的 repaired15 修复说明。

协议 `/12` 追加题型模板、精确字段反馈与被拒提交回传；新冻结源码还必须包含 `submission.py`。当前修改不替换旧运行的冻结代码，后续复测使用新的身份。

协议 `/13` 修复 fresh20 轨迹暴露的选项域、目标/尺度分离、混合维度比值、计数比值和固定起点距离比较。实际表面 `observed_surface` 与极值角色 `extent_role/extent_basis` 分开；裁切、遮挡、不完整范围和已声明疑点阻止 usable。工具派生答案、独立 RGB 回退与放弃的依赖分别记录，显式直线程序依赖不能伪称 RGB。动态程序和视觉真实性仍没有独立认证。新冻结源码须额外包含 `task_evidence.py` 和 `provenance.py`。

格式错误另外保留完整 pending_program 模型状态，在下一程序请求交付原未执行计划；它不进入 TOOL FACTS，也不会自动运行。实际重发仍经过严格围栏和原沙箱。最终轮仅保留未执行身份，反馈切换为最终 JSON，worker/audit 共同重建状态与交付哈希。

原提交回放：`.venv/bin/python -B scripts/Ominirunner/trace_replay.py <原运行目录> <独立输出目录>`。仅读取公开题文、模型提交与已保存的工具回执，不执行原程序，不调用模型或评分器；原运行的输入/结果文件在回放前后校验哈希。该结果用于检查新契约的诊断，不产生新的准确率。

## pySpatial 核查

详见 [pySpatial 源码与论文核查](pyspatial_reference.md)。所检查的公开提交没有 Omni3D 专用评测入口，不能把其 MindCube 运行命令当作 Omni3D 的复现命令。

## 独立二值复核试验（默认 InternDecision）

`jev_trial.py` 提供独立单题入口，默认复用已有 InternDecision 服务；原有 `run.py` / `rerun.py` 的默认模式不变。主模型用 `target`、`quantity` 和点坐标提出测量，独立复核器对一条测量作一个 `yes` / `no` 决策（pair 的两点一起检查）。宿主只在下一次请求交付通过的 ID、对象、原图点位、quantity、数值和单位；拒绝项只交付提案身份和重选提示。主模型不填写表面复核或完整几何字段，也不接收复核标点图、裁图、复核器历史、概率和拒绝数值。原始 RGB 与对齐深度图保留。

```python
def solve(ctx):
    r = ctx.tools.pair_point_distance(
        view_index=0, a_xy=[100, 120], b_xy=[100, 300],
        target="chair", quantity="height")
    YieldObservations([r["result_id"]])
```

上例坐标仅说明调用格式，实际运行由主模型选择可见物体的完整端点。高度、宽度、长度、直径分别固定 A/B 顺序 top/base、left/right、front/back、rim_a/rim_b；两物体间的 chord 使用 `target=["chair","desk"]`。Point 只允许 `camera_z`。答案的 binding 只填获批 ID 与 quantity；宿主补全维度 binding，并继续检查下一轮交付、精确算术、题目物体/操作数、单位和公开尺度锚点。复核器的 yes 是 RGB 命中/轴线/完整端点判断，不是预测深度精度或绝对尺度证明。

使用独立新目录预检、运行、审计；参数路径指向现有验证过的缓存、鉴权模块和共享账本：

```bash
.venv/bin/python -B scripts/Ominirunner/jev_trial.py preflight \
  --root /path/to/new-jev-trial --sample-id 15 \
  --cache-root /path/to/omni-cache \
  --auth-module-root /path/to/auth-module \
  --shared-budget-root /path/to/smallimage/requests \
  --gateway http://10.10.0.198:27443 --provider intern
.venv/bin/python -B scripts/Ominirunner/jev_trial.py run --root /path/to/new-jev-trial
.venv/bin/python -B scripts/Ominirunner/jev_trial.py audit --root /path/to/new-jev-trial
```

默认主模型至多 6 请求、独立复核至多 8 请求，共计至多 14；复核单独记为 `gate`，同时计入本地与共享预算。预检无推理请求，冻结输入、源码、协议和参数。服务错误、模型身份偏离或预算耗尽立即停机，没有重试、自动续跑或模型替换。审计离线重建主模型上下文与宿主决策，再核对本地/共享收费；不读取参考答案，不生成 benchmark 分数。

Intern 使用 `intern_review.py` 复用原生 `InternDecisionClient`。2026-11-04 实测服务在 4090 节点的 loopback `27248`，模型 `Intern-Decision-4B`，服务 identity 为 `c16e97ae7e43e8605e201de8388eb9a318bc5f260e05a59e869943a8762c0cb2`。宿主发送带空心 A/B 标点的全景图、局部原图、同一局部的 A/B 标点图，只询问一个原生 choice（yes/no）。标点从原图像素中心分别映射到全景和裁图，记录坐标转换和三图哈希。响应必须匹配请求和服务身份，并确认三张图实际编码、合法二选一 token 和决策位置。

主模型的宿主复核流程实际加载 `static_skill_host.txt`，在上下文回执中记录 Skill 哈希。步骤为确认任务操作数、选点、提交 target/quantity/coordinates、等待下一轮获批 TOOL FACTS、按原值计算。主模型不填写自审记录或几何声明。Intern 复核按 camera_z、chord 和完整尺寸分别生成问题；表面点允许命中该物体的可见部件，完整尺寸额外要求同物体的完整端点和轴线。二值提示版本为 `omni-intern-measurement-binary/2`，总览标点渲染版本单独冻结。原有自审模式仍读取 `static_skill.txt`。

主模型上下文版本为 `omni-host-binary-context/3`。`RESELECT` 中的 ID 只允许通过 `replaces` 修复，不授予计算权限；新结果需独立复核并在后续请求交付。`SUPERSEDED IDs` 明确旧 ID 与新 ID 的关系，旧值保持排除。所有获批结果保留原数值精度及对象、点位身份，比较 binding 明确公共 reference，尺度使用公共 `answer_unit_value`。阶段行展示剩余程序轮和已用宿主复核预算。

新请求保留可操作的错误修复说明，并以 `REPAIR EVIDENCE` 交付有限的、未经认证的模型声明：计算表达式、binding、公开尺度、计数实体或体积假设；不交付被拒答案、被拒测量值或模型几何声明。最终 JSON 轮仍可从获批原值重算，不复制被拒答案。未执行响应只提取字面量工具提案，不作为代码执行；工具关闭时只保留未执行身份。候选提示沿用计数验证的前八个 ID，并明确省略数量和非穷尽性。模型状态不能挤掉测量事实；上下文回执记录原状态及实际交付的哈希，离线 replay 检查所有新字段。

宿主默认要求校准后的 `p(yes) > 0.5`；精确等于阈值按 no。`jev_trial.py preflight` 和 `jev_probe.py` 可用 `--yes-threshold 0.7` 显式设置 Intern 阈值，范围 `[0.5,1]`。该参数进入冻结 policy 和回执身份；离线审计按同一阈值重算放行决定。模型原始 choice 与 probabilities 保留在宿主回执中，主模型只接收获批数值。未显式设置的历史 policy 保持原有序列化和 0.5 行为。原有类别复核的 P1/.77 不作为测量点阈值依据。

2026-11-04 使用旧复核提示做过独立阈值试验：8 张原图、48 个提案，其中 24 个表面点和 24 个完整／不完整尺寸提案。每例一次 Intern 请求，按 RGB 哈希分为选择与验证两组，视觉预期判定在推理前固定。按初始视觉预期，0.5 放行 16 个正例中的 3 个，并放行 32 个反例中的 2 个；0.6/0.7/0.77/0.8/0.9 全部拒绝。因此依用户决定保留 0.5。后续检查发现尺寸边界正例的标注有不确定性，该统计保留为原始记录，不能当作最终提示或严格几何准入的准确率。

新提示另做过 12 个开发案例的对照：初始标注下正确保留 2/6→3/6，错误放行 2/6→1/6。复核所有 6 个正例后，水壶壶身、吧凳坐面、电视屏幕为明确表面正例；水壶／垃圾桶高度是边界案例；镜子完整宽度的严格端点和透视轴线未被标注充分确定，不能作为确定正例。保留原始标注与概率，单独记录标注复核，不用事后删除样本提高分数。

输入表达对照保持目标、原图点位、二值问题和 0.5 不变。电视普通表面点在总览标点后 p(yes) 为 0.546915，原来为 0.200972；同图错点仍被拒绝。扩大上下文裁图可以放行吧凳坐面，但也将另一把椅子上的错点放行，故默认只采用已经实测的总览标点，不全局扩大裁图。该对照是开发诊断，尚未验证完整 benchmark 的泛化准确率。

`intern_threshold.py` 提供独立的 `freeze`、`run --split selection`、`analyze --split selection`、`run --split validation` 和 `analyze --split all` 入口。`cases.json` 保存目标、quantity、原图像素坐标、RGB 哈希和仅供宿主统计的视觉预期；`freeze` 固定提示、服务身份、校准配置和实际三图输入。阈值通过一次推理的 probabilities 离线扫描，不重复调用模型。每组使用独立的本地 gate 账本和独占启动记录；遇到服务失败即停，不自动重试或续跑。报告应同时给出正确保留、错误放行和误拒，全部拒绝时 precision 记为未定义。

Mac 可通过临时 SSH 隧道访问该 loopback 服务；`--review-endpoint`、`--review-identity` 可显式冻结另一已核验副本，不会自动扫描或替换服务。Intern 不需要 API key，也不启动新模型或改变服务配置。

可选 `--provider makers` 使用原生 `systemone` 的 image assets + `choice`，固定两选项 yes/no，不使用 chat 或概率阈值。凭据从 `MAKERS_MODELS_KEY` 或权限仅限用户的 `~/.config/harness3d/jev.env` 读取。显式 `--provider typesafe` 使用其自身的 `TYPESAFE_API_KEY` / `typesafe.env`；两通道绝不混用凭据。Jev 视觉模型身份固定为 `jev-1.14.0`，旧版文本模型不会被当成视觉复核成功。

可先做一个独立的图片协议探测，无主模型请求、无量化答题；其单独账本上限为 1，不改正式实验账本：

```bash
.venv/bin/python -B scripts/Ominirunner/jev_probe.py \
  --image /path/to/original.jpg --target "electric kettle" --xy 1180 610 \
  --root /path/to/new-image-probe
```

2026-11-04 的真实 Intern 图片测试中，水壶壶身点 `(1205,655)` 返回 yes，背景点 `(300,100)` 返回 no，水壶手柄点 `(1180,610)` 返回 no；三次均为 HTTP 200、三图实际编码。该结果验证了图片接口与点复核的放行/拒绝分支，未做整题在线评分或完整尺寸正确率评估。可选 Makers 通道的一次图片请求返回 HTTP 400，未获得 yes/no。
