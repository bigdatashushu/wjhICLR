# 当前证据、方法和模型接线

实际实现与方法库均在本目录；共享 `src/skill3d` 保持原样。P3离线接线验收47 passed（44.66s），其中18项新增感知/方法/solver测试，另29项宿主/预算回归。CLI `--check`通过且0真实请求。

## 当前RGB准备

`CurrentEvidencePreparer(PerceptionBackends(...))`直接将快照PNG解码进共享 `SmallImagePrecomputer` 的内存decode适配。原图500×500，不写临时图片、不裁切或resize，不读GT、历史图像或方法正文。公开类别由固定词表从指令优先、动作名其次解析；最多4类，未知词不调用额外模型补猜。解析器身份、冻结配置、实际backend身份hash均进入manifest。

`CombinedDINO`使用一次组合prompt、`max_attempts=1`、60s总时限，替换共享DINO adapter的按类别循环。选择按类别优先级、confidence、box顺序确定的1个候选，因此每当前view最多一次DINO、一次SAM、一次Intern类别核验和一次MoGe。这属于固定首版感知配置，B01/B11相同，不能按method改变。SAM/MoGe可复用既有adapter；Intern复用既有 `InternCategoryAdapter`，其client必须已经完成独立health preflight，不能在episode中隐式插入health请求。backend callable合同为单次调用且无自动retry。

每个实际请求在内部调用前从同 `EpisodeBudget` 申请对应stage额度。共享准备器原来的吞异常继续机制被服务终止信号绕过；服务异常立即停止，无后续请求。错误只以固定code进入宿主报告。Intern原生不可用receipt也视为服务故障；已有类别review身份与图像证据仍经共享严格验证。

返回准备数组具有不可改写的bytes backing；每次使用复核共享artifact hash和发往sandbox的白名单wire hash。manifest绑定episode、decision、observation/mutation修订、当前view、RGB/像素/动作表、最终evidence修订和capabilities。manifest不含final snapshot hash，因为后者本身包含manifest hash；升级证据后由preparer.validate复核全部scope与wire。每次mutation重新准备，RGB相同也不复用上一decision权限；内存cache只留一个当前decision。

局部depth/points均为当前camera相对单位；类别确认不能授权candidate表面几何。候选的mask质量/几何可用标志保持false；`candidate_depth_summary`继续拒绝。

## 完整Skill与请求

`methods.py`从 `methods/manifest.json` 加载三个独立完整正文，不读取QA槽位或active pointer。hash按共享规范化UTF-8正文计算。每条约3.7k–4.0k字符；normal和repair均经同delivery函数全文放入实际messages，再据请求结果标记delivered/request_failed。B01同工具执行合同但没有方法正文。

`ActionSolver.generate`只做一次completion，宿主已预留其synthesis/repair额度。采样配置为Qwen3.8-Flash、seed137、temperature0、thinking关闭、4096输出tokens、JSON mode、120s时限。`make_qwen_client`是后续真实接线的lazy构造器，本次没有调用。

程序输出必须是仅有 `program_source` 字段的严格JSON；重复键、额外字段、Markdown或assembler修改/截断恢复均拒绝。复用 `assemble_program_ex(..., intended_answer_slot="SubmitActions")`，不写QA AnswerPayload。Direct复用验证过的official prompt方法、examples和家务plan parser，在宿主调度下只执行完整合法计划的第一动作。Direct不加载候选/几何。

同decision的yield向同一个coding agent交付最多3个公开结果页，合计feedback≤4000字符。宿主读取也计入同decision的6次reader预算；程序下一segment不保留变量，ResultStore账本保留到mutation。

审计仅保留请求/响应/程序/方法/输入hash、数值usage和固定status；无原prompt、源码、raw response、理由、URLs、凭据或GT。usage缺失明确标记，最终一次可能跨过token软阈值。

## 离线入口

`action_run.py --check`：零请求合同/方法/预算检查。

`action_run.py --fake`：三个环境×Direct/B01/B11共9次假任务；真实prompt/assembler、宿主、局部数值工具和macOS隔离执行，模型、深度和环境均是假输入。

`--process-worker`将假环境换成真实独立NDJSON V2 worker进程；`--output`仅允许本目录既有父目录下的新文件。入口无official/live选项，不能复用原V1实验授权。

P4继续检查固定分母/配对identity、综合fake故障、来源hash与旧baseline/active pointer保护，生成最终离线验收与使用说明。
