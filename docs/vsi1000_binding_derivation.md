# VSI1000 的 M5 绑定迁移

默认 `vsi_m4_1000.json` 通过 `inputs.derived_binding_root` 指定独立的
`{data_root}/derived_bindings_v4_vsi1000_reuse_audit_20261031`，并显式设置
`runtime.binding_derivation_audit_mode = "reuse_source_audit"`。原始 `published_inputs.json`、源目录、
题目划分和绑定引用保持原样。

实验读取 1000 题：训练 300 题、测试 700 题，分别覆盖全部 164 个场景。
Skill 验收从训练题中按题型和证据槽位抽取 8 题，使用同一题单进行
137/138 配对验收。测试题不进入 Skill 修订或候选选择。

原始绑定的场景审核覆盖这些场景的 3142 道原题。迁移器私下读取这些
原始目录条目和公开输入，以保留完整场景审核范围。`provider.entries`、
普通输入获取和 `prepare_published` 仍只接受所选 1000 题；迁移范围不增加
实验题目或数据划分。迁移和审核处理完整 3142 题，但只落盘所选 1000 题的
v4 绑定；每条派生回执仍包含完整场景来源和审核范围。

迁移沿用 `prepared_binding_derivation_v11.py` 的严格契约：原始代码必须
匹配已审核的 `3588bc8` 来源和 store 身份，原始 seed 保持 0，语义策略保持
`category-only`，参数仅允许已审核的校验版本差异。迁移重新核验原始
类别回执、数组身份及旧回执对应的文件身份。复用模式还要求原始完整
场景题组完全一致，当前 mask 冲突与类别关系代码保持旧版本身份。
它继承原始 mask 审核、数值校验结论、对象标记、绑定整体状态和
materialized 状态，只增加 v4 的语义策略与对象语义身份封印，不重做
mask 冲突审核或 mask/点云数值检查，也不补造原先未执行的深度检查。
旧对象不可用结论保持原样。

派生回执记录 `audit_mode`、`source_audit_reused=true`、
`new_mask_conflict_audits=0`、`new_numeric_validations=0`，以及零模型、
零检测、零 SAM2 请求。生成身份保持原始封印，类别审核不代表已评估
mask 质量。消费端比较可信上下文、输入数组和完整派生输出身份；
复用模式不会再次调用对象数值复审。实际工具仍执行自身的使用条件检查。
重复读取已发布缓存时，仍核验完整源场景、缓存载荷和派生回执；已一致的
输出直接复用，不再创建一份临时载荷。

统一实验入口向训练及测试的 system 阶段传递同一个
`--derived-binding-root`。训练内部的配对验收沿用训练 runner 的输入
provider。32 帧直接基线不需要 M5 绑定，继续使用原始 32 帧。

其他配置的迁移默认仍为 `reaudit`，可以通过
`--binding-derivation-audit-mode` 明确选择。两个模式使用独立的缓存目录。

离线验收应同时检查完整场景迁移回执、1000 题公开访问边界和
`SystemVariantRunner._prepare_one(..., preflight=True)`。迁移和输入准备
不代表在线解题、Skill 验收或最终测试已经执行。
