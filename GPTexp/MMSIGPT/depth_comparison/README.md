# MMSI 原图与直接添加深度图配对实验

本包独立实现 `Direct` 与 `Depth` 两臂，保留旧原型和冻结运行源码。

- 从 365 题公共冻结清单中按六类各取 5 题，排除已审阅诊断题，固定 seed=137。抽样不使用标准答案。
- 30 对、最多 60 次推理。同题两臂使用同一个 API；六个 API slot 各串行执行五对。先后顺序为 15 对原图先、15 对深度先，采用固定哈希排序。
- 两臂均使用 gpt-5.6-terra、Responses、high、temperature=0、seed=137、max_output_tokens=8192；无工具。请求和服务返回参数分别记录。服务报告的非空模型或 reasoning effort 不符时停止派发。
- Direct 原始图像字节和顺序不变，仅发送官方字母回答提示。Depth 依次发送全部 RGB、全部像素对齐深度图、短说明、同一题目和选项。
- 只使用现有预测深度。验证缓存完整内容哈希、NPZ 哈希、公共题目身份和无标签来源。每图有效值 P2–P98 线性灰度，暗近亮远；无效像素洋红；常数有效区域中灰。禁止重新估计或缩放深度来补缺失。
- 固定说明上限 80 个 o200k_base 词元；全部额外说明和图标签上限 160。完整说明实测 59 词元。预处理使用 SHA256 固定的词表，超限自动短版，仍超限拒绝。
- 每次派发前持久化唯一 dispatch。服务失败停止新派发，已发请求完成后收集；无自动重试、换 key 补答或续跑。停止后的未测题保留在 30 对固定分母中。
- 所有预测冻结并复核后，独立评分进程才读取固定哈希的 host reference 文件。报告全体准确率、合法回答率、完整配对数、得失转移、配对 bootstrap 95% CI 和精确 McNemar p。

这个比较估计“深度图＋必要说明”整体效果。30 对是小样本诊断，不直接用于概括完整 MMSI 的增益。API 模型列表核验不等于成功完成视觉推理测试。

从项目根运行：

```bash
python -B -m GPTexp.MMSIGPT.depth_comparison select \
  --source <公共manifest.json> --cache-index <服务器数据索引.json> \
  --reviewed <原始诊断.json> <新诊断.json> <review-data.json> \
  --output <selection.json> --slots 6

python -B -m GPTexp.MMSIGPT.depth_comparison qualify \
  --credentials <权限600的本地env> --output <api-qualification.json> --slots 6

python -B -m GPTexp.MMSIGPT.depth_comparison prepare \
  --selection <selection.json> --tokenizer <o200k_base.tiktoken> \
  --directory <新的空运行目录>

python -B -m GPTexp.MMSIGPT.depth_comparison execute \
  --directory <运行目录> --credentials-stdin

python -B -m GPTexp.MMSIGPT.depth_comparison freeze --directory <运行目录>
python -B -m GPTexp.MMSIGPT.depth_comparison score \
  --directory <运行目录> --references <host答案文件> --reference-sha256 <固定hash>
```

服务器执行时通过 SSH 标准输入注入 `{opaque_api_id: key}` JSON，仅在进程内存使用，不复制凭据文件。stdin 内容由本地 Python 从用户指定文件读取，禁止使用 shell 展开或输出凭据。

`selfcheck` 使用临时合成输入和 fake client，并封禁网络；它不是实验结果。

服务器首批六次服务失败后的迁移使用 `prepare-from-frozen`。它核验父运行的
manifest、冻结 envelope、所有文件哈希和真实 dispatch，排除任何一臂已派发的题目；
仅复制剩余 24 对，六类和六个 API 各 4 对，先后顺序各 12 对。原始图像、
已渲染深度、说明和 payload 哈希全部保持，子运行单独固定 Mac 环境和新源码。
子运行最多 48 请求，加父运行 6 请求共 54；父运行保留且不重启。
