# MMSI GPT 开发约定

- MMSI 使用 GPT 的方法、提示、工具接入及后续修复统一放在本目录。
- `scripts/MMSIRunner` 和 `GPTexp/frozen/mmsi` 保留原始归档，不在其中修改代码。
- 公共 GPTexp 文件只负责调度、模型绑定、部署及路由连接。
- 保留 Direct/Harness 隔离：Direct 只使用原图与题目选项，不加载 prepared 几何、候选或 Skill。
- 新方法使用新运行清单及源码身份；不修改正在运行或已有的冻结实验。
- 离线工程验证使用合成输入、fake client 和临时目录；凭据、参考答案及真实数据不参与代码迁移测试。

