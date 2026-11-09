# BLINKGPT

本目录包含新增运行、传输、部署、收集与评分代码，以及独立复制的 MV / SR v2 方法源。
不修改 `BLINK`、旧 `GPTexp/frozen` 或历史实验身份。

运行模型 `gpt-5.6-terra`，Responses API，high reasoning，seed 137，temperature 0，最多 8192 输出 token。
官方验证集直答精确使用数据中的 `benchmark_prompt`，不加原方法的 JSON 后缀。
方法对照：MV 93 题 × 2，SR 100 题 × 2；官方 val 直答：MV 133、SR 143，共 662 个 episode。
MV 旧缓存 `test` 是内部划分，按原图 SHA256 映射到官方 val；SR v2 延续原先的测试集提示词开发身份。
官方隐藏 test 的 EvalAI 线上成绩不包含在本地验证集报告中。

## 可靠运行

- 请求上限 36000 秒；每个逻辑请求最多 12 次 HTTP 尝试。失败退避 60–3600 秒，遵守 Retry-After，轮换 API 槽。
- 每个槽最多一个在途请求；唯一 supervisor 文件锁防止双重运行。
- 每次尝试、失败、成功均原子保存。续跑复用成功正文并验证同一 payload 哈希。中断的未知请求保留历史，受原始尝试上限约束。
- worker 崩溃最多恢复 3 次；已经记录的回答或错误 episode 不按正确性重跑。
- 固定完整分母；无效输出与最终服务失败计入未答对。记录请求模型和真实返回元数据，不伪造缺失字段。
- 凭据从现有 `xcodeapi/credentials.local.env` 读入内存，经 SSH stdin 传递，不进入命令、源码、模型消息和实验记录。
- 全量 zero-request preflight 验证原图、缓存、工具与初始 payload。模型调用前可通过 `/models` 做只读槽就绪检查。
- 最终收集包含远端归档逐成员复读、Mac 整档哈希与逐成员核验、固定分母宿主评分及双方收集回执。

## 使用

在 `GPTexp` 父目录执行：

```bash
python3.11 -m BLINKGPT.selfcheck
python3.11 -m BLINKGPT.prepare --output /absolute/local/run --remote-root /data/wangy/wangjh/harness3d/unique-run
python3.11 -m BLINKGPT.deploy deploy --local-root /absolute/local/run --remote-root /data/wangy/wangjh/harness3d/unique-run
python3.11 -m BLINKGPT.deploy qualify --local-root /absolute/local/run --remote-root /data/wangy/wangjh/harness3d/unique-run
python3.11 -m BLINKGPT.deploy start --local-root /absolute/local/run --remote-root /data/wangy/wangjh/harness3d/unique-run
python3.11 -m BLINKGPT.collect --watch --local-root /absolute/local/run --remote-root /data/wangy/wangjh/harness3d/unique-run
```

断点续跑复用同一 manifest 和远端目录，以 `deploy start` 启动。活动 owner 会拒绝重复启动。
缺失源码或输入身份不符时停止；不要重新 prepare 已启动的运行。
报告与成绩位于 `runs/<run-id>/实验结果.md` 与 `host-final-collection.json`。
