# Xcode API

独立 Python 客户端，仅依赖标准库。以下示例在 `wjhICLR` 根目录运行，使用 `.venv/bin/python`。

## 配置

在 `xcodeapi/credentials.local.env` 填写对应分组的 API Key：

```dotenv
XCODE_GPT_API_KEY=你的GPT令牌
XCODE_GEMINI_API_KEY=你的Gemini令牌
XCODE_CLAUDE_API_KEY=
```

该文件已被 Git 忽略，权限应为 `600`。客户端自动读取它，也支持同名环境变量；
同系列环境变量优先。通用令牌可填 `XCODE_API_KEY`。

`profiles.json` 中的当前配置：

| 档案名 | 模型 ID | 接口 |
| --- | --- | --- |
| `gpt` | `gpt-6.1-sol` | `https://xcode.best/v1/responses` |
| `gemini` | `gemini-3.8-flash-tiered` | `https://xcode.best/v1/chat/completions` |
| `claude` | 未配置 | `https://xcode.best/v1/messages` |

换模型可修改 `profiles.json`，或在 `from_profile()` 中传入 `model="准确模型ID"`。
Claude 使用前需填写模型和对应分组的 Key，当前未验证。

## Python 调用

```python
from xcodeapi import XcodeClient

profile = "gpt"  # 使用 Gemini 时改为 "gemini"
thinking = {
    "gpt": {"reasoning": {"effort": "high"}},
    "gemini": {"reasoning_effort": "high"},
}
client = XcodeClient.from_profile(profile, max_requests=1)

result = client.generate(
    [{"role": "user", "content": "你好，请简短介绍自己。"}],
    max_output_tokens=4096,
    extra_body=thinking[profile],
)

print(result.text)        # 回答
print(result.metadata())  # token、耗时等数值信息
```

将 `high` 改为 `low` 可调整思考档位；省略 `extra_body` 使用模型默认设置。
档位控制思考程度，不保证固定推理 token 数量。

返回结果还包含 `usage`、`reasoning`、`tool_calls`、`finish_reason`、
`requested_model`、`returned_model` 和完整原生响应 `raw`。

### 图片输入

把上例的消息替换为以下内容即可；多张图片按列表顺序添加：

```python
from xcodeapi import text_part, image_file_part

messages = [{
    "role": "user",
    "content": [
        text_part("描述这张图片。"),
        image_file_part("/absolute/path/image.png"),
    ],
}]
# result = client.generate(messages, max_output_tokens=4096, extra_body=thinking[profile])
```

支持 JPEG、PNG、GIF、WebP。本地图片转为 base64，也可用 `image_url_part("https://...")`。

### 其他接口

| 方法 | 用途 |
| --- | --- |
| `client.prepare(messages, **options)` | 离线构造请求，不读取 Key、不访问网络 |
| `client.send(payload)` | 发送当前协议的原生请求，用于完整工具历史等场景 |
| `client.list_models()` | 请求服务模型目录 |

`generate()` 等价于 `send(prepare(...))`。`extra_body` 接受当前协议的原生参数，
例如工具定义、JSON 格式和思考设置；原生 `payload` 的模型 ID 必须与客户端配置一致。

## 命令行

```bash
# 离线检查
.venv/bin/python -m xcodeapi preflight
.venv/bin/python -m xcodeapi prepare --profile gpt

# 实际访问 API
.venv/bin/python -m xcodeapi smoke --profile gpt --live --print-text
.venv/bin/python -m xcodeapi smoke --profile gemini --live --print-text
.venv/bin/python -m xcodeapi models --profile gemini --live
```

## 调用约定

- `max_requests` 是单个客户端实例的请求尝试上限，含模型目录查询和失败请求。
- 客户端采用同步、非流式调用，不自动重试；错误可捕获 `XcodeAPIError`。
- 库不自动保存请求或回答。Gemini 的 token 分项可能不完整，保留服务返回值。

[平台文档](https://docs.xcode.best/docs/register)
