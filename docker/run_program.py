"""沙箱容器入口：读取经 M9 AST 检查的 program 并执行（M10）。

容器以 --read-only --tmpfs /tmp --network none 启动，
重建产物只读挂载于 /data（final-test 目录绝不挂载，硬约束 9）。
"""

import json
import sys


def main() -> int:
    # program 通过环境变量/stdin 传入（具体注入方式 TODO，与 M10 kernel 对齐）
    program_source = sys.stdin.read()
    namespace: dict = {}
    try:
        exec(compile(program_source, "<episode_program>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # 执行失败上报 error_code，由宿主侧归因
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "answer": namespace.get("ANSWER")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
