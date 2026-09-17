"""M14 LanceDB 后端：本地嵌入式向量库，支持 table branch / 版本回滚。

lancedb 未安装在当前环境，必须 lazy import（函数内 import）。
接口语义：create_or_open_table、write_entries、list_versions、checkout_version（回滚）、
create_branch。具体 lancedb API 版本未核验，标 TODO。
"""

from __future__ import annotations

from typing import Any


class LanceDBNotInstalledError(ImportError):
    """lancedb 未安装时抛出。"""


def _import_lancedb():
    try:
        import lancedb  # lazy import：顶层禁止 import 未安装库
    except ImportError as e:  # pragma: no cover
        raise LanceDBNotInstalledError(
            "lancedb 未安装；请安装后使用 LanceDBBackend，或用内存后端"
        ) from e
    return lancedb


class LanceDBBackend:
    """LanceDB 持久化后端（branch/版本回滚）。"""

    def __init__(self, db_path: str) -> None:
        lancedb = _import_lancedb()
        self._db = lancedb.connect(db_path)

    def create_or_open_table(self, name: str, schema: Any = None):
        """创建或打开表。TODO：lancedb 具体 create_table API 版本未核验。"""
        existing = set(self._db.table_names())
        if name in existing:
            return self._db.open_table(name)
        if schema is None:
            raise ValueError("首次建表必须提供 schema/data")
        return self._db.create_table(name, schema=schema)

    def write_entries(self, table_name: str, rows: list[dict]) -> None:
        tbl = self._db.open_table(table_name)
        tbl.add(rows)

    def list_versions(self, table_name: str) -> list[int]:
        """列出表版本号（用于回滚）。"""
        tbl = self._db.open_table(table_name)
        return [v["version"] for v in tbl.list_versions()]

    def checkout_version(self, table_name: str, version: int) -> None:
        """回滚到指定版本（M14 验收：回滚后可见条目回到旧版本）。"""
        tbl = self._db.open_table(table_name)
        tbl.checkout(version)

    def create_branch(self, table_name: str, branch_name: str) -> None:
        """创建 table branch（候选实验在 branch 上写，不影响主支）。TODO：API 未核验。"""
        tbl = self._db.open_table(table_name)
        tbl.create_branch(branch_name)  # TODO: lancedb branch API 具体名待核验
