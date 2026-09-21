"""skill3d.tools 包（系统架构3.md §2.1 目录树）。

import 本包即注册全部确定性几何 Tool 到 REGISTRY。
"""

from . import geometry_tools  # noqa: F401  # 侧效应：注册 Tool
from .mock_switch import MockContaminationError, MockSwitch
from .registry import REGISTRY, ToolRegistry, call_tool
from .scene_handle import SceneHandle

