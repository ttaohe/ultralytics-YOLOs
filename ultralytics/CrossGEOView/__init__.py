"""
CrossGEOView: 基于几何对齐的跨视角晚期融合工具集（一期）。

模块结构：
- geom.py: 单应性矩阵加载/保存、点/框透视变换、可视化辅助。
- box_ops.py: 常用框操作（NMS、WBF简化版、IoU计算）。
- late_fusion.py: 两视角独立推理 + 正射平面对齐 + 融合 + 回投。
- cli.py: 命令行入口。
"""

from .late_fusion import CrossViewLateFusion

__all__ = ["CrossViewLateFusion"]


