import json
from pathlib import Path
from typing import Tuple

import numpy as np


def load_homography_json(path: str | Path) -> np.ndarray:
	"""从JSON文件读取3x3单应性矩阵，键为"H"或直接3x3数组。"""
	p = Path(path)
	obj = json.loads(p.read_text())
	H = obj["H"] if isinstance(obj, dict) and "H" in obj else obj
	H = np.asarray(H, dtype=np.float64).reshape(3, 3)
	return H


def save_homography_json(H: np.ndarray, path: str | Path) -> None:
	"""保存3x3单应性矩阵到JSON文件。"""
	p = Path(path)
	p.write_text(json.dumps({"H": np.asarray(H, dtype=float).tolist()}, ensure_ascii=False, indent=2))


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
	"""用单应性 H 变换二维点。

	Args:
		H: [3,3] 单应性矩阵，src->dst
		pts: [N,2] (x,y)

	Returns:
		[N,2] 变换后的(x,y)
	"""
	pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)  # [N,3]
	m = (H @ pts_h.T).T
	m = m[:, :2] / (m[:, 2:3] + 1e-12)
	return m


def warp_boxes_corners(H: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
	"""将xyxy框以四角点透视变换，再以最小外接轴对齐框回归为xyxy。"""
	if boxes_xyxy.size == 0:
		return boxes_xyxy.copy()
	xyxy = boxes_xyxy.astype(np.float64)
	x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
	corners = np.stack([
		np.stack([x1, y1], axis=1),
		np.stack([x2, y1], axis=1),
		np.stack([x2, y2], axis=1),
		np.stack([x1, y2], axis=1),
	], axis=1)  # [N,4,2]
	N = corners.shape[0]
	flat = corners.reshape(N * 4, 2)
	flat_w = warp_points(H, flat).reshape(N, 4, 2)
	min_xy = flat_w.min(axis=1)
	max_xy = flat_w.max(axis=1)
	ret = np.concatenate([min_xy, max_xy], axis=1)
	return ret


def clip_boxes_xyxy(boxes: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
	"""将框裁剪到图像宽高范围内。"""
	if boxes.size == 0:
		return boxes
	w, h = wh
	boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
	boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
	return boxes


