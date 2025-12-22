from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .geom import load_homography_json, warp_boxes_corners, clip_boxes_xyxy
from .box_ops import nms_by_class, weighted_box_fusion


@dataclass
class FusionConfig:
	conf_thr: float = 0.25
	iou_thr_nms: float = 0.6
	iou_thr_wbf: float = 0.55
	max_det: int = 300
	use_wbf: bool = True


class CrossViewLateFusion:
	"""两视角YOLO推理后在正射平面做晚期融合，并可选回投到各自视角保存结果。"""

	def __init__(self, model_path: str | Path, device: str | None = None):
		self.model = YOLO(str(model_path))
		if device is not None:
			self.model.to(device)

	@staticmethod
	def _yolo_predict(model: YOLO, img_path: str | Path, conf: float, max_det: int):
		res = model.predict(source=str(img_path), conf=conf, max_det=max_det, verbose=False)
		# 仅取第一张图
		res0 = res[0]
		boxes = res0.boxes.xyxy.cpu().numpy()
		scores = res0.boxes.conf.cpu().numpy()
		classes = res0.boxes.cls.cpu().numpy().astype(int)
		wh = (res0.orig_shape[1], res0.orig_shape[0])
		return boxes, scores, classes, wh

	def fuse_on_ortho(
		self,
		img_ortho: str | Path,
		img_view: str | Path,
		H_view2ortho_json: str | Path,
		cfg: FusionConfig = FusionConfig(),
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
		"""在正射图上融合来自正射与倾斜视角的检测框。"""
		H_vo = load_homography_json(H_view2ortho_json)
		b1, s1, c1, wh1 = self._yolo_predict(self.model, img_ortho, cfg.conf_thr, cfg.max_det)
		b2, s2, c2, _ = self._yolo_predict(self.model, img_view, cfg.conf_thr, cfg.max_det)
		# 将view框投到正射
		b2o = warp_boxes_corners(H_vo, b2)
		# 合并
		boxes = np.concatenate([b1, b2o], axis=0)
		scores = np.concatenate([s1, s2], axis=0)
		classes = np.concatenate([c1, c2], axis=0)
		boxes = clip_boxes_xyxy(boxes, wh1)
		# 融合
		if cfg.use_wbf:
			f_boxes, f_scores, f_classes = weighted_box_fusion(boxes, scores, classes, iou_thr=cfg.iou_thr_wbf)
		else:
			keep = nms_by_class(boxes, scores, classes, iou_thr=cfg.iou_thr_nms)
			f_boxes, f_scores, f_classes = boxes[keep], scores[keep], classes[keep]
		return f_boxes, f_scores, f_classes, wh1

	def project_back(
		self,
		f_boxes_ortho: np.ndarray,
		H_ortho2view_json: str | Path,
		wh_view: Tuple[int, int],
	) -> np.ndarray:
		"""将正射平面融合结果回投到某视角。"""
		H_ov = load_homography_json(H_ortho2view_json)
		boxes_view = warp_boxes_corners(H_ov, f_boxes_ortho)
		boxes_view = clip_boxes_xyxy(boxes_view, wh_view)
		return boxes_view

	@staticmethod
	def draw_and_save(img_path: str | Path, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, out_path: str | Path) -> None:
		img = cv2.imread(str(img_path))
		for (x1, y1, x2, y2), sc, cl in zip(boxes.astype(int), scores, classes):
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(img, f"{int(cl)}:{sc:.2f}", (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		cv2.imwrite(str(out_path), img)


