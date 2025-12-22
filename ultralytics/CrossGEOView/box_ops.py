from __future__ import annotations

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
	if boxes.size == 0:
		return boxes.copy()
	ret = boxes.copy()
	ret[:, 2] = boxes[:, 2] - boxes[:, 0]
	ret[:, 3] = boxes[:, 3] - boxes[:, 1]
	ret[:, 0] = boxes[:, 0] + ret[:, 2] * 0.5
	ret[:, 1] = boxes[:, 1] + ret[:, 3] * 0.5
	return ret


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""两组xyxy框的IoU，[Na,Nb]。"""
	if a.size == 0 or b.size == 0:
		return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
	ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
	bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
	inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
	inter_y1 = np.maximum(ay1[:, None], by1[None, :])
	inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
	inter_y2 = np.minimum(ay2[:, None], by2[None, :])
	inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
	inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
	inter = inter_w * inter_h
	a_area = (ax2 - ax1) * (ay2 - ay1)
	b_area = (bx2 - bx1) * (by2 - by1)
	union = a_area[:, None] + b_area[None, :] - inter
	iou = inter / (union + 1e-12)
	return iou.astype(np.float32)


def nms_by_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
	"""按类别分组的NMS，返回保留索引。"""
	keep = []
	for c in np.unique(classes):
		idx = np.where(classes == c)[0]
		order = idx[np.argsort(-scores[idx])]
		picked = []
		while order.size > 0:
			i = order[0]
			picked.append(i)
			if order.size == 1:
				break
			ious = iou_xyxy(boxes[i : i + 1], boxes[order[1:]])[0]
			remain = np.where(ious <= iou_thr)[0]
			order = order[1:][remain]
		keep.extend(picked)
	return np.array(keep, dtype=int)


def weighted_box_fusion(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.55) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""简化WBF：同类、IoU>thr进行加权平均，分数为均值或最大值。"""
	if boxes.size == 0:
		return boxes, scores, classes
	keep_boxes, keep_scores, keep_classes = [], [], []
	used = np.zeros(len(boxes), dtype=bool)
	for c in np.unique(classes):
		idx = np.where(classes == c)[0]
		sub_boxes, sub_scores = boxes[idx], scores[idx]
		order = idx[np.argsort(-sub_scores)]
		for i in order:
			if used[i]:
				continue
			cand = [i]
			ious = iou_xyxy(boxes[i : i + 1], boxes[idx])[0]
			neighbors = idx[(ious >= iou_thr) & (~used[idx])]
			cand = list(neighbors)
			w = scores[cand]
			bb = boxes[cand]
			w = w / (w.sum() + 1e-12)
			mbox = (bb * w[:, None]).sum(axis=0)
			s = scores[cand].mean()
			keep_boxes.append(mbox)
			keep_scores.append(s)
			keep_classes.append(c)
			used[cand] = True
	return np.array(keep_boxes), np.array(keep_scores), np.array(keep_classes)


