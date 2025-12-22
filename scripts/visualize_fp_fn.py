#!/usr/bin/env python3
"""
Visualize false positives (FP) and false negatives (FN) for a YOLO model on a given dataset split.

- Loads dataset structure from a Ultralytics YAML (e.g., VisDrone.yaml)
- Runs inference with provided weights on the requested split (train/val/test)
- Matches predictions to GT via IoU threshold per class
  - Unmatched predictions -> FP (red)
  - Unmatched GT boxes -> FN (blue)
  - (Optional) TPs drawn light green for context
- Saves per-image visualizations and a summary CSV
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

try:
	from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
	print("Error: Failed to import ultralytics. Please ensure 'ultralytics' is installed in your environment.", file=sys.stderr)
	raise


@dataclass
class Box:
	cls: int
	conf: float
	xyxy: np.ndarray  # [x1, y1, x2, y2], float


def load_dataset_paths(data_yaml_path: str, split: str) -> Tuple[Path, Path, Path]:
	"""
	Returns:
		images_dir, labels_dir, dataset_root
	"""
	with open(data_yaml_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)

	dataset_root = Path(cfg.get("path", "")).expanduser().resolve()
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

	split_key = split.lower().strip()
	if split_key not in {"train", "val", "test"}:
		raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test.")

	images_rel = cfg.get(split_key, None)
	if images_rel is None:
		raise KeyError(f"Split '{split_key}' not defined in {data_yaml_path}")
	images_dir = (dataset_root / images_rel).resolve()
	if not images_dir.exists():
		raise FileNotFoundError(f"Images directory does not exist for split '{split_key}': {images_dir}")

	# YOLO label convention: labels/<split>
	labels_dir = (dataset_root / "labels" / Path(images_rel).name).resolve()
	return images_dir, labels_dir, dataset_root


def read_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
	"""
	Read YOLO-format label file and return list of (cls, xyxy) with absolute pixel coords.
	"""
	gt: List[Tuple[int, np.ndarray]] = []
	if not label_path.exists():
		return gt

	with open(label_path, "r", encoding="utf-8") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) < 5:
				continue
			c = int(float(parts[0]))
			x, y, w, h = map(float, parts[1:5])
			# convert from normalized xywh (center-based) to absolute xyxy
			cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
			x1 = max(0.0, cx - bw / 2.0)
			y1 = max(0.0, cy - bh / 2.0)
			x2 = min(float(img_w - 1), cx + bw / 2.0)
			y2 = min(float(img_h - 1), cy + bh / 2.0)
			gt.append((c, np.array([x1, y1, x2, y2], dtype=np.float32)))
	return gt


def compute_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""
	Compute IoU matrix between a (N,4) and b (M,4) in xyxy format.
	"""
	if a.size == 0 or b.size == 0:
		return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

	ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
	bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

	inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
	inter_y1 = np.maximum(ay1[:, None], by1[None, :])
	inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
	inter_y2 = np.minimum(ay2[:, None], by2[None, :])
	inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
	inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
	inter_area = inter_w * inter_h

	area_a = (ax2 - ax1) * (ay2 - ay1)
	area_b = (bx2 - bx1) * (by2 - by1)
	union = area_a[:, None] + area_b[None, :] - inter_area
	iou = np.where(union > 0, inter_area / union, 0.0).astype(np.float32)
	return iou


def greedy_match_by_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thr: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
	"""
	Greedy IoU matching:
	- returns (matches, unmatched_pred_idx, unmatched_gt_idx)
	- matches: list of (pi, gi)
	"""
	if pred_boxes.size == 0 and gt_boxes.size == 0:
		return [], [], []
	if pred_boxes.size == 0:
		return [], [], list(range(gt_boxes.shape[0]))
	if gt_boxes.size == 0:
		return [], list(range(pred_boxes.shape[0])), []

	iou = compute_iou_matrix(pred_boxes, gt_boxes)
	# consider pairs over threshold
	candidates = np.argwhere(iou >= iou_thr)
	if candidates.size == 0:
		return [], list(range(pred_boxes.shape[0])), list(range(gt_boxes.shape[0]))

	# sort by IoU descending for greedy selection
	scores = iou[candidates[:, 0], candidates[:, 1]]
	order = np.argsort(-scores)
	candidates = candidates[order]

	matched_pred = set()
	matched_gt = set()
	matches: List[Tuple[int, int]] = []
	for pi, gi in candidates:
		if (int(pi) in matched_pred) or (int(gi) in matched_gt):
			continue
		matched_pred.add(int(pi))
		matched_gt.add(int(gi))
		matches.append((int(pi), int(gi)))

	unmatched_pred = [i for i in range(pred_boxes.shape[0]) if i not in matched_pred]
	unmatched_gt = [i for i in range(gt_boxes.shape[0]) if i not in matched_gt]
	return matches, unmatched_pred, unmatched_gt


def draw_boxes(
	image: np.ndarray,
	tp_boxes: List[np.ndarray],
	fp_boxes: List[np.ndarray],
	fn_boxes: List[np.ndarray],
	class_names: Dict[int, str],
	tp_classes: List[int],
	fp_classes: List[int],
	fn_classes: List[int],
	fp_ious: List[float] = None,
) -> np.ndarray:
	"""
	Draw boxes on a copy of the image.
	- TP: green
	- FP: red
	- FN: blue
	"""
	out = image.copy()

	def put_box(box: np.ndarray, color: Tuple[int, int, int], label: str) -> None:
		x1, y1, x2, y2 = map(int, box.tolist())
		cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
		if label:
			cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + 7 * len(label), y1), color, -1)
			cv2.putText(out, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

	for b, c in zip(tp_boxes, tp_classes):
		cls_name = class_names.get(int(c), str(int(c)))
		put_box(b, (0, 200, 0), f"TP {cls_name}")
	
	if fp_ious is None:
		fp_ious = [0.0] * len(fp_boxes)

	for b, c, iou in zip(fp_boxes, fp_classes, fp_ious):
		cls_name = class_names.get(int(c), str(int(c)))
		label = f"FP {cls_name}"
		if iou > 0:
			label += f" ({iou:.2f})"
		put_box(b, (0, 0, 255), label)

	for b, c in zip(fn_boxes, fn_classes):
		cls_name = class_names.get(int(c), str(int(c)))
		put_box(b, (255, 100, 0), f"FN {cls_name}")
	return out


def draw_gt_boxes(
	image: np.ndarray,
	gt_boxes: np.ndarray,
	gt_classes: np.ndarray,
	class_names: Dict[int, str],
) -> np.ndarray:
	"""
	Draw ground-truth boxes only, with class labels.
	"""
	out = image.copy()

	for box, c in zip(gt_boxes, gt_classes):
		x1, y1, x2, y2 = map(int, box.tolist())
		cls_name = class_names.get(int(c), str(int(c)))
		color = (0, 200, 0)  # green for GT
		label = f"GT {cls_name}"
		cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
		cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + 7 * len(label), y1), color, -1)
		cv2.putText(out, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

	return out


def load_class_names(data_yaml_path: str) -> Dict[int, str]:
	with open(data_yaml_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	names = cfg.get("names", {})
	if isinstance(names, dict):
		return {int(k): str(v) for k, v in names.items()}
	if isinstance(names, list):
		return {i: str(n) for i, n in enumerate(names)}
	return {}


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize FP/FN for a YOLO model on a dataset split.")
	parser.add_argument("--weights", type=str, required=True, help="Path to model weights (e.g., best.pt)")
	parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/VisDrone.yaml", help="Dataset YAML path")
	parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
	parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
	parser.add_argument("--batch", type=int, default=1, help="Inference batch size")
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
	parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for TP matching")
	parser.add_argument("--device", type=str, default="", help="Device for inference, e.g., '0' or 'cpu'")
	parser.add_argument("--save_dir", type=str, default="", help="Output directory. Auto if empty.")
	parser.add_argument("--draw_tp", action="store_true", help="Also draw TP boxes (green)")
	args = parser.parse_args()

	images_dir, labels_dir, _ = load_dataset_paths(args.data, args.split)

	# Validate labels availability
	labels_available = labels_dir.exists()
	if not labels_available:
		print(f"Warning: Labels directory not found for split '{args.split}': {labels_dir}", file=sys.stderr)
		print(
			"Cannot compute FP/FN without GT labels. Please ensure this split has labels, or use --split val.",
			file=sys.stderr,
		)
		sys.exit(2)

	class_names = load_class_names(args.data)

	model = YOLO(args.weights)

	# Collect image paths
	img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	image_paths = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in img_exts]
	if len(image_paths) == 0:
		raise FileNotFoundError(f"No images found under: {images_dir}")

	# Prepare output directories
	model_stem = Path(args.weights).stem
	time_tag = time.strftime("%Y%m%d-%H%M%S")
	base_save_dir = Path(args.save_dir) if args.save_dir else Path("runs") / "error_vis" / f"{model_stem}_{args.split}_{time_tag}"
	vis_dir = (base_save_dir / "images").resolve()
	gt_dir = (base_save_dir / "images_gt").resolve()
	vis_dir.mkdir(parents=True, exist_ok=True)
	gt_dir.mkdir(parents=True, exist_ok=True)

	summary_csv = (base_save_dir / "summary.csv").resolve()
	images_with_fp_txt = (base_save_dir / "images_with_fp.txt").resolve()
	images_with_fn_txt = (base_save_dir / "images_with_fn.txt").resolve()

	images_with_fp: List[str] = []
	images_with_fn: List[str] = []

	with open(summary_csv, "w", newline="", encoding="utf-8") as f_csv:
		writer = csv.writer(f_csv)
		writer.writerow(["image", "num_gt", "num_pred", "tp", "fp", "fn", "fp_classes", "fn_classes"])

		for img_path in image_paths:
			res_list = model.predict(
				source=str(img_path),
				imgsz=args.imgsz,
				conf=args.conf,
				batch=1,
				device=args.device if args.device else None,
				verbose=False,
			)
			if not res_list:
				print(f"Warning: empty prediction result for image: {img_path}", file=sys.stderr)
				continue
			res = res_list[0]

			img = cv2.imread(str(img_path))
			if img is None:
				print(f"Warning: failed to read image: {img_path}", file=sys.stderr)
				continue
			h, w = img.shape[:2]
			# GT
			# Handle subdirectories (e.g. VisDrone VID: images/test/seq/001.jpg -> labels/test/seq/001.txt)
			try:
				rel_path = img_path.relative_to(images_dir)
				label_path = labels_dir / rel_path.with_suffix(".txt")
			except ValueError:
				label_path = labels_dir / (img_path.stem + ".txt")

			gt_pairs = read_yolo_label_file(label_path, w, h)
			gt_classes = np.array([int(c) for c, _ in gt_pairs], dtype=np.int32)
			gt_boxes = np.stack([b for _, b in gt_pairs], axis=0) if len(gt_pairs) else np.zeros((0, 4), dtype=np.float32)
			# Predictions
			preds: List[Box] = []
			if res.boxes is not None and len(res.boxes) > 0:
				xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
				conf = res.boxes.conf.cpu().numpy().astype(np.float32)
				cls = res.boxes.cls.cpu().numpy().astype(np.int32)
				for i in range(xyxy.shape[0]):
					preds.append(Box(cls=int(cls[i]), conf=float(conf[i]), xyxy=xyxy[i]))
			pred_classes = np.array([b.cls for b in preds], dtype=np.int32) if preds else np.zeros((0,), dtype=np.int32)
			pred_boxes = np.stack([b.xyxy for b in preds], axis=0) if preds else np.zeros((0, 4), dtype=np.float32)

			# Per-class matching
			tp_boxes: List[np.ndarray] = []
			fp_boxes: List[np.ndarray] = []
			fn_boxes: List[np.ndarray] = []
			tp_cls_list: List[int] = []
			fp_cls_list: List[int] = []
			fn_cls_list: List[int] = []
			fp_ious_list: List[float] = []

			all_classes = set(pred_classes.tolist()) | set(gt_classes.tolist())
			for c in sorted(all_classes):
				p_idx = np.where(pred_classes == c)[0]
				g_idx = np.where(gt_classes == c)[0]
				p_boxes_c = pred_boxes[p_idx] if p_idx.size else np.zeros((0, 4), dtype=np.float32)
				g_boxes_c = gt_boxes[g_idx] if g_idx.size else np.zeros((0, 4), dtype=np.float32)

				matches, un_p, un_g = greedy_match_by_iou(p_boxes_c, g_boxes_c, args.iou_thr)
				# TP
				for pi, gi in matches:
					tp_boxes.append(p_boxes_c[pi])
					tp_cls_list.append(int(c))

				# Compute FP IoUs (with any GT of same class)
				if len(un_p) > 0 and g_boxes_c.shape[0] > 0:
					# IoU matrix between FP boxes and all GT boxes of same class
					fp_boxes_c = p_boxes_c[un_p]
					iou_matrix = compute_iou_matrix(fp_boxes_c, g_boxes_c)
					# For each FP, find max IoU with any GT
					max_ious = iou_matrix.max(axis=1) if iou_matrix.size > 0 else np.zeros(len(un_p))
				else:
					max_ious = np.zeros(len(un_p))

				# FP
				for i, pi in enumerate(un_p):
					fp_boxes.append(p_boxes_c[pi])
					fp_cls_list.append(int(c))
					fp_ious_list.append(float(max_ious[i]))

				# FN
				for gi in un_g:
					fn_boxes.append(g_boxes_c[gi])
					fn_cls_list.append(int(c))

			out_img = draw_boxes(
				image=img,
				tp_boxes=tp_boxes if args.draw_tp else [],
				fp_boxes=fp_boxes,
				fn_boxes=fn_boxes,
				class_names=class_names,
				tp_classes=tp_cls_list if args.draw_tp else [],
				fp_classes=fp_cls_list,
				fn_classes=fn_cls_list,
				fp_ious=fp_ious_list,
			)

			# Construct relative path string for CSV/txt (e.g., "seq1/001.jpg")
			try:
				rel_path_obj = img_path.relative_to(images_dir)
				img_rel_name = str(rel_path_obj)
			except ValueError:
				img_rel_name = img_path.name
				rel_path_obj = Path(img_path.name)

			num_gt = gt_boxes.shape[0]
			num_pred = pred_boxes.shape[0]
			num_tp = len(tp_boxes)
			num_fp = len(fp_boxes)
			num_fn = len(fn_boxes)
			if num_fp > 0:
				images_with_fp.append(img_rel_name)
			if num_fn > 0:
				images_with_fn.append(img_rel_name)

			# Save detection visualization (preserve directory structure)
			save_path = vis_dir / rel_path_obj.with_name(f"{rel_path_obj.stem}_fpfn.jpg")
			save_path.parent.mkdir(parents=True, exist_ok=True)
			cv2.imwrite(str(save_path), out_img)


			if gt_boxes.shape[0] > 0 and num_fp > 0:
				fp_arr = np.stack(fp_boxes, axis=0).astype(np.float32)
				iou_fp_gt = compute_iou_matrix(fp_arr, gt_boxes)

				close_gt_idx = set()
				for i in range(iou_fp_gt.shape[0]):
					j = int(np.argmax(iou_fp_gt[i]))
					if iou_fp_gt[i, j] > 0.1:  
						close_gt_idx.add(j)

				if close_gt_idx:
					sel_idx = sorted(close_gt_idx)
					gt_boxes_sel = gt_boxes[sel_idx]
					gt_classes_sel = gt_classes[sel_idx]
				else:
					gt_boxes_sel = gt_boxes
					gt_classes_sel = gt_classes

				gt_img = draw_gt_boxes(img, gt_boxes_sel, gt_classes_sel, class_names)
				gt_save_path = gt_dir / rel_path_obj.with_name(f"{rel_path_obj.stem}_gt.jpg")
				gt_save_path.parent.mkdir(parents=True, exist_ok=True)
				cv2.imwrite(str(gt_save_path), gt_img)

			writer.writerow(
				[
					img_rel_name,
					num_gt,
					num_pred,
					num_tp,
					num_fp,
					num_fn,
					";".join([str(c) for c in fp_cls_list]),
					";".join([str(c) for c in fn_cls_list]),
				]
			)

	# Save helper lists
	if images_with_fp:
		with open(images_with_fp_txt, "w", encoding="utf-8") as f_fp:
			f_fp.write("\n".join(images_with_fp))
	if images_with_fn:
		with open(images_with_fn_txt, "w", encoding="utf-8") as f_fn:
			f_fn.write("\n".join(images_with_fn))

	print(f"Done. Visualizations: {vis_dir}")
	print(f"Summary CSV: {summary_csv}")
	if images_with_fp:
		print(f"Images with FP: {images_with_fp_txt}")
	if images_with_fn:
		print(f"Images with FN: {images_with_fn_txt}")


if __name__ == "__main__":
	main()


