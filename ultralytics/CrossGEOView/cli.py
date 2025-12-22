from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np

from .late_fusion import CrossViewLateFusion, FusionConfig


def main():
	parser = argparse.ArgumentParser(description="跨视角YOLO晚期融合（正射平面）")
	parser.add_argument("model", type=str, help="*.pt 模型路径")
	parser.add_argument("ortho", type=str, help="正射（伪正射）图像路径")
	parser.add_argument("view", type=str, help="倾斜视角图像路径")
	parser.add_argument("H_view2ortho", type=str, help="view->ortho 的H(3x3) json文件")
	parser.add_argument("H_ortho2view", type=str, help="ortho->view 的H(3x3) json文件（用于回投，可与前者互逆）")
	parser.add_argument("out_dir", type=str, help="输出目录")
	parser.add_argument("--device", type=str, default=None, help="cuda:0 或 cpu")
	parser.add_argument("--use-wbf", action="store_true", help="使用WBF，否则使用NMS")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--nms-iou", type=float, default=0.6)
	parser.add_argument("--wbf-iou", type=float, default=0.55)
	parser.add_argument("--max-det", type=int, default=300)

	args = parser.parse_args()
	out = Path(args.out_dir)
	out.mkdir(parents=True, exist_ok=True)

	fuser = CrossViewLateFusion(args.model, device=args.device)
	cfg = FusionConfig(conf_thr=args.conf, iou_thr_nms=args.nms_iou, iou_thr_wbf=args.wbf_iou, max_det=args.max_det, use_wbf=args.use_wbf)

	# 融合于正射
	f_boxes, f_scores, f_classes, wh_ortho = fuser.fuse_on_ortho(args.ortho, args.view, args.H_view2ortho, cfg)

	# 保存正射融合结果
	from .late_fusion import CrossViewLateFusion as _C
	_C.draw_and_save(args.ortho, f_boxes, f_scores, f_classes, out / "ortho_fused.jpg")

	# 回投到view并保存
	boxes_view = fuser.project_back(f_boxes, args.H_ortho2view, wh_view=_infer_wh_from_image(args.view))
	_C.draw_and_save(args.view, boxes_view, f_scores, f_classes, out / "view_backprojected.jpg")

	# 导出融合结果为json
	(Path(out) / "ortho_fused.json").write_text(json.dumps({
		"boxes": f_boxes.tolist(),
		"scores": f_scores.tolist(),
		"classes": f_classes.tolist(),
	}, ensure_ascii=False, indent=2))


def _infer_wh_from_image(path: str | Path) -> tuple[int, int]:
	from PIL import Image
	with Image.open(path) as im:
		w, h = im.size
	return (w, h)


if __name__ == "__main__":
	main()


