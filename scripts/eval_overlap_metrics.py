#!/usr/bin/env python3
"""
Evaluate detection metrics in overlap vs non-overlap regions using PE masks.

Usage:
  python scripts/eval_overlap_metrics.py \
    --weights runs/.../best.pt \
    --data ultralytics/cfg/datasets/MDMT_multiview.yaml \
    --imgsz 800 --batch 4 --device 0 --split val
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr, YAML
from ultralytics.utils.metrics import DetMetrics
from ultralytics.models.yolo.multiview.train import MultiviewTrainer
from ultralytics.models.yolo.multiview.val import MultiviewValidator


def _build_overlap_mask(coords_view, const_pe_eps=1e-6):
    """
    coords_view: (H, W, 4) tensor
    Returns: (H, W) bool, True = overlap (non-constant PE)
    """
    const0 = coords_view.new_tensor([1.0, 0.0, 1.0, 0.0])
    const1 = coords_view.new_tensor([-1.0, 0.0, -1.0, 0.0])
    diff0 = (coords_view - const0).abs().amax(dim=-1)
    diff1 = (coords_view - const1).abs().amax(dim=-1)
    # overlap region encoded as non-constant in both views
    valid0 = diff0 > const_pe_eps
    valid1 = diff1 > const_pe_eps
    return valid0, valid1


def _filter_by_mask_boxes_xyxy(boxes, mask):
    """Filter xyxy boxes by whether center lies inside mask."""
    if boxes.numel() == 0:
        return boxes, torch.zeros((0,), dtype=torch.bool, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = ((x1 + x2) * 0.5).round().long()
    cy = ((y1 + y2) * 0.5).round().long()
    h, w = mask.shape
    cx = cx.clamp(0, w - 1)
    cy = cy.clamp(0, h - 1)
    keep = mask[cy, cx]
    return boxes[keep], keep


class OverlapMultiviewValidator(MultiviewValidator):
    def __init__(self, *args, overlap_mode="per_view", **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_mode = overlap_mode
        self.overlap_metrics = [DetMetrics() for _ in range(self.num_views)]
        self.nonoverlap_metrics = [DetMetrics() for _ in range(self.num_views)]

    def init_metrics(self, model):
        super().init_metrics(model)
        for m in self.overlap_metrics + self.nonoverlap_metrics:
            m.names = model.names
            m.nc = len(model.names)

    def update_metrics(self, preds, batch):
        batch_coords = batch.get("coords", None)
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0

            # Determine view index
            view_idx = si % self.num_views
            batch_idx = si // self.num_views

            if batch_coords is None:
                LOGGER.warning("No coords found in batch; overlap metrics skipped.")
                return super().update_metrics(preds, batch)

            coords_view = batch_coords[batch_idx, view_idx]  # (H, W, 4)
            valid0, valid1 = _build_overlap_mask(coords_view)
            if self.overlap_mode == "intersection":
                overlap_mask = valid0 & valid1
            else:
                overlap_mask = valid0 if view_idx == 0 else valid1
            nonoverlap_mask = ~overlap_mask

            # Filter GT by overlap/nonoverlap
            gt_boxes = pbatch["bboxes"]
            gt_boxes_overlap, keep_gt_overlap = _filter_by_mask_boxes_xyxy(gt_boxes, overlap_mask)
            gt_boxes_nonoverlap, keep_gt_non = _filter_by_mask_boxes_xyxy(gt_boxes, nonoverlap_mask)
            gt_cls_overlap = pbatch["cls"][keep_gt_overlap] if keep_gt_overlap.numel() else pbatch["cls"][:0]
            gt_cls_non = pbatch["cls"][keep_gt_non] if keep_gt_non.numel() else pbatch["cls"][:0]

            # Filter preds by overlap/nonoverlap
            pred_boxes = predn["bboxes"]
            pred_boxes_overlap, keep_pred_overlap = _filter_by_mask_boxes_xyxy(pred_boxes, overlap_mask)
            pred_boxes_nonoverlap, keep_pred_non = _filter_by_mask_boxes_xyxy(pred_boxes, nonoverlap_mask)
            pred_conf_overlap = predn["conf"][keep_pred_overlap] if keep_pred_overlap.numel() else predn["conf"][:0]
            pred_cls_overlap = predn["cls"][keep_pred_overlap] if keep_pred_overlap.numel() else predn["cls"][:0]
            pred_conf_non = predn["conf"][keep_pred_non] if keep_pred_non.numel() else predn["conf"][:0]
            pred_cls_non = predn["cls"][keep_pred_non] if keep_pred_non.numel() else predn["cls"][:0]

            # Build pbatch dicts
            pbatch_overlap = dict(pbatch)
            pbatch_overlap["bboxes"] = gt_boxes_overlap
            pbatch_overlap["cls"] = gt_cls_overlap

            pbatch_non = dict(pbatch)
            pbatch_non["bboxes"] = gt_boxes_nonoverlap
            pbatch_non["cls"] = gt_cls_non

            # Build pred dicts
            pred_overlap = {
                "bboxes": pred_boxes_overlap,
                "conf": pred_conf_overlap,
                "cls": pred_cls_overlap,
            }
            pred_non = {
                "bboxes": pred_boxes_nonoverlap,
                "conf": pred_conf_non,
                "cls": pred_cls_non,
            }

            # Update overall metrics (unchanged)
            stat = {
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
            }
            self.metrics.update_stats(stat)
            self.view_metrics[view_idx].update_stats(stat)

            # Update overlap metrics
            cls_o = pbatch_overlap["cls"].cpu().numpy()
            no_pred_o = pred_overlap["cls"].shape[0] == 0
            stat_o = {
                **self._process_batch(pred_overlap, pbatch_overlap),
                "target_cls": cls_o,
                "target_img": np.unique(cls_o),
                "conf": np.zeros(0) if no_pred_o else pred_overlap["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred_o else pred_overlap["cls"].cpu().numpy(),
            }
            self.overlap_metrics[view_idx].update_stats(stat_o)

            # Update non-overlap metrics
            cls_n = pbatch_non["cls"].cpu().numpy()
            no_pred_n = pred_non["cls"].shape[0] == 0
            stat_n = {
                **self._process_batch(pred_non, pbatch_non),
                "target_cls": cls_n,
                "target_img": np.unique(cls_n),
                "conf": np.zeros(0) if no_pred_n else pred_non["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred_n else pred_non["cls"].cpu().numpy(),
            }
            self.nonoverlap_metrics[view_idx].update_stats(stat_n)

    def get_overlap_stats(self, save_dir):
        overlap_stats = {}
        non_stats = {}
        for i in range(self.num_views):
            vdir = Path(save_dir) / f"view_{i+1}"
            vdir.mkdir(parents=True, exist_ok=True)
            self.overlap_metrics[i].process(save_dir=vdir / "overlap", plot=False, on_plot=self.on_plot)
            self.overlap_metrics[i].clear_stats()
            self.nonoverlap_metrics[i].process(save_dir=vdir / "nonoverlap", plot=False, on_plot=self.on_plot)
            self.nonoverlap_metrics[i].clear_stats()
            for k, v in self.overlap_metrics[i].results_dict.items():
                overlap_stats[f"{k}_overlap_view{i+1}"] = v
            for k, v in self.nonoverlap_metrics[i].results_dict.items():
                non_stats[f"{k}_nonoverlap_view{i+1}"] = v
        return overlap_stats, non_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--overlap-mode", type=str, default="per_view", choices=["per_view", "intersection"])
    args = parser.parse_args()

    data_dict = YAML.load(args.data)
    trainer = MultiviewTrainer(
        overrides={
            "model": args.weights,
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
        }
    )
    trainer.data = data_dict

    # Build dataloader (val mode)
    dataset_path = data_dict.get(args.split, data_dict.get("val"))
    dataloader = trainer.get_dataloader(dataset_path, args.batch, rank=-1, mode="val")

    # Load model
    model = YOLO(args.weights).model
    model = model.to(trainer.device)
    model.eval()

    validator = OverlapMultiviewValidator(dataloader=dataloader, save_dir=trainer.save_dir, args=trainer.args)
    validator.num_views = data_dict.get("num_views", 2)
    validator.overlap_mode = args.overlap_mode
    validator(model=model)

    overlap_stats, non_stats = validator.get_overlap_stats(trainer.save_dir)
    LOGGER.info(colorstr("bright_green") + "Overlap metrics:")
    for k, v in overlap_stats.items():
        LOGGER.info(f"{k}: {v}")
    LOGGER.info(colorstr("bright_green") + "Non-overlap metrics:")
    for k, v in non_stats.items():
        LOGGER.info(f"{k}: {v}")


if __name__ == "__main__":
    main()
