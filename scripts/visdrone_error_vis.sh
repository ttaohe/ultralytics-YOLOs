#!/bin/bash
# Visualize FP/FN on VisDrone split using a given YOLO weights file.
# Usage:
#   bash scripts/visdrone_error_vis.sh [WEIGHT] [DATA] [SPLIT] [IMGSZ] [CONF] [IOU_THR] [DEVICE] [DRAW_TP] [BATCH]
# Example:
#   bash scripts/visdrone_error_vis.sh /path/to/best.pt ultralytics/cfg/datasets/VisDrone.yaml test 1280 0.25 0.5 0 true
#
# Defaults aim to mirror your current val settings.
WEIGHT="${1:-/home/hetao/graduate/ultralytics-YOLOs/runs/train/train_yolov10l_visroneDet_epoch300_imgsz1280_batch8_mixup02/weights/best.pt}"
DATA="${2:-ultralytics/cfg/datasets/VisDrone.yaml}"
SPLIT="${3:-test}"
IMGSZ="${4:-1280}"
CONF="${5:-0.4}"
IOU_THR="${6:-0.2}"
DEVICE="${7:-}"
DRAW_TP="${8:-false}"
BATCH="${9:-1}"

TP_FLAG=""
if [ "$DRAW_TP" = "true" ] || [ "$DRAW_TP" = "1" ]; then
	TP_FLAG="--draw_tp"
fi
if [ -n "$DEVICE" ]; then
	python3 scripts/visualize_fp_fn.py --weights "$WEIGHT" --data "$DATA" --split "$SPLIT" --imgsz "$IMGSZ" --conf "$CONF" --iou_thr "$IOU_THR" --device "$DEVICE" --batch "$BATCH" $TP_FLAG
else
	python3 scripts/visualize_fp_fn.py --weights "$WEIGHT" --data "$DATA" --split "$SPLIT" --imgsz "$IMGSZ" --conf "$CONF" --iou_thr "$IOU_THR" --batch "$BATCH" $TP_FLAG
fi