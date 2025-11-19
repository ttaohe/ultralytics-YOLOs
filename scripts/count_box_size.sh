python scripts/yolo_resize_bbox_hist.py \
  --labels-root /home/hetao/graduate/data/VisDrone-VID-yolo/labels \
  --images-root /home/hetao/graduate/data/VisDrone-VID-yolo/images \
  --splits train val \
  --imgsz 640 1280 \
  --out-dir /home/hetao/graduate/ultralytics-YOLOs/runs/stats/resize_hist_bins_px \
  --bin-edges 0,8,16,32,64,128,256,512,1024


# python scripts/yolo_resize_bbox_hist.py \
#   --labels-root /home/hetao/graduate/data/VisDrone-VID-yolo/labels \
#   --images-root /home/hetao/graduate/data/VisDrone-VID-yolo/images \
#   --splits train val \
#   --imgsz 640 1280 \
#   --out-dir /home/hetao/graduate/ultralytics-YOLOs/runs/stats/resize_hist \
#   --plots