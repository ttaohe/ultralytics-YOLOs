#!/usr/bin/env bash
GPU=3
FREE_TH=8000  # MB

CMD="python train_multiview.py \
  --model ultralytics/cfg/models/v10/yolov10l-multiview_earlyfusion-nodownfactor.yaml \
  --name yolov10l-multiview-disaug-earlyfusion-P3-nooverlapweight-alignloss05-nomask-randvid-stride3-imgsz640-residualgate-gate_xout_base0.693 \
  --mask-key False \
  --align-loss-weight 0.5 \
  --device 3 \
  --video-stride 3 \
  --video-rand-start \
  --imgsz 640 \
  --gate-method gate_xout \
  --random-crop-size 640 --random-crop-prob 1.0 --val-imgsz 1920"

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU)
  FREE=${FREE//[^0-9]/}
  if [ "$FREE" -ge "$FREE_TH" ]; then
    echo "GPU $GPU free ${FREE}MB >= ${FREE_TH}MB, starting..."
    eval "$CMD" 2>&1 | tee run_log
    break
  fi
  echo "GPU $GPU free ${FREE}MB < ${FREE_TH}MB, wait..."
  sleep 30
done
