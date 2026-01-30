#!/usr/bin/env bash
GPU=5
FREE_TH=12000  # MB

CMD="python train_multiview.py \
  --model ultralytics/cfg/models/v10/yolov10l-multiview_earlyfusion-nodownfactor.yaml \
  --name yolov10l-multiview-disaug-earlyfusion-P3-nooverlapweight-alignloss05-nomask-randvid-stride3-imgsz640-residualgate-gate_pe_base0.693-valorigin-overlapsample \
  --mask-key False \
  --align-loss-weight 0.5 \
  --device 5 \
  --video-stride 3 \
  --video-rand-start \
  --imgsz 640 \
  --gate-method gate_pe \
  --random-crop-size 640 --random-crop-prob 1.0 --val-original"

# --val-tile  --val-tile-size 640 --val-tile-stride 480

while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU)
  FREE=${FREE//[^0-9]/}
  if [ "$FREE" -ge "$FREE_TH" ]; then
    echo "GPU $GPU free ${FREE}MB >= ${FREE_TH}MB, starting..."
    NAME=$(echo "$CMD" | sed -n 's/.*--name[[:space:]]\([^[:space:]]\+\).*/\1/p')
    BASE_DIR="runs/train_multiview"
    LOG_DIR="${BASE_DIR}/_launch_logs/${NAME}"
    mkdir -p "$LOG_DIR"
    echo "$CMD" > "${LOG_DIR}/launch_command.txt"
    eval "$CMD" 2>&1 | tee "${LOG_DIR}/run_log.txt"
    break
  fi
  echo "GPU $GPU free ${FREE}MB < ${FREE_TH}MB, wait..."
  sleep 30
done
