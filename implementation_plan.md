# Implementation Plan - Optimized VisDrone Training

## Goal
Improve VisDrone training performance by fixing "Cold Start" issues, optimizing data quality (High-Res Mosaic), and implementing an efficient Sparse Memory mechanism for Video models.

## User Review Required
> [!IMPORTANT]
> **Data Strategy Change**: We will set `random_crop_prob=1.0` when `mosaic` is enabled. 
> This ensures that Mosaic is built **exclusively** from high-resolution crops (e.g. 640x640 crops from 2000x1500 images), rather than a mix of crops and resized whole images. This maximizes small object detail.

## Proposed Changes

### 1. Fix Baseline Training (`train_baseline.py`)
- [x] **Load Pretrained Weights**: Change `model = YOLO(...)` to load `yolo12n.pt` weights.
- [x] **Strategy: Super-Mosaic**: Set `random_crop_prob=1.0` and `mosaic=1.0`. 
    - *Why*: Image model handles mixed-context (Mosaic) well. This maximizes small object exposure.
- [x] **Baseline Mode Optimization**: Implemented `baseline_mode` to skip history IO for 2x-4x speedup.

### 2. Optimize Video Training (`train_video.py`)
- [x] **Strategy: Super-Mosaic (Sync-Stacking)**: Enable `mosaic=1.0` and `random_crop_prob=1.0`.
    - *Implementation*: Modify `VisDroneVideoDataset`.
        1. **Stacking**: In `get_image_and_label`, return 6-channel `img` (Current + History).
        2. **Pipeline**: `v8_transforms` runs geometric aug (Mosaic etc.) on 6-channel image.
        3. **Safe Unstacking**: In `__getitem__`, **before** the final `Format` (normalization), split 6-channel tensor back to two 3-channel tensors. 
        4. **Manual Formatting**: Manually apply `Format` (ToTensor + Normalize) to both Current and History images separately. This avoids applying RGB stats to History channels effectively.

### 3. Sparse Memory Attention (`ultralytics/models/sam/modules/memory_attention.py`)
- [x] **Top-K Feature Selection**: Implement `select_topk_features` to filter only the most salient features from the P3 layer.
- [x] **Sparse Cross Attention**: Modify `YOLOMemoryAttention` to handle sparse memory tokens (Query=Dense, Key/Value=Sparse).
- [x] **Layer Adjustment**: Move Memory Attention from P4 (Stride 16) to P3 (Stride 8) in `yolo12-video-sparse-p3.yaml`.

## Verification Plan

### Automated Tests
- [x] **Shape Check**: Run `train_video.py` for 1 epoch (or small batch) to verify Memory Attention shapes and Top-K logic (no OOM).
- [ ] **Resolution Check**: Use `debug_val_resolution.py` to confirm validation is running at high resolution (1920) and crops are working as expected.

### Manual Verification
- **Loss Curve**: Observe `box_loss` and `dfl_loss` in the first 5 epochs of `train_baseline.py`. It should drop significantly faster with pretraining.
- **Mosaic Visualization**: Inspect `train_batch*.jpg` in `runs/train-baseline` to visually confirm that the mosaic contains high-res crops (objects look big/clear) rather than resized whole images (objects look tiny).
