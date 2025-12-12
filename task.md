# Task: Optimize Memory Mechanism for Video Object Detection

## 1. Analysis & Design
- [x] Analyze current global memory implementation (P4 layer)
- [ ] Evaluate computational cost for Training (640sz) vs Inference (1920sz) <!-- id: 0 -->
- [x] Design Top-K Sparse Memory strategy <!-- id: 1 -->
- [ ] **Investigate train_baseline.py performance issues** <!-- id: 9 -->
    - [x] Analyze loss curves (Backbone from scratch)
    - [x] Check RandomCrop implementation (Resize vs Crop) <!-- id: 10 -->
    - [x] **Phase 1: Configure Baseline (Super Mosaic + Pretrained Weights)** <!-- id: 11 -->
    - [x] **Phase 2: Optimize Video (Channel Stacking + Sync Mosaic)** <!-- id: 12 -->
        - [x] Verify `Augment` classes support multi-channel inputs
        - [x] Implement Channel Stacking in `VisDroneVideoDataset`
        - [x] Implement Unstacking in `__getitem__`
        - [x] Enable Super Mosaic Config in `train_video.py`
    - [x] **Phase 3: Optimize Baseline Speed (Baseline Mode)** <!-- id: 13 -->
        - [x] Implement `baseline_mode` in `VisDroneVideoDataset` to skip History IO
        - [x] Register `baseline_mode` in `default.yaml`


## 2. Implementation
- [x] Create `SparseMemoryAttention` or modify `YOLOMemoryAttention` <!-- id: 2 -->
- [x] Implement `select_topk_features` function <!-- id: 3 -->
- [x] Update `forward` loop to store only sparse tokens in Memory Bank <!-- id: 4 -->
- [x] Implement sparse Cross-Attention (Query is dense, Key/Value is sparse) <!-- id: 5 -->

## 3. Verification
- [x] Validate shape compatibility during Training <!-- id: 6 -->
- [ ] Validate speed improvement during Inference (1920sz) <!-- id: 7 -->
- [ ] Check for accuracy regression (sanity check) <!-- id: 8 -->
