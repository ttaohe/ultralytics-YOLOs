# YOLO12-Video Dataset & Training Improvement Plan

This document outlines future experiments and potential improvements for the video object detection dataset processing logic, inspired by SAM2/SAM3 and current observations.

## 1. Temporal Context & Sampling Strategy (Enhancing Robustness)

**Current Implementation:**
- **Random Stride:** `t - k` where `k ∈ [1, 10]` (Training only).
- **Validation:** Fixed `t - 1`.

**Future Improvements:**
- [ ] **Bi-directional Sampling (Time Reversal):**
  - *Concept:* Allow the model to "look ahead" during training by sampling `memory` from `t + k` with a probability (e.g., 20%).
  - *Benefit:* Helps the model learn appearance consistency regardless of motion direction and improves generalization.
  - *Action:* Add `random_reverse_time_axis` logic in `__getitem__`.
- [ ] **Multi-Frame Memory (Long-term Dependency):**
  - *Concept:* Instead of a single history frame, feed a sequence (e.g., `t-10`, `t-5`, `t-1`).
  - *Benefit:* Handles occlusion better; provides a longer temporal receptive field.
  - *Complexity:* Requires modifying `YOLOMemoryAttention` to accept `Memory = [B, T, C, H, W]` and using 3D Attention or averaging features.
- [ ] **Adaptive Stride based on Motion:**
  - *Concept:* Dynamically adjust stride based on optical flow magnitude (skip more frames in static scenes, fewer in fast scenes).

## 2. Spatial Alignment & Augmentation (Handling Motion)

**Current Implementation:**
- **Homography:** Optional (default `False`).
- **Augmentation:** Synchronized `Fliplr` for Current/History. `LetterBox` ensures consistent padding.

**Future Improvements:**
- [ ] **Explicit Motion Distillation (if Homography is Off):**
  - *Concept:* If we don't use Homography for alignment, we can use it as a supervision signal (auxiliary loss) to teach the Attention map to follow the motion.
- [ ] **Stronger Geometric Augmentation (Synchronized):**
  - *Concept:* Currently, `Mosaic/Mixup` is disabled or limited for video to avoid breaking temporal continuity.
  - *Action:* Implement "Video Mosaic" – apply the *same* random crop/mosaic parameters to both Current and History frames simultaneously. This allows using strong augmentations without breaking the `t` vs `t-k` spatial relationship.
- [ ] **Random "Camera Motion" Simulation:**
  - *Concept:* Artificially synthesize camera movement (zoom in/out, translation) between `t` and `t-k` using affine transforms, forcing the model to learn robustness to ego-motion.

## 3. Data Quality & Efficiency

**Current Implementation:**
- **Loading:** `cv2.imread` from disk for every frame.

**Future Improvements:**
- [ ] **Video Reader Integration (Decord/PyAV):**
  - *Concept:* Read directly from video files (`.mp4`) instead of extracted frames.
  - *Benefit:* Saves massive storage space; aligns with SAM2/3 dataloading.
- [ ] **Keyframe-based Caching:**
  - *Concept:* Cache features of keyframes to speed up training, re-computing features only for "current" frames.

## 4. Loss & Supervision

**Current Implementation:**
- **VideoDetectionLoss:** Filters out History frame predictions; loss calculated only on Current frame.

**Future Improvements:**
- [ ] **Consistency Loss:**
  - *Concept:* Enforce that the features of the object in `Current` frame are similar to the features in `History` frame (after alignment).
- [ ] **Trajectory Supervision:**
  - *Concept:* If track IDs are available in GT, use them to supervise the attention weights (the attention should peak at the location of the *same* object instance in memory).

## 5. Performance Optimization

**Current Implementation:**
- **Flash Attention:** Enabled in `video_block.py`.
- **Memory:** `reset_memory()` called on video boundaries.

**Future Improvements:**
- [ ] **FP16/BF16 Training Stability:** Verify if Flash Attention causes instability in half-precision.
- [ ] **Gradient Accumulation:** If `Batch Size` is constrained by Memory frames, increase `Grad Accumulate` steps.

---
*Last Updated: Nov 28, 2025*

