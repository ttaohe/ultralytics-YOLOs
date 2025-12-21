
import numpy as np
import torch
import cv2
import random
import copy
from pathlib import Path
from ultralytics.utils.ops import xywhn2xyxy, xyxy2xywhn
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

class VSAVideoDataset(YOLODataset):
    """
    VSA Video Dataset for training long-term memory attention.
    Loads a sequence of T frames (a clip) for each index.
    
    T (time_steps) is determined by the trainer (default 16 or 20).
    Mosaic/Mixup are DISABLED to ensure spatial alignment across the clip.
    Strictly applies the same Random Crop & Resize to all frames in the clip.
    
    Args:
        time_steps: Number of frames to load per clip
        vid_stride: Frame sampling stride (1=consecutive, 5=every 5th frame for ~20% sampling)
        random_crop_size: Size of random crop (0 to disable)
        random_crop_prob: Probability of applying random crop
    """
    def __init__(self, *args, time_steps=16, vid_stride=1, random_crop_size=0, random_crop_prob=1.0, **kwargs):
        # Force disable Mosaic/Mixup/CopyPaste in dataset args if passed
        # This is critical for VSA spatial alignment across frames
        kwargs['augment'] = True 
        
        # IMPORTANT: Save VSA parameters BEFORE calling super().__init__()
        # because parent class YOLODataset may overwrite attributes from hyp/args
        _time_steps = int(time_steps)
        _vid_stride = max(1, int(vid_stride))
        _random_crop_size = int(random_crop_size)
        _random_crop_prob = float(random_crop_prob)
        
        super().__init__(*args, **kwargs)
        
        # Restore VSA parameters AFTER parent init (parent class clobbers them!)
        self.time_steps = _time_steps
        self.vid_stride = _vid_stride
        self.random_crop_size = _random_crop_size
        self.random_crop_prob = _random_crop_prob
        
        # Clear incompatible transforms (just to be safe)
        if hasattr(self, 'transforms') and hasattr(self.transforms, 'transforms'):
             self.transforms.transforms = [
                 t for t in self.transforms.transforms 
                 if type(t).__name__ not in ('Mosaic', 'MixUp', 'CopyPaste', 'CutMix')
             ]
             
        # Parse video indices (VisDrone structure)
        self.video_indices = self._get_video_indices()
        
        # Apply dataset-level sparse sampling (reduce dataset size by vid_stride)
        if self.vid_stride > 1:
            self._filter_dataset_by_stride()
        
        # Debug log
        LOGGER.info(f"VSAVideoDataset initialized: time_steps={self.time_steps}, vid_stride={self.vid_stride}, "
                   f"random_crop_size={self.random_crop_size}, random_crop_prob={self.random_crop_prob}, "
                   f"imgsz={self.imgsz}, total_samples={len(self.im_files)}")
    
    def _get_video_indices(self):
        """Map images to video IDs based on parent folder."""
        video_ids = []
        vid_map = {}
        current_id = 0
        for img_file in self.im_files:
            parent_dir = Path(img_file).parent.name
            if parent_dir not in vid_map:
                vid_map[parent_dir] = current_id
                current_id += 1
            video_ids.append(vid_map[parent_dir])
        return np.array(video_ids)
    
    def _filter_dataset_by_stride(self):
        """
        Create sparse sampling indices to reduce training iterations.
        
        We keep ALL frames in im_files/labels for history loading,
        but only iterate over a subset of "current frames" (every Nth frame per video).
        
        Example: vid_stride=5, dataset=24198 -> iterate ~4840 samples
                 but can still load all 24198 frames for history
        """
        n_total = len(self.im_files)
        sample_indices = []
        
        # Group indices by video
        unique_vids = np.unique(self.video_indices)
        for vid_id in unique_vids:
            # Get indices for this video
            vid_mask = self.video_indices == vid_id
            indices = np.where(vid_mask)[0]
            
            # Select every Nth frame as "current frame" candidates
            # Start from the last frame of each stride group to ensure we have history
            # Example: indices=[0,1,2,3,4,5,6,7,8,9], stride=5 -> [4, 9]
            selected = indices[self.vid_stride - 1::self.vid_stride]
            sample_indices.extend(selected.tolist())
        
        self._sample_indices = np.array(sorted(sample_indices))
        
        LOGGER.info(f"{self.prefix}Sparse Video Sampling enabled: vid_stride={self.vid_stride}. "
                   f"Training on {len(self._sample_indices)} samples out of {n_total} total frames "
                   f"(~{100*len(self._sample_indices)/n_total:.1f}%)")
    
    def __len__(self):
        """Return number of training samples (may be filtered by vid_stride)."""
        if hasattr(self, '_sample_indices'):
            return len(self._sample_indices)
        return len(self.im_files)

    def get_image_and_label(self, index):
        """
        Return the PRIMARY image (last frame of the clip) and label.
        Actual clip loading happens in __getitem__.
        This is kept for compatibility with YOLODataset internals.
        """
        return super().get_image_and_label(index)

    def load_clip(self, idx, length, label_info=None):
        """
        Load a sequence of 'length' frames ending at 'idx' with sparse sampling.
        If history extends beyond video start, pad with the first frame.
        
        With vid_stride=5 and length=8:
            idx=100 → frames [65, 70, 75, 80, 85, 90, 95, 100]
            Covers 35 frames of temporal context with only 8 frames loaded
        
        Returns:
            imgs: List of processed images (BGR, HWC, uint8)
            indices: List of frame indices
            label_info: Updated label dict with transformed bboxes and im_file list
        """
        vid_id = self.video_indices[idx]
        
        # Collect indices backwards with stride (sparse sampling)
        indices = []
        curr = idx
        for _ in range(length):
            if curr < 0 or self.video_indices[curr] != vid_id:
                # Boundary reached, replicate the oldest valid frame
                indices.append(indices[-1] if indices else curr)
            else:
                indices.append(curr)
                curr -= self.vid_stride  # Sparse sampling with stride
        
        indices = indices[::-1]  # Reverse to [t-N*stride, ..., t]
        
        imgs = []
        
        # 0. Global Transform Params (Synced for Clip)
        crop_box = None
        crop_w, crop_h = 0, 0
        
        # Read first image to get original dimensions
        im0_ref = cv2.imread(self.im_files[indices[-1]])
        if im0_ref is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[indices[-1]]}")
        h0, w0 = im0_ref.shape[:2]
        
        if self.random_crop_size > 0 and self.random_crop_prob > 0.0 and random.random() < self.random_crop_prob:
            # Crop logic
            crop_h = min(self.random_crop_size, h0)
            crop_w = min(self.random_crop_size, w0)
            
            # Random offset
            y_off = random.randint(0, h0 - crop_h)
            x_off = random.randint(0, w0 - crop_w)
            
            crop_box = (x_off, y_off, x_off + crop_w, y_off + crop_h)

        # Variables to store transform info for labels
        r = 1.0
        left, top = 0, 0

        # 1. Process Images
        for i in indices:
            f = self.im_files[i]
            im = cv2.imread(f)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")
            
            # Apply Crop (Common to all frames)
            if crop_box:
                x1, y1, x2, y2 = crop_box
                im = im[y1:y2, x1:x2]
            
            # Resize
            h, w = im.shape[:2]
            r = self.imgsz / max(h, w)
            if r != 1:
                im = cv2.resize(im, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square
            h, w = im.shape[:2]
            dh, dw = self.imgsz - h, self.imgsz - w
            top, bottom = dh // 2, dh - (dh // 2)
            left, right = dw // 2, dw - (dw // 2)
            
            if dh > 0 or dw > 0:
                im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

            imgs.append(im)
        
        # 2. Update Labels (Bboxes) - Only for the last frame
        if label_info is not None and 'bboxes' in label_info and len(label_info['bboxes']) > 0:
            # Convert Normalized XYWH to Absolute XYXY
            boxes = xywhn2xyxy(label_info['bboxes'].copy(), w=w0, h=h0)
            
            # Apply Crop
            if crop_box:
                x1, y1, x2, y2 = crop_box
                
                # Shift coordinates
                boxes[:, [0, 2]] -= x1
                boxes[:, [1, 3]] -= y1
                
                # Clip boxes to crop boundaries
                boxes[:, 0] = np.clip(boxes[:, 0], 0, crop_w)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, crop_h)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, crop_w)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, crop_h)
            
            # Apply Resize (Scale)
            boxes *= r
            
            # Apply Pad
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            
            # Filter out invalid boxes (zero area after crop)
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            valid_mask = (box_w > 1) & (box_h > 1)
            
            if valid_mask.sum() > 0:
                boxes = boxes[valid_mask]
                label_info['cls'] = label_info['cls'][valid_mask]
            else:
                # No valid boxes, create empty
                boxes = np.zeros((0, 4), dtype=np.float32)
                label_info['cls'] = np.array([], dtype=np.float32)
            
            # Normalize to Output Space
            boxes = xyxy2xywhn(boxes, w=self.imgsz, h=self.imgsz, clip=True)
            
            # Update label_info
            label_info['bboxes'] = boxes
        
        # 3. Always set im_file as list for all frames (for visualization)
        label_info['im_file'] = [self.im_files[i] for i in indices]

        return imgs, indices, label_info

    def __getitem__(self, index):
        """
        Load a video clip and labels for training.
        
        Returns:
            dict: Contains:
                - 'img': Stacked images tensor [T*3, H, W]
                - 'cls': Class labels tensor
                - 'bboxes': Bounding boxes tensor (normalized xywh)
                - 'batch_idx': Batch indices (all zeros, will be remapped in collate)
                - 'im_file': List of T file paths for visualization
                - 'ori_shape': Original image shape
                - 'resized_shape': Resized image shape
        """
        # Map index to actual frame index if sparse sampling is enabled
        if hasattr(self, '_sample_indices'):
            real_index = self._sample_indices[index]
        else:
            real_index = index
        
        # 1. Load Labels first (for random crop sync)
        label_info = copy.deepcopy(self.labels[real_index])
        
        # 2. Load Raw Clip (Augmented) - label_info is modified in-place
        # Use real_index to load clip starting from the actual frame position
        raw_imgs, indices, label_info = self.load_clip(real_index, self.time_steps, label_info)
        
        # 3. Stack and Format Images
        # Convert to Tensor (T*3, H, W) format
        processed_imgs = []
        for im in raw_imgs:
            # BGR to RGB, HWC to CHW
            im = im[:, :, ::-1].transpose(2, 0, 1) 
            im = np.ascontiguousarray(im)
            processed_imgs.append(im)
            
        stack = np.concatenate(processed_imgs, axis=0)  # (T*3, H, W)
        stack = torch.from_numpy(stack).float() / 255.0
        
        # 4. Get bboxes (already transformed by load_clip)
        bboxes = label_info.get('bboxes', np.zeros((0, 4), dtype=np.float32))
        cls = label_info.get('cls', np.array([], dtype=np.float32))
        
        # Ensure numpy arrays
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32)
        if not isinstance(cls, np.ndarray):
            cls = np.array(cls, dtype=np.float32)
        
        # 5. Build output dict
        data = {
            'img': stack,
            'cls': torch.from_numpy(cls).float(),
            'bboxes': torch.from_numpy(bboxes).float(),
            'ori_shape': label_info.get('ori_shape', (self.imgsz, self.imgsz)),
            'resized_shape': (self.imgsz, self.imgsz),
            'batch_idx': torch.zeros(len(cls)),
            'im_file': label_info['im_file'],  # List of T file paths
        }
        
        return data

