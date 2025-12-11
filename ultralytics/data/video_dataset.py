import cv2
import math
import numpy as np
import torch
from pathlib import Path

from ultralytics.data.augment import LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

class VisDroneVideoDataset(YOLODataset):
    """
    Dataset class for VisDrone-VID (Video Object Detection) with History Support.
    
    This dataset loads pairs of (current_frame, history_frame) and computes 
    the Homography matrix between them to support Trajectory-Guided Attention.
    """
    def __init__(
        self,
        *args,
        use_homography: bool = False,
        random_crop_size: int = 0,
        random_crop_prob: float = 1.0,
        **kwargs,
    ):
        self._hyp = kwargs.get("hyp")
        self.use_homography = use_homography
        # Random window crop for high-res VisDrone frames (applied before transforms)
        self.random_crop_size = int(random_crop_size or 0)
        self.random_crop_prob = float(random_crop_prob or 0.0)
        
        # Save original flip probabilities and disable them for super()
        # We will manually handle flip in __getitem__ to ensure synchronization
        if self._hyp:
            self.fliplr = getattr(self._hyp, "fliplr", 0.0)
            self.flipud = getattr(self._hyp, "flipud", 0.0)
            self._hyp.fliplr = 0.0
            self._hyp.flipud = 0.0
        else:
            self.fliplr = 0.0
            self.flipud = 0.0
            
        # Pass random crop params to super class (YOLODataset) to avoid overwrite
        kwargs["random_crop_size"] = self.random_crop_size
        kwargs["random_crop_prob"] = self.random_crop_prob
        
        # Disable incompatible augmentations BEFORE calling super().__init__ (which builds transforms)
        # We need to set self.augment manually because super().__init__ hasn't run yet
        self.augment = kwargs.get("augment", True)
        self.prefix = kwargs.get("prefix", "")
        self.prefix = kwargs.get("prefix", "")
        # self._check_video_augmentations() # Removed: We now support augmentations via Channel Stacking
        
        if self.augment:
            LOGGER.info(f"{self.prefix}Random Crop Config: size={self.random_crop_size}, prob={self.random_crop_prob}")

        super().__init__(*args, **kwargs)
        
        # 3. No longer filtering incompatible transforms because we handle them via Channel Stacking
        # if hasattr(self, 'transforms') and hasattr(self.transforms, 'transforms'):
        #     original_len = len(self.transforms.transforms)
        #     self.transforms.transforms = [
        #         t for t in self.transforms.transforms 
        #         if type(t).__name__ not in ('Mosaic', 'MixUp', 'CopyPaste', 'CutMix')
        #     ]
        #     if len(self.transforms.transforms) < original_len:
        #         LOGGER.info(f"{self.prefix}Removed {original_len - len(self.transforms.transforms)} incompatible transforms (Mosaic/MixUp/CutMix/CopyPaste) from pipeline.")
        # Parse video sequences from image paths
        self.video_indices = self._get_video_indices()
        # ORB detector for motion estimation
        if self.use_homography:
            self.orb = cv2.ORB_create(nfeatures=500)
        else:
            self.orb = None

    # ---------------------------- Random window crop helpers ---------------------------- #

    def get_image_and_label(self, index: int) -> dict:
        """
        Override BaseDataset.get_image_and_label to return 6-channel Stacked Image.
        Stack: [Current(3) + History(3)] -> (H, W, 6)
        
        This enables 'Mosaic', 'MixUp', and 'RandomPerspective' to process
        temporal pairs synchronously as a single 'image'.
        """
        # 1. Load Current Label & Image (Standard YOLO logic)
        label = super().get_image_and_label(index)
        
        # FIX: Ensure index is in buffer (required for Mosaic because super() crop path bypasses load_image)
        if self.augment:
            self.buffer.append(index)
            # Maintain buffer size
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        
        # 2. Load History Image
        # Note: 'img' in label is already loaded (and potentially cropped if random_crop_size>0)
        # But super().get_image_and_label might have resized it? 
        # Actually in YOLODataset, 'img' is loaded via load_image.
        # Let's inspect label dictionary.
        
        current_img = label['img'] # (H, W, 3)
        h, w = current_img.shape[:2]
        
        # Load History (Strictly same shape, potentially synced crop/resize)
        img_hist, hist_file = self.load_history(index, current_img, label=label)
        
        # 3. Stack Channels -> (H, W, 6)
        stacked_img = np.concatenate((current_img, img_hist), axis=2)
        
        # 4. Update label
        label['img'] = stacked_img
        label['history_im_file'] = hist_file
        
        # Note: We rely on Albumentations/RandomHSV to skip this 6-channel image
        # and on Mosaic/RandomPerspective to handle it correctly.
        
        return label


    def _get_video_indices(self):
        """
        Group images by video sequence based on parent directory name.
        Returns a list where list[i] = video_id of image i.
        Also logs video statistics.
        """
        video_ids = []
        vid_map = {}
        current_id = 0
        video_counts = {}
        
        for img_file in self.im_files:
            parent_dir = Path(img_file).parent.name
            if parent_dir not in vid_map:
                vid_map[parent_dir] = current_id
                current_id += 1
                video_counts[parent_dir] = 0
            video_ids.append(vid_map[parent_dir])
            video_counts[parent_dir] += 1
        
        # Log video statistics
        if len(vid_map) > 0:
            LOGGER.info(f"{self.prefix}Video Dataset Statistics: Found {len(vid_map)} videos.")
            # Sort by video name for cleaner output
            for vid_name in sorted(vid_map.keys()):
                count = video_counts[vid_name]
                LOGGER.debug(f"{self.prefix}  - Video {vid_name}: {count} frames")
        
        return np.array(video_ids)

    def load_image(self, i, rect_mode=True):
        """
        Overridden to load current image.
        Returns: (img, (h0, w0), (h, w))
        
        Note: Direct cropping from original image is handled in get_image_and_label,
        so this method uses standard resize logic.
        """
        return super().load_image(i, rect_mode=rect_mode)

    def load_history(self, i, current_img, label=None):
        """
        Load history image (t-k) and compute Homography.
        Now supports random temporal stride for training.
        
        Args:
            i (int): Current index.
            current_img (np.array): Current image (already processed by super().get_image_and_label).
            label (dict, optional): Label dict from get_image_and_label containing sync metadata.
        
        Returns:
             (prev_img, hist_file)
        """
        from ultralytics.utils.patches import imread
        import random
        
        hist_file = self.im_files[i]
        curr_h, curr_w = current_img.shape[:2]
        
        # Random Stride Logic (SAM3-style)
        stride_min = 1
        stride_max = 10 if self.augment else 1
        # Use np.random.randint for consistency
        stride = np.random.randint(stride_min, stride_max + 1)

        # Check if index i-stride exists and belongs to the same video
        # Determine valid history index
        prev_i = i
        if i >= stride and self.video_indices[i] == self.video_indices[i-stride]:
            prev_i = i - stride
        elif i > 0 and self.video_indices[i] == self.video_indices[i-1]:
            # Fallback to t-1
            prev_i = i - 1
        else:
            # Self-reference (First frame)
            prev_img = current_img.copy()
            return prev_img, self.im_files[i]

        hist_file = self.im_files[prev_i]
        
        # Load history image from disk (no resize, no cache)
        prev_img = imread(hist_file, flags=self.cv2_flag)
        if prev_img is None:
            raise FileNotFoundError(f"History image not found: {hist_file}")
        
        # Sync Shape: Match current_img via Crop or Resize
        prev_h, prev_w = prev_img.shape[:2]
        
        # 1. Check for Sync Crop
        crop_ori = label.get("crop_window_ori") if label else None
        if crop_ori and len(crop_ori) == 4 and sum(crop_ori) > 0:
             x0, y0, x1, y1 = crop_ori
             # Verify valid window
             if x1 > x0 and y1 > y0:
                 # Ensure crop is within bounds (history might be slightly different size? No, should be compliant)
                 # Apply SAME crop to History (Strict Sync)
                 prev_img = prev_img[y0:y1, x0:x1]
                 prev_h, prev_w = prev_img.shape[:2]

        # 2. Check for Resize Mismatch (Fallback)
        if prev_h != curr_h or prev_w != curr_w:
             # Resize history to match current (Standard YOLODataset behavior compatibility)
             prev_img = cv2.resize(prev_img, (curr_w, curr_h), interpolation=cv2.INTER_LINEAR)
             prev_h, prev_w = prev_img.shape[:2]

        # 3. Final Strict Check
        if prev_h != curr_h or prev_w != curr_w:
            raise ValueError(
                f"History frame shape mismatch! Current: ({curr_h}, {curr_w}), "
                f"History: ({prev_h}, {prev_w}). File: {hist_file}. "
                "Unable to sync shapes via Crop or Resize."
            )
            
        return prev_img, hist_file

    def compute_homography(self, img1, img2):
        """
        Compute Homography matrix H that maps img2 to img1 (H_{t-1 -> t}).
        Note: The model needs H_{t -> t-1}, which is inverse of this if we map p_t to p_{t-1}.
        Let's stick to definition: p_{t-1} = H * p_t
        """
        if not self.use_homography:
            return torch.eye(3)

        # Resize for faster computation if images are too large
        scale = 0.5
        h, w = img1.shape[:2]
        s_img1 = cv2.resize(img1, (int(w*scale), int(h*scale)))
        s_img2 = cv2.resize(img2, (int(w*scale), int(h*scale)))

        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(s_img1, None)
        kp2, des2 = self.orb.detectAndCompute(s_img2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return torch.eye(3)

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 10:
            return torch.eye(3)

        # Sort matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute Homography: maps src(t) to dst(t-1)
        # We want p_{t-1} approx H * p_t
        # So dst_pts (t-1) = H * src_pts (t)
        # Wait, kp1 is from img1 (current t), kp2 is from img2 (history t-1) ? 
        # Argument order in detectAndCompute matters.
        # Let's assume img1=Current(t), img2=History(t-1).
        
        # ORB matches img1 features to img2 features.
        # queryIdx -> img1 (Current)
        # trainIdx -> img2 (History)
        
        # We want matrix M such that: p_history = M * p_current
        # So dst points should be history (img2), src points should be current (img1)
        pts_current = src_pts / scale
        pts_history = dst_pts / scale
        
        M, mask = cv2.findHomography(pts_current, pts_history, cv2.RANSAC, 5.0)
        
        if M is None:
            return torch.eye(3)
            
        return torch.from_numpy(M).float()

    def __getitem__(self, index):
        """
        Returns dictionary with 'img', 'labels', ... and NEW keys:
        'history_img': Previous frame tensor
        'homography': 3x3 Motion Matrix
        """
        # 1. Standard YOLO Dataset load (Current Frame) with all image transforms
        # Since we modified get_image_and_label to return 6-CH stack,
        # data['img'] here will be a 6-CH Tensor (augmented synchronously)
        data = super().__getitem__(index)
        
        # 2. Unstack 6-channel Tensor -> Current (3-ch) + History (3-ch)
        img_stack = data['img'] # Tensor [6, H, W]
        
        # Safety check for Channel Stacking
        if img_stack.shape[0] == 6:
            # Split stack
            img_curr = img_stack[:3]    # [3, H, W]
            img_hist = img_stack[3:]    # [3, H, W]

            # Update data dictionary
            data['img'] = img_curr
            data['history_img'] = img_hist
            
            # 3. Compute Homography on Augmented Images
            # We need to convert back to numpy (HWC uint8) for ORB feature matching
            # Note: Format transform produces Float Tensor (div 255?) -> Check augment.py
            # Verified: Format just does ToTensor (0-255 uint8 to Float Tensor but NOT div 255 if normalize=False)
            # Actually Format leaves it as encoded. But usually ToTensor scales to [0.0, 1.0] only if pixel value > 255?
            # torch.from_numpy preserves type. If input was uint8, tensor is uint8.
            # Let's assume uint8 for now, but handle float just in case.
            
            def to_numpy_img(tensor):
                # [C, H, W] -> [H, W, C]
                arr = tensor.permute(1, 2, 0).numpy()
                if arr.dtype == np.float32 and arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
                return np.ascontiguousarray(arr)

            curr_np = to_numpy_img(img_curr)
            hist_np = to_numpy_img(img_hist)
            
            # Compute Homography on the final augmented frames ensures perfect alignment with visual features
            homography = self.compute_homography(curr_np, hist_np)
            data['homography'] = homography
            
            # History file path (was set in get_image_and_label)
            # data has 'history_im_file' key from get_image_and_label
            if 'history_im_file' not in data: 
                 # Fallback if lost (e.g. Mosaic recreation might lose custom keys?)
                 # Mosaic recycles labels dict, so keys should be preserved or mixed.
                 # If mixed, it might be from a different image, but that's what we want for Mosaic history.
                 # Wait, Mosaic uses 'mix_labels' which contains 'history_im_file' loops?
                 # Mosaic output 'data' is the 'final_labels'. 
                 # We need to trust Mosaic to carry over keys or we manually fix it?
                 # Mosaic preserves custom keys in `final_labels` from the main image (index 0).
                 pass

        else:
            # Fallback for unexpected case (e.g. validation set might not stack?)
            # Or if Mosaic failed to stack.
            # For now assume Stacking always works if get_image_and_label is called.
            # But if config random_crop=0, get_image_and_label still stacks.
            LOGGER.warning(f"Unexpected image shape {img_stack.shape} in Video Dataset. Expected 6 channels.")
            
        # Ensure auxiliary keys exist
        if "crop_window_resized" not in data:
            data["crop_window_resized"] = (0, 0, 0, 0)
        if "crop_window_ori" not in data:
            data["crop_window_ori"] = (0, 0, 0, 0)

        return data


                
