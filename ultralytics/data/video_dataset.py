import cv2
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
    def __init__(self, *args, **kwargs):
        self._hyp = kwargs.get("hyp")
        super().__init__(*args, **kwargs)
        # Parse video sequences from image paths
        self.video_indices = self._get_video_indices()
        # ORB detector for motion estimation
        self.orb = cv2.ORB_create(nfeatures=500)
        self._check_video_augmentations()

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
                LOGGER.info(f"{self.prefix}  - Video {vid_name}: {count} frames")
        
        return np.array(video_ids)

    def load_image(self, i, rect_mode=True):
        """
        Overridden to load current image.
        Returns: (img, (h0, w0), (h, w))
        """
        return super().load_image(i, rect_mode=rect_mode)

    def load_history(self, i, current_img):
        """
        Load history image (t-1) and compute Homography.
        """
        # Check if index i-1 exists and belongs to the same video
        if i > 0 and self.video_indices[i] == self.video_indices[i-1]:
            prev_i = i - 1
            # Load raw previous image (without augmentation for now to compute H reliably)
            # Note: ideally we compute H on raw images, then apply same aug to H.
            # For simplicity here, we re-load raw previous image.
            try:
                prev_img, _, _ = super().load_image(prev_i, rect_mode=False)
            except Exception:
                prev_img = current_img.copy()
        else:
            # First frame of sequence, use current frame as history
            prev_img = current_img.copy()
            
        return prev_img

    def compute_homography(self, img1, img2):
        """
        Compute Homography matrix H that maps img2 to img1 (H_{t-1 -> t}).
        Note: The model needs H_{t -> t-1}, which is inverse of this if we map p_t to p_{t-1}.
        Let's stick to definition: p_{t-1} = H * p_t
        """
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
        # 1. Standard YOLO Dataset load (Current Frame)
        data = super().__getitem__(index)
        
        # Load history frame from previous index if available
        img_curr, _, _ = self.load_image(index, rect_mode=False)
        img_hist = self.load_history(index, img_curr)
        
        homography = self.compute_homography(img_curr, img_hist)
        ratio_pad = data.get('ratio_pad')
        target_size = tuple(int(dim) for dim in data['img'].shape[1:])  # (H, W)
        img_hist_tensor = self._prepare_history_tensor(img_hist, ratio_pad, target_size)
        data['history_img'] = img_hist_tensor
        data['homography'] = homography
        
        return data

    def _prepare_history_tensor(
        self,
        img_hist: np.ndarray,
        ratio_pad: tuple | None,
        target_size: tuple[int, int],
        pad_value: int = 114,
    ) -> torch.Tensor:
        """
        Apply the current frame's letterbox ratio/pad to the history frame to ensure perfect spatial alignment.

        Args:
            img_hist (np.ndarray): Raw history frame (BGR).
            ratio_pad (tuple | None): ((ratio_h, ratio_w), (pad_w, pad_h)) from current frame.
            target_size (tuple[int, int]): Final (H, W) after letterbox for the current frame.
            pad_value (int): Constant value used when padding.
        """
        if img_hist is None:
            raise ValueError("History image cannot be None.")

        target_h, target_w = target_size
        if ratio_pad is not None:
            ratio, pad = ratio_pad
            ratio = ratio.tolist() if hasattr(ratio, "tolist") else ratio
            pad = pad.tolist() if hasattr(pad, "tolist") else pad
            ratio_h, ratio_w = (ratio if isinstance(ratio, (list, tuple)) else (ratio, ratio))
            pad_w, pad_h = (pad if isinstance(pad, (list, tuple)) else (0.0, 0.0))

            new_w = max(int(round(img_hist.shape[1] * ratio_w)), 1)
            new_h = max(int(round(img_hist.shape[0] * ratio_h)), 1)
            resized = cv2.resize(img_hist, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((target_h, target_w, resized.shape[2]), pad_value, dtype=resized.dtype)
            left = int(round(pad_w))
            top = int(round(pad_h))
            right = min(left + new_w, target_w)
            bottom = min(top + new_h, target_h)
            canvas[top:bottom, left:right] = resized[: bottom - top, : right - left]
        else:
            # Fallback to standard letterbox if ratio/pad information is missing.
            letterbox = LetterBox(
                new_shape=(target_h, target_w),
                auto=self.rect,
                scaleup=self.augment,
                stride=self.stride,
                center=True,
            )
            canvas = letterbox(image=img_hist)
            if canvas.ndim == 2:
                canvas = np.expand_dims(canvas, axis=-1)

        img_hist_tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy())
        return img_hist_tensor

    def _check_video_augmentations(self):
        """Strong geometric augmentations break temporal alignment, so disallow them."""
        if not self.augment or self._hyp is None:
            return
        risky = ("degrees", "translate", "scale", "shear", "perspective")
        violations = {}
        for key in risky:
            value = getattr(self._hyp, key, 0.0)
            if abs(float(value)) > 1e-6:
                violations[key] = value
        if violations:
            raise ValueError(
                "VisDroneVideoDataset does not support strong geometric augmentations for video training. "
                f"Please set the following hyper-parameters to 0: {violations}"
            )
