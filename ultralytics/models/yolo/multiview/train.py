# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.multiview_dataset import MultiviewDataset
from ultralytics.utils import LOGGER, colorstr, RANK
from ultralytics.data.augment import Compose, LetterBox, Format, RandomHSV, RandomFlip
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from copy import copy

class MultiviewTrainer(DetectionTrainer):
    """
    A custom trainer for YOLO12-Multiview models.
    Overrides build_dataset to use MultiviewDataset.
    For multiview data, disables geometric augmentations (mosaic, mixup, random perspective)
    to maintain geometric consistency across views, while keeping color augmentations.
    """

    def __init__(self, overrides=None):
        """
        Initialize MultiviewTrainer and explicitly disable geometric augmentations.
        """
        # Disable geometric augmentations that break multiview consistency
        # These must be in overrides so they take effect before BaseTrainer.__init__ prints args
        aug_overrides = {
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
        }

        # Merge with user-provided overrides (user overrides take precedence)
        if overrides is None:
            overrides = aug_overrides
        else:
            merged_overrides = {**aug_overrides, **overrides}
            overrides = merged_overrides

        super().__init__(overrides=overrides)

    def build_transforms(self, hyp=None):
        """
        Build transforms for multiview training.
        Disable geometric augmentations (mosaic, mixup, random perspective) to maintain
        geometric consistency across camera views. Keep only color augmentations.
        """
        if hyp is None:
            hyp = self.args

        if self.augment:
            # For multiview, disable geometric augmentations that break consistency
            # Keep only: LetterBox resize, HSV color, Flip (same seed applied across views)
            transforms = Compose([
                LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=[]),
                RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=[]),
            ])
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build MultiviewDataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' or 'val'.
            batch (int, optional): Batch size.

        Returns:
            MultiviewDataset: Custom dataset instance.
        """
        # Get num_views from model args or data config, default to 2
        num_views = getattr(self.args, 'num_views', None) or self.data.get('num_views', 2)

        # Get camera_dirs from data config for separate mode
        camera_dirs = self.data.get('camera_dirs', None)

        # Get PE cache size from data config, default to 1000 (~5-10MB)
        pe_cache_size = self.data.get('pe_cache_size', 1000)
        pe_mmap = self.data.get('pe_mmap', False)
        pe_lmdb = self.data.get('pe_lmdb', None)
        pe_disable = self.data.get('pe_disable', False)

        LOGGER.info(colorstr(f"Building MultiviewDataset with num_views={num_views} for {mode}..."))
        if camera_dirs:
            LOGGER.info(colorstr(f"Separate camera mode: {camera_dirs}"))
        LOGGER.info(colorstr(f"Geometric augmentations disabled for multiview consistency"))

        return MultiviewDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            num_views=num_views,
            camera_dirs=camera_dirs,
            pe_cache_size=pe_cache_size,
            pe_mmap=pe_mmap,
            pe_lmdb=pe_lmdb,
            pe_disable=pe_disable,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader.
        """
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLOMultiview model.
        """
        from ultralytics.models.yolo.multiview.model import YOLOMultiview

        # Get channels from data config, default to 3 (RGB)
        ch = self.data.get("channels", 3)

        model = YOLOMultiview(cfg, nc=self.data["nc"], ch=ch, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """
        Return a MultiviewValidator for YOLO Multiview model validation.
        Computes metrics separately for each camera view.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        from ultralytics.models.yolo.multiview.val import MultiviewValidator

        # Copy args and remove custom attributes that aren't valid YOLO args
        args = copy(self.args)
        # Store num_views separately before removing invalid args
        num_views = getattr(self.args, 'num_views', None) or self.data.get('num_views', 2)

        # Remove custom args that aren't valid YOLO arguments
        for attr in ['num_views', 'random_crop_size', 'random_crop_prob']:
            if hasattr(args, attr):
                delattr(args, attr)

        # High-res validation support
        if hasattr(self, 'val_imgsz') and self.val_imgsz > 0:
            args.imgsz = self.val_imgsz

        # Create validator and set num_views directly
        validator = MultiviewValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
        validator.num_views = num_views
        return validator

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their annotations.
        Additionally generates position encoding visualization for multiview data.

        Args:
            batch (dict): Dictionary containing batch data with 'coords' key for PE.
            ni (int): Number of iterations.
        """
        # First, call parent method to generate standard train_batch.jpg (for label checking)
        super().plot_training_samples(batch, ni)

        # Then, generate PE visualization
        self._plot_position_encodings(batch, ni)

    def _plot_position_encodings(self, batch, ni):
        """
        Visualize position encodings overlaid on images.
        Generates a separate image file with PE visualization.

        Args:
            batch (dict): Batch data containing 'img', 'coords', 'im_file', etc.
            ni (int): Number of iterations.
        """
        if 'coords' not in batch:
            LOGGER.warning(f"{colorstr('yellow')}No 'coords' in batch, skipping PE visualization")
            return

        images = batch['img']  # (N*V, C, H, W)
        coords = batch['coords']  # (N, V, H, W, 4)
        im_files = batch['im_file']  # (N*V)

        # Get num_views from first sample
        num_views = coords.shape[1] if coords.ndim == 5 else 2

        # Convert to numpy if tensor
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().float().numpy()

        bs, _, h, w = images.shape  # Total images = batch_size * num_views

        # Limit number of images to plot
        max_subplots = 16
        bs = min(bs, max_subplots)
        ns = int(np.ceil(bs**0.5))

        # De-normalize images if needed
        if np.max(images[0]) <= 1:
            images = (images * 255).astype(np.uint8)

        # Build mosaic grid
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

        # Reshape coords to match image layout: (N*V, H, W, 4)
        # Original coords: (N, V, H, W, 4) -> flatten to (N*V, H, W, 4)
        coords_flat = coords.reshape(-1, h, w, 4)

        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))

            # Get image
            img = images[i].transpose(1, 2, 0)  # (H, W, 3)

            # Get position encoding for this image
            pe = coords_flat[i]  # (H, W, 4)

            # Normalize PE to [0, 255] for visualization
            # PE channels: sin_y, cos_y, sin_x, cos_x (each in [-1, 1])
            pe_norm = ((pe + 1) / 2 * 255).astype(np.uint8)

            # Create RGB PE visualization to match offline visualization
            # R: sin_y, G: cos_y, B: sin_x
            pe_vis = np.stack([
                pe_norm[:, :, 0],  # sin_y -> R
                pe_norm[:, :, 1],  # cos_y -> G
                pe_norm[:, :, 2],  # sin_x -> B
            ], axis=2)

            # Resize PE to match image size if needed
            if pe_vis.shape[:2] != img.shape[:2]:
                pe_vis = cv2.resize(pe_vis, (img.shape[1], img.shape[0]))

            # Blend: 60% original image, 40% PE overlay
            blended = cv2.addWeighted(img, 0.6, pe_vis, 0.4, 0)

            # Place in mosaic
            mosaic[y:y+h, x:x+w, :] = blended

            # Add filename text
            if i < len(im_files):
                filename = Path(im_files[i]).name[:30]
                cv2.putText(mosaic, filename, (x + 5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Add border
            cv2.rectangle(mosaic, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)

            # Add view indicator
            view_idx = i % num_views
            view_text = f"View {view_idx}"
            cv2.putText(mosaic, view_text, (x + 5, y + h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Add title
        title = f"Position Encoding Visualization - Batch {ni}"
        cv2.putText(mosaic, title, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save PE visualization
        pe_fname = self.save_dir / f"train_batch{ni}_pe.jpg"
        cv2.imwrite(str(pe_fname), mosaic)

        if self.args.plots:
            LOGGER.info(f"{colorstr('bold')}Saved PE visualization to {pe_fname.name}")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch. Log PE cache statistics."""
        super().on_train_epoch_end()

        # Log PE cache statistics if available
        if hasattr(self.train_loader.dataset, 'log_cache_stats'):
            self.train_loader.dataset.log_cache_stats()
