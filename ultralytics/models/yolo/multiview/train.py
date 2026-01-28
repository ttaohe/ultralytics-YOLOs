# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.multiview_dataset import MultiviewDataset
from ultralytics.utils import LOGGER, colorstr, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data.augment import Compose, LetterBox, Format, RandomHSV, RandomFlip
import random
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from copy import copy


class VideoSparseSampler(torch.utils.data.Sampler):
    """
    Sparse sampler for multiview video frames.

    For each video segment, pick a random start offset each epoch, then keep frames with fixed stride.
    This operates on scene indices (not per-view indices).
    """

    def __init__(self, dataset, stride=5, random_start=True, shuffle=True, seed=0):
        self.dataset = dataset
        self.stride = max(int(stride), 1)
        self.random_start = bool(random_start)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_views = int(getattr(dataset, "num_views", 1) or 1)
        self._video_items = self._build_index()

    def _parse_video_frame(self, im_file):
        stem = Path(im_file).stem  # e.g., 30-1_00000669
        # Video id = part before '-' (scene id). Fallback to full stem.
        video_id = stem
        frame_id = 0
        try:
            parts = stem.split("-")
            if len(parts) >= 2:
                video_id = parts[0]
            if "_" in stem:
                frame_str = stem.split("_")[-1]
                frame_id = int(frame_str.lstrip("0") or "0")
        except Exception:
            pass
        return video_id, frame_id

    def _build_index(self):
        video_items = {}
        n_scenes = len(self.dataset)
        for scene_idx in range(n_scenes):
            base_idx = scene_idx * self.num_views
            if base_idx >= len(self.dataset.im_files):
                break
            im_file = self.dataset.im_files[base_idx]
            video_id, frame_id = self._parse_video_frame(im_file)
            video_items.setdefault(video_id, []).append((scene_idx, frame_id))
        # Sort each video list by frame id to keep temporal order
        for vid in video_items:
            video_items[vid].sort(key=lambda x: x[1])
        return video_items

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _select_indices(self):
        if self.stride <= 1:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                rng = random.Random(self.seed + self.epoch)
                rng.shuffle(indices)
            return indices

        rng = random.Random(self.seed + self.epoch)
        selected = []
        for vid, items in self._video_items.items():
            offset = rng.randint(0, self.stride - 1) if self.random_start else 0
            for scene_idx, frame_id in items:
                if (frame_id - offset) % self.stride == 0:
                    selected.append(scene_idx)
        if self.shuffle:
            rng.shuffle(selected)
        return selected

    def __iter__(self):
        return iter(self._select_indices())

    def __len__(self):
        return len(self._select_indices())

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

        use_random_crop = getattr(hyp, "random_crop_size", 0) > 0 and getattr(hyp, "random_crop_prob", 0.0) > 0
        if self.augment:
            # For multiview, disable geometric augmentations that break consistency
            # Keep only: HSV color, Flip. Skip LetterBox when random crop is enabled.
            t_list = []
            if not use_random_crop:
                t_list.append(LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False))
            t_list.extend(
                [
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=[]),
                    RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=[]),
                ]
            )
            transforms = Compose(t_list)
        else:
            transforms = Compose([] if use_random_crop else [LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

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

    def progress_string(self):
        """Return a formatted string of training progress with run name prefix."""
        prefix = f"[{self.args.name}] " if getattr(self.args, "name", None) else ""
        return ("\n" + prefix + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

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
            random_crop_size=getattr(self.args, "random_crop_size", 0) if mode == "train" else 0,
            random_crop_prob=getattr(self.args, "random_crop_prob", 0.0) if mode == "train" else 0.0,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        original_imgsz = self.args.imgsz
        if mode == "val" and self.val_imgsz > 0:
            self.args.imgsz = self.val_imgsz
            LOGGER.info(f"Using high-resolution validation with imgsz={self.val_imgsz}")

        try:
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)

            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle:
                LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
                shuffle = False

            sampler = None
            video_stride = int(getattr(self.args, "video_stride", 1) or 1)
            if mode == "train" and video_stride > 1:
                if rank != -1:
                    LOGGER.warning("Video sparse sampling is not supported with DDP; falling back to default sampler.")
                else:
                    sampler = VideoSparseSampler(
                        dataset=dataset,
                        stride=video_stride,
                        random_start=bool(getattr(self.args, "video_rand_start", True)),
                        shuffle=shuffle,
                        seed=int(getattr(self.args, "seed", 0) or 0),
                    )
                    shuffle = False

            from ultralytics.data.build import build_dataloader
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
                sampler=sampler,
            )
        finally:
            self.args.imgsz = original_imgsz

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
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "align_loss"
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

    def save_metrics(self, metrics):
        """Save training metrics to CSV, optionally minimal columns for multiview."""
        if getattr(self.args, "results_minimal", False):
            keep = {
                # overall
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                # per-view
                "metrics/mAP50(B)_view1",
                "metrics/mAP50-95(B)_view1",
                "metrics/mAP50(B)_view2",
                "metrics/mAP50-95(B)_view2",
                # overlap per-view
                "metrics/mAP50(B)_overlap_view1",
                "metrics/mAP50-95(B)_overlap_view1",
                "metrics/mAP50(B)_overlap_view2",
                "metrics/mAP50-95(B)_overlap_view2",
                # align loss
                "train/align_loss",
                "val/align_loss",
            }
            metrics = {k: v for k, v in metrics.items() if k in keep}
            # shorten column names for readability
            rename = {
                "metrics/mAP50(B)": "mAP50_all",
                "metrics/mAP50-95(B)": "mAP50-95_all",
                "metrics/mAP50(B)_view1": "mAP50_v1",
                "metrics/mAP50-95(B)_view1": "mAP50-95_v1",
                "metrics/mAP50(B)_view2": "mAP50_v2",
                "metrics/mAP50-95(B)_view2": "mAP50-95_v2",
                "metrics/mAP50(B)_overlap_view1": "mAP50_ov_v1",
                "metrics/mAP50-95(B)_overlap_view1": "mAP50-95_ov_v1",
                "metrics/mAP50(B)_overlap_view2": "mAP50_ov_v2",
                "metrics/mAP50-95(B)_overlap_view2": "mAP50-95_ov_v2",
                "train/align_loss": "train_align",
                "val/align_loss": "val_align",
            }
            metrics = {rename.get(k, k): v for k, v in metrics.items()}
        return super().save_metrics(metrics)

    def plot_metrics(self):
        """Plot metrics from a CSV file (skip when minimal results enabled)."""
        if getattr(self.args, "results_minimal", False):
            return
        return super().plot_metrics()

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
