from copy import copy
import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.video.model import YOLOVideo
from ultralytics.data.video_dataset import VisDroneVideoDataset
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.utils.loss import v8DetectionLoss

class VideoValidator(DetectionValidator):
    """
    Custom Validator for Video Model.
    Ensures dataset returns history frames for validation metrics.
    Also manages memory reset on video boundaries.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_video_id = None

    def build_dataset(self, img_path, mode="val", batch=None):
        return VisDroneVideoDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False, 
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
        )

    def preprocess(self, batch):
        """
        Preprocess batch and handle memory reset.
        Assumes batch_size=1 for safe memory management during validation.
        """
        # Check if we need to reset memory
        # We need access to video indices from the dataset.
        # The dataset is self.data_loader.dataset (if it exists) or passed via batch metadata if we had it.
        # VisDroneVideoDataset loads image paths. We can guess video ID from path parent folder.
        
        if "im_file" in batch:
            # Just check the first image in batch (assuming batch size 1 or consistent batch)
            img_path = batch["im_file"][0]
            # Simple logic: parent folder name determines video ID
            from pathlib import Path
            video_id = Path(img_path).parent.name
            
            if self.last_video_id is not None and video_id != self.last_video_id:
                # Video changed, reset memory
                # Use trainer.model if available, otherwise try to access model from validator if possible
                model_to_reset = getattr(self, "model", None)
                if model_to_reset is None and hasattr(self, "trainer"):
                     model_to_reset = self.trainer.model
                
                if model_to_reset and hasattr(model_to_reset, "reset_memory"):
                    model_to_reset.reset_memory()
                elif model_to_reset and hasattr(model_to_reset, "module") and hasattr(model_to_reset.module, "reset_memory"):
                     # Handle DDP case
                     model_to_reset.module.reset_memory()
            
            self.last_video_id = video_id
        
        return super().preprocess(batch)


class VideoDetectionLoss(v8DetectionLoss):
    """
    Custom Detection Loss for Video Training.
    It filters out the History Frame predictions before calculating loss,
    ensuring that only the Current Frame (which has GT labels) contributes to the loss.
    """
    def __call__(self, preds, batch):
        """
        Calculate loss for video detection.
        
        Args:
            preds: Model predictions. Can be a list of tensors (P3, P4, P5 outputs).
                   Shape of each tensor: [Batch*2, Reg+Cls, H, W] (where Batch*2 = History + Current)
            batch: Batch dictionary containing GT labels.
                   batch['batch_idx'] maps labels to image indices. 
                   In preprocess_batch, we set batch_idx to odd numbers (1, 3, 5...) for Current Frames.
        """
        # 1. Filter Preds: Keep only odd indices (Current Frames)
        # preds[i] shape: [B_total, 4+nc, H, W] -> we want [1::2]
        if isinstance(preds, (list, tuple)):
            # Handle multiple feature scales (e.g., 3 scales for YOLO)
            # preds[0] is usually the list of features for v8DetectionLoss
            # But v8DetectionLoss expects 'preds' to be the output of the model directly.
            # Model output is usually: list[torch.Tensor] (train) or (torch.Tensor, list[torch.Tensor]) (val/deploy)
            # During training, it is list[torch.Tensor].
            
            # However, we need to check if 'preds' is the tuple (loss, feats) structure or just feats.
            # But here we are overriding __call__, which receives the raw model output.
            
            # Slice the batch dimension (dim 0)
            new_preds = [p[1::2].contiguous() for p in preds]
        else:
            # Fallback for single tensor output
            new_preds = preds[1::2].contiguous()
            
        # 2. Filter Batch Indices
        # batch['batch_idx'] contains values like 1, 3, 5...
        # We need to map them to 0, 1, 2... to match the new sliced preds.
        # Create a shallow copy to avoid modifying the original batch permanently
        new_batch = batch.copy()
        
        # Remap batch_idx: 1->0, 3->1, 5->2 ...
        # Formula: new_idx = (old_idx - 1) / 2
        # Note: We should only remap if we are sure about the even/odd structure. 
        # Given preprocess_batch logic, this is consistent.
        new_batch['batch_idx'] = (batch['batch_idx'] - 1) // 2
        
        return super().__call__(new_preds, new_batch)


class SAM2VideoTrainer(DetectionTrainer):
    """
    Trainer tailored for SAM2-style Memory Attention YOLO.
    It constructs batches with explicit time dimension: [Batch * Time, C, H, W].
    """
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOVideo model."""
        model = YOLOVideo(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a customized VideoValidator."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        # Force batch size to 1 for validation to ensure correct video memory handling
        args = copy(self.args)
        args.batch = 2  # For DDP with 2 GPUs, batch size must be multiple of 2
        return VideoValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
    
    def init_criterion(self):
        """Initialize the loss criterion for the VideoTrainer."""
        return VideoDetectionLoss(self.model)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build VisDrone Video Dataset.
        This dataset yields 'img' (current) and 'history_img' (past).
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return VisDroneVideoDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val", 
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: ") if mode == "train" else "",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            use_homography=mode == "train" and getattr(self.args, "use_homography", False),
        )

    def preprocess_batch(self, batch):
        """
        Custom preprocessing to stack current and history images into a time sequence.
        """
        hist_img = batch.get("history_img")
        if hist_img is not None:
            if isinstance(hist_img, (list, tuple)):
                hist_img = torch.stack(hist_img)
            curr_img = batch["img"]  # [B, C, H, W]
            if hist_img.shape != curr_img.shape:
                hist_img = nn.functional.interpolate(
                    hist_img.float(), size=curr_img.shape[-2:], mode="bilinear", align_corners=False
                ).to(curr_img.dtype)
            seq_img = torch.stack([hist_img, curr_img], dim=1).view(-1, *curr_img.shape[1:])
            batch["img"] = seq_img
            batch["batch_idx"] = batch["batch_idx"] * 2 + 1
            
            # Duplicate metadata for plotting
            if "im_file" in batch:
                if "history_im_file" in batch:
                    new_files = []
                    for h, c in zip(batch["history_im_file"], batch["im_file"]):
                        new_files.extend([h, c])
                    batch["im_file"] = new_files
                else:
                    if self.epoch == 0 and batch["batch_idx"][0] == 0: # Only log once
                         from ultralytics.utils import LOGGER
                         LOGGER.warning("WARNING ⚠️ 'history_im_file' not found in batch. Duplicating 'im_file' for visualization, but this might indicate an issue with the Video Dataset.")
                    batch["im_file"] = [x for x in batch["im_file"] for _ in (0, 1)]

            if "ori_shape" in batch:
                batch["ori_shape"] = [x for x in batch["ori_shape"] for _ in (0, 1)]
            if "resized_shape" in batch:
                batch["resized_shape"] = [x for x in batch["resized_shape"] for _ in (0, 1)]

            if hasattr(self.model, "time_steps"):
                self.model.time_steps = 2
            batch.pop("history_img", None)
        return super().preprocess_batch(batch)
