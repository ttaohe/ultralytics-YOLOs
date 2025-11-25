import warnings
import torch
import torch.nn as nn
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.video import YOLOVideo
from ultralytics.data.video_dataset import VisDroneVideoDataset
from ultralytics.utils import LOGGER, colorstr, RANK, DEFAULT_CFG

warnings.filterwarnings("ignore")

class VideoValidator(DetectionValidator):
    """
    Custom Validator for Video Model.
    Ensures dataset returns history frames for validation metrics.
    """
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
        return VideoValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build VisDrone Video Dataset.
        This dataset yields 'img' (current) and 'history_img' (past).
        """
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
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
                batch["im_file"] = [x for x in batch["im_file"] for _ in (0, 1)]
            if "ori_shape" in batch:
                batch["ori_shape"] = [x for x in batch["ori_shape"] for _ in (0, 1)]
            if "resized_shape" in batch:
                batch["resized_shape"] = [x for x in batch["resized_shape"] for _ in (0, 1)]

            if hasattr(self.model, "time_steps"):
                self.model.time_steps = 2
            batch.pop("history_img", None)
        return super().preprocess_batch(batch)

def train():
    args = dict(
        model='ultralytics/cfg/models/12/yolo12-video.yaml', 
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',   
        epochs=100,
        imgsz=800, 
        batch=8,   
        project='runs/train-video',
        name='yolo12-sam2-video',
        device='0',
        
        mosaic=0.0,
        mixup=0.0,
        scale=0.0,
        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0, 
    )
    
    trainer = SAM2VideoTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()

