# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.multiview_dataset import MultiviewDataset
from ultralytics.utils import LOGGER, colorstr
import os
import torch

class MultiviewTrainer(DetectionTrainer):
    """
    A custom trainer for YOLO12-Multiview models.
    Overrides build_dataset to use MultiviewDataset.
    """
    
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
        # Get num_views from model args or config, default to 1 if not specified
        # In a real scenario, this might come from data.yaml or model cfg
        # For now, we assume it's passed via overrides or defaults to 3 for testing
        num_views = self.args.num_views if hasattr(self.args, 'num_views') else 3
        
        LOGGER.info(colorstr(f"Building MultiviewDataset with num_views={num_views} for {mode}..."))
        
        return MultiviewDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train", 
            hyp=self.args,
            rect=False, # Rectangular training might be tricky with multiview
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            num_views=num_views
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader.
        We override this to ensure our custom collate_fn is used (though build_dataloader does check getattr).
        """
        # The base implementation already uses getattr(dataset, "collate_fn", None)
        # So we just call super().get_dataloader
        # But we need to ensure batch_size is interpreted correctly.
        # In MultiviewDataset, 1 sample = V images.
        # So batch_size=16 means 16 scenes (16*V images).
        # This might be too large for GPU memory.
        # Users should adjust batch size accordingly.
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLOMultiview model.
        """
        from ultralytics.models.yolo.multiview.model import YOLOMultiview
        
        model = YOLOMultiview(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
