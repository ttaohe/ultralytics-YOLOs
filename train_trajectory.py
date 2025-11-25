import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, colorstr
from ultralytics.data.video_dataset import VisDroneVideoDataset

# Fix for "RuntimeError: received 0 items of ancdata" when using multiple workers
# This allows us to use workers > 0 for faster data loading
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class TrajectoryDetectionModel(DetectionModel):
    """
    A wrapper around DetectionModel to handle history features extraction.
    """
    def forward(self, batch, augment=False, embed=None):
        # Unpack extra data from batch
        if isinstance(batch, dict):
            x = batch['img']
            history_img = batch.get('history_img')
            homography = batch.get('homography')
        else:
            x = batch
            history_img = None
            homography = None
        
        # If we are in training/validation with history available
        if history_img is not None and homography is not None:
            # Inject context into the model via property or dictionary
            self.history_context = {
                'img': history_img,
                'homography': homography,
            }
        else:
            self.history_context = None

        # Standard forward
        return super().forward(x, augment=augment, embed=embed)

class TrajectoryValidator(DetectionValidator):
    """
    Custom Validator to handle Trajectory Dataset in Validation.
    """
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build VisDrone Video Dataset for Validation.
        """
        return VisDroneVideoDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False, # No augmentation in validation
            hyp=self.args,
            rect=True, # Rectangular batches for validation
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
        )

class TrajectoryTrainer(DetectionTrainer):
    """
    Custom Trainer that supports Trajectory-Guided Attention.
    """
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build VisDrone Video Dataset.
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

    def get_validator(self):
        """Returns a customized DetectionValidator for Trajectory Model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return TrajectoryValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a customized DetectionModel."""
        model = super().get_model(cfg, weights, verbose)
        # model.extract_history = self._extract_history_features
        return model

    def plot_alignment(self, curr_img, hist_img, homography, batch_idx):
        """
        Visualize alignment: [Current | History | Warped History]
        Saves image to experiment directory.
        
        Args:
            curr_img (Tensor): [C, H, W] normalized
            hist_img (Tensor): [C, H, W] normalized
            homography (Tensor): [3, 3]
        """
        save_dir = self.save_dir / 'alignment_vis'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to Numpy & uint8
        # Assuming image is RGB, normalized 0-1
        def to_numpy(t):
            img = t.detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            # RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(img)

        curr_np = to_numpy(curr_img)
        hist_np = to_numpy(hist_img)
        H_np = homography.detach().cpu().numpy()
        
        h, w = curr_np.shape[:2]
        
        # Warp history to current frame coordinates
        # p_hist = H * p_curr  =>  p_curr = H^-1 * p_hist
        # cv2.warpPerspective(src, M, dsize) uses destination coordinates (x, y)
        # dst(x, y) = src(M^-1 * [x, y, 1]) ??? 
        # No, warpPerspective applies: dst(x,y) = src(M*[x,y,1]) ?
        # Actually, warpPerspective uses INVERSE mapping under the hood if flag is set, 
        # but typically M maps SRC to DST.
        # Here H maps Current(SRC) to History(DST).
        # We want to warp History(SRC) to Current(DST).
        # So we need a matrix M' that maps History -> Current.
        # M' = H^-1.
        
        try:
            H_inv = np.linalg.inv(H_np)
            warped_hist = cv2.warpPerspective(hist_np, H_inv, (w, h))
        except np.linalg.LinAlgError:
            # Singular matrix, fallback
            warped_hist = np.zeros_like(hist_np)

        # Concatenate: Current | History | Warped History
        vis_img = np.hstack([curr_np, hist_np, warped_hist])
        
        # Add text labels
        cv2.putText(vis_img, "Current (t)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, "History (t-1)", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_img, "Warped History -> t", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(str(save_dir / f'batch_{batch_idx}_alignment.jpg'), vis_img)


    def preprocess_batch(self, batch):
        """
        Preprocess batch data: scale images, move to device.
        Also handle history_img and homography.
        """
        # 1. Standard preprocessing for 'img'
        batch = super().preprocess_batch(batch)
        
        # 2. Handle 'history_img'
        if 'history_img' in batch:
            # Fix: batch['history_img'] comes as a tuple from collate_fn, stack it!
            if isinstance(batch['history_img'], (list, tuple)):
                batch['history_img'] = torch.stack(batch['history_img'])
                
            batch['history_img'] = batch['history_img'].to(self.device, non_blocking=True)
        
        # 3. Handle 'homography'
        if 'homography' in batch:
            # Fix: stack homography matrices
            if isinstance(batch['homography'], (list, tuple)):
                batch['homography'] = torch.stack(batch['homography'])
                
            batch['homography'] = batch['homography'].to(self.device, non_blocking=True)
            
        # VISUALIZATION: Check alignment periodically
        # Visualize 1st image of every 100th batch
        if self.epoch == 0 and hasattr(self, 'seen') and (self.seen // self.batch_size) % 100 == 0:
            # Check if we have valid data
            if 'history_img' in batch and 'homography' in batch:
                self.plot_alignment(
                    batch['img'][0], 
                    batch['history_img'][0], 
                    batch['homography'][0], 
                    batch_idx=self.seen // self.batch_size
                )
        
        # 4. Inject context into the model
        if hasattr(self.model, 'model'): # self.model is usually an instance of DetectionModel
            for module in self.model.modules():
                if module.__class__.__name__ == 'TrajectoryBlock':
                    if hasattr(module, 'set_context'):
                        module.set_context(batch.get('history_img'), batch.get('homography'))
                    else:
                        module.history_img = batch.get('history_img')
                        module.homography = batch.get('homography')
        
        return batch

def train():
    # Initialize TrajectoryTrainer
    
    args = dict(
        model='ultralytics/cfg/models/experimental/yolo12-trajectory.yaml',
        data='ultralytics/cfg/datasets/VisDrone-vid.yaml',
        epochs=100,
        imgsz=1280,
        batch=16, # Keep batch small for video training
        project='runs/train-trajectory',
        name='yolo12-traj_exp',
        device='0', # Use GPU 0
        scale=0.0, # Disable scale jitter for video consistency
        workers=8, # Restore workers for speed, rely on sharing_strategy='file_system'
        
        # Disable augmentations that break temporal/spatial alignment
        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
    )
    
    trainer = TrajectoryTrainer(overrides=args)
    trainer.train()

if __name__ == '__main__':
    train()
