
from copy import copy
import torch
import torch.nn as nn
from ultralytics.models.yolo.video.train import SAM2VideoTrainer
import math
import time
from copy import copy
import torch
import torch.nn as nn
from ultralytics.models.yolo.video.train import SAM2VideoTrainer
from datetime import timedelta
from ultralytics.data.vsa_dataset import VSAVideoDataset
from ultralytics.utils import colorstr, LOGGER, RANK, LOCAL_RANK, callbacks
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, attempt_compile, TORCH_2_4
from ultralytics.utils.checks import check_imgsz, check_amp
import torch.distributed as dist
from ultralytics.cfg import DEFAULT_CFG

class VSAVideoTrainer(SAM2VideoTrainer):
    """
    Trainer for VSA (Video Sparse Attention) models.
    Supports loading long video clips (e.g., T=20) via VSAVideoDataset.
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize VSAVideoTrainer, handling custom VSA arguments.
        
        Extra args (extracted before passing to parent):
            time_steps: Number of frames per clip (default 20)
            vid_stride: Frame sampling stride for sparse sampling (default 1, use 5 for ~20%)
        """
        if overrides is None:
            overrides = {}
            
        # Extract VSA-specific args ONLY (leave random_crop_* for parent to handle)
        self.time_steps = int(overrides.pop('time_steps', 20))
        # Note: vid_stride is also popped by parent, but we need our own copy
        vsa_vid_stride = int(overrides.get('vid_stride', 1))  # Get but don't pop yet
        
        super().__init__(cfg, overrides, _callbacks)
        
        # Override vid_stride with our VSA value (parent may have used default)
        self.vid_stride = vsa_vid_stride
        
        # Store back in args for pickling/DDP
        self.args.time_steps = self.time_steps
        self.args.vid_stride = self.vid_stride
        
        LOGGER.info(f"VSAVideoTrainer initialized: time_steps={self.time_steps}, vid_stride={self.vid_stride}, "
                   f"random_crop_size={self.random_crop_size}, random_crop_prob={self.random_crop_prob}")
    
    def get_validator(self):
        """
        Returns a customized VideoValidator.
        Override to remove VSA-specific args before validation to avoid SyntaxError.
        """
        # Temporarily remove VSA-specific args from self.args
        saved_args = {}
        for attr in ['time_steps', 'vid_stride']:
            if hasattr(self.args, attr):
                saved_args[attr] = getattr(self.args, attr)
                delattr(self.args, attr)
            
        try:
            # Call parent get_validator which handles other cleanups (random_crop, etc.)
            validator = super().get_validator()
        finally:
            # Restore VSA args
            for attr, val in saved_args.items():
                setattr(self.args, attr, val)
                
        return validator

    def _setup_train(self):
        """
        Setup training environment. 
        Overridden to DISABLE ModelEMA due to deepcopy issues with dynamic VSA params.
        """
        # 1. Model Setup (Missing in previous override)
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Initialize loss criterion
        if hasattr(self.model, "init_criterion"):
            self.model.criterion = self.model.init_criterion()

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (self.args.freeze if isinstance(self.args.freeze, list) 
                       else range(self.args.freeze) if isinstance(self.args.freeze, int) else [])
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(f"setting 'requires_grad=True' for frozen layer '{k}'")
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and self.world_size > 1:
            dist.broadcast(self.amp.int(), src=0)
        self.amp = bool(self.amp)
        self.scaler = (torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 
                       else torch.cuda.amp.GradScaler(enabled=self.amp))

        # DDP
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], 
                                                             find_unused_parameters=True, 
                                                             gradient_as_bucket_view=True)
        
        # 2. Dataset & Loader Setup
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            
            # DISABLE EMA for VSA Training due to RuntimeError
            # self.ema = ModelEMA(self.model) 
            self.ema = None # Explicitly set to None
            
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(self.resume) # Original uses ckpt, but resume handles it? No.
        # Original: self.resume_training(ckpt) -> ckpt is passed to _setup_train? 
        # No, ckpt is a local var in _do_train, NOT passed to _setup_train in original code?
        # WAIT. In original code:
        # def _do_train(self):
        #    ...
        #    self._setup_train()
        #
        # def _setup_train(self):
        #    ...
        #    ckpt = self.setup_model()
        #    ...
        #    self.resume_training(ckpt)
        
        # Ah, self.setup_model() returns ckpt or None.
        ckpt = self.setup_model()
        self.resume_training(ckpt)
        
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build VSAVideoDataset for training with sparse sampling support.
        """
        if mode == "train":
            # For Training: Load Sequences with sparse sampling
            time_steps = getattr(self.args, 'time_steps', 20)
            vid_stride = getattr(self.args, 'vid_stride', 1)
            
            # Get random crop params from parent (set in SAM2VideoTrainer.__init__)
            random_crop_size = getattr(self, 'random_crop_size', self.args.imgsz)
            random_crop_prob = getattr(self, 'random_crop_prob', 1.0)
            
            LOGGER.info(f"VSA Dataset: time_steps={time_steps}, vid_stride={vid_stride}, "
                       f"temporal_span={time_steps * vid_stride} frames, "
                       f"random_crop_size={random_crop_size}, random_crop_prob={random_crop_prob}")
            
            return VSAVideoDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=False,  # Rectangular training disabled for VSA synced resizing
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=32,
                pad=0.0,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                # VSA-specific args
                time_steps=time_steps,
                vid_stride=vid_stride,
                random_crop_size=random_crop_size,
                random_crop_prob=random_crop_prob
            )
        else:
            # For Validation: Use standard VisDroneVideoDataset (history pair)
            return super().build_dataset(img_path, mode, batch)

    def preprocess_batch(self, batch):
        """
        Custom preprocessing for VSA long clips.
        Batch 'img' is [B, T*3, H, W].
        We need to reshape it to [B, T, 3, H, W] -> Flatten to [B*T, 3, H, W] for model input.
        """
        # Check if we are using the VSA Dataset (Train mode)
        img = batch['img']
        
        # Heuristic: If channels > 6, it's likely a long clip
        if img.shape[1] > 6:
            B, C_total, H, W = img.shape
            T = C_total // 3
            
            # Reshape: [B, T*3, H, W] -> [B, T, 3, H, W] -> [B*T, 3, H, W]
            seq_img = img.view(B, T, 3, H, W).view(-1, 3, H, W)
            batch["img"] = seq_img
            
            # Update Model Time Steps
            # This is critical for VSA forward loop to know T
            if hasattr(self.model, "time_steps"):
                self.model.time_steps = T
                
            # Handle Labels
            # Dataset only returns labels for the LAST frame of the clip.
            # batch['batch_idx'] maps to image indices in the flattened batch.
            # Flattened batch indices: 
            # Clip 0: [0, 1, ..., T-1] -> Label is for T-1
            # Clip 1: [T, T+1, ..., 2T-1] -> Label is for 2T-1
            
            # We need to remap 'batch_idx' to point to the correct frame in the flattened batch.
            # Incoming batch['batch_idx'] is [0, 0, ..., 1, 1, ...] (Sample Index)
            # We want it to point to: Sample_Index * T + (T-1)
            
            original_batch_idx = batch['batch_idx']
            new_batch_idx = original_batch_idx * T + (T - 1)
            batch['batch_idx'] = new_batch_idx
            
            # IMPORTANT: Batch Visualization Fix
            # If we don't fix metadata, plotting code might crash or look wrong
            if "im_file" in batch:
                 # vsa_dataset now returns a LIST of strings for each sample [Path1, Path2... PathT]
                 # collate_fn turns this into a tuple of tuples/lists? 
                 # Let's inspect what we get.
                 # If batch['im_file'] is a tuple of T-length lists:
                 # We want a single flat tuple of size B*T
                 
                 im_files = batch["im_file"]
                 new_im_files = []
                 
                 for item in im_files:
                     if isinstance(item, (list, tuple)):
                         # It is a sequence of paths for this clip
                         new_im_files.extend(item)
                     else:
                         # Fallback if dataset returns single string (should not happen with new code)
                         new_im_files.extend([item] * T)
                         
                 batch["im_file"] = tuple(new_im_files)
                
        else:
            # Fallback for Validation (Standard SAM2 behavior)
            return super().preprocess_batch(batch)
            
        return batch
