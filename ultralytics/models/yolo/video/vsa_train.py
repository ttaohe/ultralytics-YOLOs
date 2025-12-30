
from copy import copy
import torch
import torch.nn as nn
from ultralytics.models.yolo.video.train import SAM2VideoTrainer
import math
from ultralytics.data.vsa_dataset import VSAVideoDataset
from ultralytics.utils import colorstr, LOGGER, RANK, LOCAL_RANK, callbacks
from ultralytics.utils.torch_utils import EarlyStopping, attempt_compile, TORCH_2_4
from ultralytics.utils.checks import check_imgsz, check_amp
import torch.distributed as dist
from ultralytics.cfg import DEFAULT_CFG
import os


class VSAVideoTrainer(SAM2VideoTrainer):
    """
    Trainer for VSA (Video Sparse Attention) models.
    Supports loading long video clips (e.g., T=8) via VSAVideoDataset.
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize VSAVideoTrainer.

        Extra args:
            time_steps: Number of frames per clip (default 8)
            vid_stride: Frame sampling stride for sparse sampling (default 1)
        """
        # ============ DDP DEBUG: Early log to confirm RANK1 starts ============
        rank_env = int(os.getenv("RANK", -1))
        LOGGER.info(f"[VSA-INIT-RANK{rank_env}] VSAVideoTrainer.__init__() START")
        # ===============================================================

        if overrides is None:
            overrides = {}

        # Extract VSA-specific args
        self.time_steps = int(overrides.pop('time_steps', 8))
        vsa_vid_stride = int(overrides.get('vid_stride', 1))

        super().__init__(cfg, overrides, _callbacks)

        self.vid_stride = vsa_vid_stride
        self.args.time_steps = self.time_steps
        self.args.vid_stride = self.vid_stride

        LOGGER.info(f"[VSA-INIT-RANK{rank_env}] VSAVideoTrainer.__init__() DONE, time_steps={self.time_steps}")

        # Register debug callbacks
        self.add_callback("on_train_epoch_start", self._on_epoch_start_debug)
        self.add_callback("on_train_epoch_end", self._on_epoch_end_debug)
        self.add_callback("on_fit_epoch_end", self._on_fit_epoch_end_debug)
        self.add_callback("on_train_batch_start", self._on_batch_start_debug)
        self.add_callback("on_train_batch_end", self._on_batch_end_debug)
        self.add_callback("on_params_update", self._on_params_update_debug)
        self.add_callback("on_before_forward", self._on_before_forward_debug)

        LOGGER.info(f"VSAVideoTrainer: time_steps={self.time_steps}, vid_stride={self.vid_stride}, "
                   f"random_crop_size={self.random_crop_size}, random_crop_prob={self.random_crop_prob}")

    def _on_epoch_start_debug(self, trainer):
        """Debug callback to track epoch starts."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        # NOTE: Ultralytics LOGGER is typically muted for non-zero DDP ranks, so use print() for sync debugging.
        print(f"[VSA-RANK{rank_env}] on_train_epoch_start: epoch={trainer.epoch}", flush=True)

        # Propagate debug flags to dataset so we can see where DDP rank1 gets stuck (dataloader vs GPU).
        try:
            if hasattr(trainer, "train_loader") and hasattr(trainer.train_loader, "dataset"):
                ds = trainer.train_loader.dataset
                setattr(ds, "_debug_epoch", getattr(trainer, "epoch", -1))
                setattr(ds, "_debug_rank", rank_env)
                setattr(ds, "_debug_count", 0)  # reset each epoch
        except Exception:
            pass

        # ============ OPTIONAL DDP DEBUG: Barrier at epoch start ============
        # WARNING: If a previous NCCL collective is desynced (e.g., dynamic unused params across ranks),
        # the *next* collective may hang. Keep this barrier opt-in.
        if (
            os.getenv("VSA_EPOCH_START_BARRIER", "0") == "1"
            and RANK != -1
            and self.world_size > 1
        ):
            print(f"[VSA-RANK{rank_env}] on_train_epoch_start: BARRIER", flush=True)
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dist.barrier()
                print(f"[VSA-RANK{rank_env}] on_train_epoch_start: BARRIER PASSED", flush=True)
            except Exception as e:
                print(f"[VSA-RANK{rank_env}] on_train_epoch_start: BARRIER FAILED: {e}", flush=True)
                raise
        # ===================================================================

    def _on_epoch_end_debug(self, trainer):
        """Debug callback to track epoch ends."""
        rank_env = int(os.getenv("RANK", -1))
        print(f"[VSA-RANK{rank_env}] _on_epoch_end_debug START, epoch={trainer.epoch}", flush=True)
        # No barrier - just log to see if RANK1 reaches here
        print(f"[VSA-RANK{rank_env}] _on_epoch_end_debug DONE, epoch={trainer.epoch}", flush=True)

    def _on_fit_epoch_end_debug(self, trainer):
        """Debug callback to track fit epoch ends (all ranks execute this)."""
        rank_env = int(os.getenv("RANK", -1))
        print(f"[VSA-RANK{rank_env}] _on_fit_epoch_end_debug START, epoch={trainer.epoch}", flush=True)

        # CRITICAL: Reset memory banks on ALL ranks after each epoch
        # This ensures DDP ranks stay in sync, since only RANK0 does validation
        print(f"[VSA-RANK{rank_env}] on_fit_epoch_end: resetting VSA memory banks", flush=True)
        self._reset_vsa_memory_banks()
        print(f"[VSA-RANK{rank_env}] _on_fit_epoch_end_debug DONE, epoch={trainer.epoch}", flush=True)

    def _on_batch_start_debug(self, trainer):
        """Debug callback to track batch starts."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        if trainer.epoch == 1:
            LOGGER.info(f"[VSA-RANK{rank_env}] on_train_batch_start: epoch={trainer.epoch}, step starting...")

    def _on_batch_end_debug(self, trainer):
        """Debug callback to track batch ends."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        if trainer.epoch == 1:
            LOGGER.info(f"[VSA-RANK{rank_env}] on_train_batch_end: epoch={trainer.epoch}, step done")

    def _on_params_update_debug(self, trainer):
        """Debug callback to track params update."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        if trainer.epoch == 1:
            LOGGER.info(f"[VSA-RANK{rank_env}] on_params_update: epoch={trainer.epoch}")

    def _on_before_forward_debug(self, trainer):
        """Debug callback called right before forward."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        if trainer.epoch == 1:
            LOGGER.info(f"[VSA-RANK{rank_env}] on_before_forward: epoch={trainer.epoch}")
    
    def get_validator(self):
        """Returns VideoValidator, temporarily removing VSA-specific args."""
        saved_args = {}
        for attr in ['time_steps', 'vid_stride']:
            if hasattr(self.args, attr):
                saved_args[attr] = getattr(self.args, attr)
                delattr(self.args, attr)
        try:
            validator = super().get_validator()
        finally:
            for attr, val in saved_args.items():
                setattr(self.args, attr, val)
        return validator

    def _setup_train(self):
        """
        Setup training environment.
        Overridden to disable ModelEMA (causes issues with dynamic VSA params).
        """
        # Model Setup
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        if hasattr(self.model, "init_criterion"):
            self.model.criterion = self.model.init_criterion()

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
                v.requires_grad = True

        # AMP
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
            # VSA model has some parameters that may not be used in every forward pass
            # (e.g., maskmem_tpos_enc when memory bank is empty)
            # We need find_unused_parameters=True, but this can cause deadlock at epoch boundaries
            # Solution: Add explicit barrier after backward to ensure synchronization
            LOGGER.info(f"[VSA-DDP-RANK{RANK}] Creating DDP model with find_unused_parameters=True")
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[RANK],
                find_unused_parameters=True,  # Required for VSA model
                gradient_as_bucket_view=True
            )
        
        # Dataset & Loader
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs

        if self.batch_size < 1 and RANK == -1:
            self.args.batch = self.batch_size = self.auto_batch()

        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1, mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            
            # DISABLE EMA for VSA (causes RuntimeError with dynamic params)
            self.ema = None
            
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
        
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build VSAVideoDataset for training."""
        if mode == "train":
            time_steps = getattr(self.args, 'time_steps', 8)
            vid_stride = getattr(self.args, 'vid_stride', 1)
            random_crop_size = getattr(self, 'random_crop_size', self.args.imgsz)
            random_crop_prob = getattr(self, 'random_crop_prob', 1.0)
            
            LOGGER.info(f"VSA Dataset: time_steps={time_steps}, vid_stride={vid_stride}, "
                       f"temporal_span={time_steps * vid_stride} frames")
            
            return VSAVideoDataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=False,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=32,
                pad=0.0,
                prefix=colorstr(f"{mode}: "),
                task=self.args.task,
                classes=self.args.classes,
                data=self.data,
                time_steps=time_steps,
                vid_stride=vid_stride,
                random_crop_size=random_crop_size,
                random_crop_prob=random_crop_prob
            )
        else:
            return super().build_dataset(img_path, mode, batch)

    def preprocess_batch(self, batch):
        """
        Preprocess VSA long clips.
        Reshape [B, T*3, H, W] -> [B*T, 3, H, W] for model input.
        """
        img = batch['img']

        # Debug: Log both ranks. NOTE: LOGGER is often muted for non-zero DDP ranks, so use print().
        import os
        import time
        rank_env = int(os.getenv("RANK", -1))
        t0 = time.time()
        if hasattr(self, "epoch") and self.epoch == 1:
            print(f"[VSA-RANK{rank_env}] preprocess_batch epoch=1 START", flush=True)
            print(f"[VSA-RANK{rank_env}]   img.shape={img.shape}, img.device={img.device}", flush=True)
            print(f"[VSA-RANK{rank_env}]   self.device={self.device}", flush=True)

        # VSA training mode (multi-frame clip)
        if img.shape[1] > 6:
            B, C_total, H, W = img.shape
            T = C_total // 3

            if hasattr(self, "epoch") and self.epoch == 1:
                print(f"[VSA-RANK{rank_env}]   VSA mode: B={B}, C_total={C_total}, T={T}", flush=True)

            # First move all tensors to GPU (parent class does this, but we need it before reshape)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and not v.is_cuda:
                    if hasattr(self, "epoch") and self.epoch == 1:
                        print(f"[VSA-RANK{rank_env}]   Moving {k} from {v.device} to {self.device}", flush=True)
                    batch[k] = v.to(self.device, non_blocking=True)

            # Update img reference after moving to GPU (CRITICAL!)
            img = batch['img']

            if hasattr(self, "epoch") and self.epoch == 1:
                print(f"[VSA-RANK{rank_env}]   After moving: img.device={img.device}", flush=True)

            # Reshape to [B*T, 3, H, W]
            batch["img"] = img.view(B, T, 3, H, W).view(-1, 3, H, W)

            if hasattr(self, "epoch") and self.epoch == 1:
                print(
                    f"[VSA-RANK{rank_env}]   After reshape: img.shape={batch['img'].shape}, device={batch['img'].device}",
                    flush=True,
                )

            # Set time_steps on the actual model (handle DDP)
            model = self.model.module if hasattr(self.model, 'module') else self.model
            self._set_model_time_steps(model, T)

            if hasattr(self, "epoch") and self.epoch == 1:
                print(f"[VSA-RANK{rank_env}]   After _set_model_time_steps", flush=True)

            # Remap batch_idx for labels (labels are for last frame only)
            batch['batch_idx'] = batch['batch_idx'] * T + (T - 1)

            # Flatten im_file for visualization
            if "im_file" in batch:
                new_im_files = []
                for item in batch["im_file"]:
                    if isinstance(item, (list, tuple)):
                        new_im_files.extend(item)
                    else:
                        new_im_files.extend([item] * T)
                batch["im_file"] = tuple(new_im_files)

            if hasattr(self, "epoch") and self.epoch == 1:
                # Print a small hint about which files this rank got, to spot "bad" samples.
                im_file = batch.get("im_file", None)
                if im_file is not None:
                    try:
                        if isinstance(im_file, (list, tuple)) and len(im_file):
                            hint = im_file[-1]
                        else:
                            hint = im_file
                        print(f"[VSA-RANK{rank_env}]   im_file(last)={hint}", flush=True)
                    except Exception:
                        pass
                print(f"[VSA-RANK{rank_env}]   preprocess_batch DONE, returning batch", flush=True)

            # NOTE: We don't call super().preprocess_batch() here because:
            # 1. Data is already moved to GPU above
            # 2. Data is already normalized in VSAVideoDataset (stack / 255.0)
            # 3. Calling super() would re-normalize, causing values / 255 / 255
            if hasattr(self, "epoch") and self.epoch == 1:
                print(f"[VSA-RANK{rank_env}]   About to return from preprocess_batch (dt={time.time()-t0:.3f}s)", flush=True)
            return batch
        else:
            # Non-VSA mode: use parent's preprocessing
            return super().preprocess_batch(batch)
    
    def _set_model_time_steps(self, model, T):
        """Set time_steps on all VSAMemoryAttention modules."""
        for module in model.modules():
            if hasattr(module, 'time_steps'):
                module.time_steps = T
    
    def validate(self):
        """
        Override validation to reset memory banks and ensure clean state.

        IMPORTANT: We temporarily disable AMP during validation to avoid
        model.half()/float() calls which create inference tensors under
        torch.inference_mode() that cannot be used for backward pass.

        NOTE: validate() is only called on RANK0 in DDP training, so no DDP barriers here.
        """
        import os
        rank_env = int(os.getenv("RANK", -1))
        LOGGER.info(f"[VSA-RANK{rank_env}] validate() START, epoch={getattr(self, 'epoch', '?')}")

        # Reset memory banks before validation
        LOGGER.info(f"[VSA-RANK{rank_env}] validate(): calling _reset_vsa_memory_banks BEFORE")
        self._reset_vsa_memory_banks()
        LOGGER.info(f"[VSA-RANK{rank_env}] validate(): _reset_vsa_memory_banks DONE")

        # CRITICAL FIX: Temporarily disable AMP to avoid model.half()/float()
        # which creates inference tensors under torch.inference_mode()
        original_amp = self.amp
        self.amp = False
        LOGGER.info(f"[VSA-RANK{rank_env}] validate(): AMP disabled, calling super().validate()")

        try:
            result = super().validate()
            LOGGER.info(f"[VSA-RANK{rank_env}] validate(): super().validate() DONE")
        finally:
            # Restore AMP setting
            self.amp = original_amp
            LOGGER.info(f"[VSA-RANK{rank_env}] validate(): AMP restored")

        # Reset memory banks after validation
        LOGGER.info(f"[VSA-RANK{rank_env}] validate(): calling _reset_vsa_memory_banks AFTER")
        self._reset_vsa_memory_banks()
        LOGGER.info(f"[VSA-RANK{rank_env}] validate(): _reset_vsa_memory_banks DONE")

        # Ensure model is back in training mode
        if hasattr(self, 'model') and not isinstance(self.model, str):
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.train()
            LOGGER.info(f"[VSA-RANK{rank_env}] validate(): model.train() called")

        LOGGER.info(f"[VSA-RANK{rank_env}] validate() END, returning to training")
        return result


    def _reset_vsa_memory_banks(self):
        """Reset all VSA memory banks in the model."""
        import os
        rank_env = int(os.getenv("RANK", -1))
        LOGGER.info(f"[VSA-RANK{rank_env}] _reset_vsa_memory_banks: START")

        if not hasattr(self, 'model') or isinstance(self.model, str):
            LOGGER.info(f"[VSA-RANK{rank_env}] _reset_vsa_memory_banks: early return (no model)")
            return

        model = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model, 'modules'):
            LOGGER.info(f"[VSA-RANK{rank_env}] _reset_vsa_memory_banks: early return (no modules)")
            return

        LOGGER.info(f"[VSA-RANK{rank_env}] _reset_vsa_memory_banks: iterating modules...")
        reset_count = 0
        for module in model.modules():
            if hasattr(module, 'reset_memory'):
                LOGGER.info(f"[VSA-RANK{rank_env}]   Calling reset_memory on {type(module).__name__}")
                module.reset_memory()
                reset_count += 1
            elif hasattr(module, 'memory_bank'):
                LOGGER.info(f"[VSA-RANK{rank_env}]   Clearing memory_bank on {type(module).__name__}, len={len(module.memory_bank)}")
                module.memory_bank = []
                reset_count += 1
            if hasattr(module, 'key_bank'):
                LOGGER.info(f"[VSA-RANK{rank_env}]   Clearing key_bank on {type(module).__name__}, len={len(module.key_bank)}")
                module.key_bank = []
                reset_count += 1
            if hasattr(module, 'val_bank'):
                LOGGER.info(f"[VSA-RANK{rank_env}]   Clearing val_bank on {type(module).__name__}, len={len(module.val_bank)}")
                module.val_bank = []
            if hasattr(module, 'pos_bank'):
                LOGGER.info(f"[VSA-RANK{rank_env}]   Clearing pos_bank on {type(module).__name__}, len={len(module.pos_bank)}")
                module.pos_bank = []

        LOGGER.info(f"[VSA-RANK{rank_env}] _reset_vsa_memory_banks: DONE (reset {reset_count} modules)")
