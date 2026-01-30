#!/usr/bin/env python3
"""
Multiview YOLO Training Script
Directly uses MultiviewTrainer for training.
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics.utils import LOGGER, YAML, colorstr

torch.multiprocessing.set_sharing_strategy("file_system")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xxxx")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/MDMT_multiview.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/train_multiview")
    parser.add_argument("--name", type=str, default="xxxx")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--val-imgsz", type=int, default=0, help="Override validation image size (0 disables)")
    parser.add_argument("--val-tile", action="store_true", help="Enable sliding-window val/test")
    parser.add_argument("--val-tile-size", type=int, default=640, help="Tile size for val/test sliding window")
    parser.add_argument("--val-tile-stride", type=int, default=480, help="Tile stride for val/test sliding window")
    parser.add_argument(
        "--val-tile-disable-fusion",
        action="store_true",
        help="Disable multiview fusion (coords=None) during tiled val/test",
    )
    parser.add_argument(
        "--val-tile-original",
        action="store_true",
        help="Use original-resolution tiles for val/test (no letterbox tiles)",
    )
    parser.add_argument(
        "--val-original",
        action="store_true",
        help="Validate on original image size (skip LetterBox)",
    )
    parser.add_argument("--cache", type=str, default='False')
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use for fast runs")
    parser.add_argument("--verbose-test-only", default=True, help="Only print per-class results during test/val.")
    parser.add_argument("--pe-lmdb", type=str, default=None, help="Use LMDB PE storage: fp16, fp32, or custom path")
    parser.add_argument("--pe-disable", action="store_true", help="Disable PE loading (ablation)")
    parser.add_argument("--pe-npz-cam1", type=str, default=None, help="Override camera1 pes_npz root")
    parser.add_argument("--pe-npz-cam2", type=str, default=None, help="Override camera2 pes_npz root")
    parser.add_argument("--time-log-interval", type=int, default=0, help="Print data/compute time every N iterations")
    parser.add_argument("--align-loss-weight", type=float, default=0.0, help="Overlap alignment loss weight")
    parser.add_argument("--debug-dtype", action="store_true", help="Log dtype alignment during val (first batch only)")
    parser.add_argument("--strict-overlap-mask", action="store_true", help="Enable strict overlap fusion mask")
    parser.add_argument("--overlap-weight", action="store_true", help="Enable overlap weight scaling")
    parser.add_argument("--mask-key", default=True, help="Enable key masking in attention")
    parser.add_argument("--overlap-eval", action="store_true", help="Compute overlap/non-overlap metrics in val/test")
    parser.add_argument("--results-minimal", action="store_true", help="Save minimal metrics to results.csv")
    parser.add_argument(
        "--gate-method",
        type=str,
        default="gate_pe",
        help="Fusion gate method: residual02 | gate_xout | gate_pe | gt_soft",
    )
    parser.add_argument("--random-crop-size", type=int, default=0, help="Random crop size (0 disables)")
    parser.add_argument("--random-crop-prob", type=float, default=0.0, help="Random crop probability")
    parser.add_argument("--video-stride", type=int, default=1, help="Sparse video sampling stride (1 disables)")
    parser.add_argument(
        "--video-rand-start",
        dest="video_rand_start",
        action="store_true",
        default=True,
        help="Randomize video start offset each epoch (only with --video-stride > 1).",
    )
    parser.add_argument(
        "--no-video-rand-start",
        dest="video_rand_start",
        action="store_false",
        help="Disable random start offset (use fixed offset 0).",
    )
    parser.add_argument(
        "--multiview-cache-ignore-hash",
        action="store_true",
        default=True,
        help="Skip multiview cache hash validation (use cached file list directly).",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Profile this many training batches and then stop (0 disables profiling).",
    )
    parser.add_argument(
        "--profile-stack",
        action="store_true",
        help="Record Python stack traces in profiler (larger trace).",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record memory usage in profiler (larger trace).",
    )
    parser.add_argument(
        "--profile-shapes",
        action="store_true",
        help="Record tensor shapes in profiler (larger trace).",
    )
    opt = parser.parse_args()

    # If val_original is enabled, ignore val_imgsz to avoid forced resize/letterbox.
    if opt.val_original:
        opt.val_imgsz = 0

    # Load data config
    data_dict = YAML.load(opt.data)
    if opt.pe_lmdb:
        data_dict["pe_lmdb"] = opt.pe_lmdb
    if opt.pe_disable:
        data_dict["pe_disable"] = True
    if opt.pe_npz_cam1 or opt.pe_npz_cam2:
        data_dict["pe_npz_dirs"] = {}
        if opt.pe_npz_cam1:
            data_dict["pe_npz_dirs"]["camera1"] = opt.pe_npz_cam1
        if opt.pe_npz_cam2:
            data_dict["pe_npz_dirs"]["camera2"] = opt.pe_npz_cam2
    if opt.multiview_cache_ignore_hash:
        data_dict["multiview_cache_ignore_hash"] = True
    LOGGER.info(f"{colorstr('bright_blue')}Classes: {data_dict.get('names')}")
    LOGGER.info(f"{colorstr('bright_green')}Starting training...")
    LOGGER.info(f"  Model: {opt.model}")
    LOGGER.info(f"  Data: {opt.data}")
    LOGGER.info(f"  Epochs: {opt.epochs}")
    LOGGER.info(f"  Batch: {opt.batch} scenes (batch*{opt.num_views} images)")
    LOGGER.info(f"  Device: {opt.device}")

    # Import trainer after data is loaded
    from ultralytics.models.yolo.multiview.train import MultiviewTrainer

    # Create trainer with all necessary parameters in overrides
    # Note: num_views is NOT a standard YOLO argument, so we don't pass it here.
    # Instead, MultiviewTrainer will read it from self.data (the data config file)
    
    trainer = MultiviewTrainer(
        overrides={
            "model": opt.model,
            "data": opt.data,
            "epochs": opt.epochs,
            "imgsz": opt.imgsz,
            "batch": opt.batch,
            "device": opt.device,
            "workers": opt.workers,
            "project": opt.project,
            "name": opt.name,
            "cache": opt.cache or False,
            "patience": opt.patience,
            "save_period": opt.save_period,
            "seed": opt.seed,
            "deterministic": opt.deterministic,
            "fraction": opt.fraction,
            "verbose_test_only": opt.verbose_test_only,
            "time_log_interval": opt.time_log_interval,
            "align_loss_weight": opt.align_loss_weight,
            "debug_dtype": opt.debug_dtype,
            "video_stride": opt.video_stride,
            "video_rand_start": opt.video_rand_start,
            "overlap_eval": opt.overlap_eval,
            "results_minimal": opt.results_minimal,
            "gate_method": opt.gate_method,
            "random_crop_size": opt.random_crop_size,
            "random_crop_prob": opt.random_crop_prob,
            "val_original": opt.val_original,
            "val_tile": opt.val_tile,
            "val_tile_size": opt.val_tile_size,
            "val_tile_stride": opt.val_tile_stride,
            "val_tile_disable_fusion": opt.val_tile_disable_fusion,
            "val_tile_original": opt.val_tile_original,
        }
    )

    if opt.profile_steps and opt.profile_steps > 0:
        profile_state = {"steps": 0, "profiler": None, "trace_path": None}

        def _start_profiler(tr):
            from torch.profiler import ProfilerActivity, profile, schedule

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            trace_path = Path(tr.save_dir) / "profile_trace.json"
            prof = profile(
                activities=activities,
                schedule=schedule(wait=0, warmup=0, active=opt.profile_steps, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
                record_shapes=opt.profile_shapes,
                with_stack=opt.profile_stack,
                profile_memory=opt.profile_memory,
            )
            prof.start()
            profile_state.update({"profiler": prof, "trace_path": trace_path})
            LOGGER.info(f"{colorstr('bright_blue')}Profiling enabled for {opt.profile_steps} batches...")

        def _step_profiler(tr):
            prof = profile_state["profiler"]
            if prof is None:
                return
            prof.step()
            profile_state["steps"] += 1
            if profile_state["steps"] >= opt.profile_steps:
                prof.stop()
                tr.stop = True
                LOGGER.info(
                    f"{colorstr('bright_green')}Profile trace saved to {profile_state['trace_path']}"
                )

        trainer.callbacks["on_train_start"].append(_start_profiler)
        trainer.callbacks["on_train_batch_end"].append(_step_profiler)

    # Set data dict directly (already loaded)
    # num_views will be read from this dict by MultiviewTrainer
    trainer.data = data_dict
    if opt.val_imgsz and opt.val_imgsz > 0:
        trainer.val_imgsz = int(opt.val_imgsz)
    if opt.val_tile:
        trainer.args.val_tile = True
        trainer.args.val_tile_size = int(opt.val_tile_size)
        trainer.args.val_tile_stride = int(opt.val_tile_stride)
        trainer.args.val_tile_disable_fusion = bool(opt.val_tile_disable_fusion)
        trainer.args.val_tile_original = bool(opt.val_tile_original)
    if opt.val_original:
        trainer.args.val_original = True

    if opt.strict_overlap_mask:
        def _set_strict_overlap(tr):
            from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock
            for m in tr.model.modules():
                if isinstance(m, MultiviewFusionBlock):
                    m.strict_overlap_mask = True
        trainer.callbacks["on_train_start"].append(_set_strict_overlap)

    if opt.overlap_weight:
        def _set_overlap_weight(tr):
            from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock
            for m in tr.model.modules():
                if isinstance(m, MultiviewFusionBlock):
                    m.overlap_weight_enabled = True
        trainer.callbacks["on_train_start"].append(_set_overlap_weight)

    if not opt.mask_key:
        def _disable_mask_key(tr):
            from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock
            for m in tr.model.modules():
                if isinstance(m, MultiviewFusionBlock):
                    m.mask_key_enabled = False
        trainer.callbacks["on_train_start"].append(_disable_mask_key)

    def _set_gate_method(tr):
        from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock
        for m in tr.model.modules():
            if isinstance(m, MultiviewFusionBlock):
                m.gate_method = opt.gate_method
    trainer.callbacks["on_train_start"].append(_set_gate_method)

    # Start training
    trainer.train()


if __name__ == "__main__":
    train()
