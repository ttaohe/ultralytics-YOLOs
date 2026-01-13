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
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/v10/yolov10l-multiview_earlyfusion.yaml")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/MDMT_multiview.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/train_multiview")
    parser.add_argument("--name", type=str, default="yolo10l-multiview-disaug-earlyfusion")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--cache", type=str, default='disk')
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--verbose-test-only", default=True, help="Only print per-class results during test/val.")
    parser.add_argument("--pe-lmdb", type=str, default=None, help="Use LMDB PE storage: fp16, fp32, or custom path")
    parser.add_argument("--time-log-interval", type=int, default=0, help="Print data/compute time every N iterations")
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

    # Load data config
    data_dict = YAML.load(opt.data)
    if opt.pe_lmdb:
        data_dict["pe_lmdb"] = opt.pe_lmdb
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
            "verbose_test_only": opt.verbose_test_only,
            "time_log_interval": opt.time_log_interval,
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

    # Start training
    trainer.train()


if __name__ == "__main__":
    train()
