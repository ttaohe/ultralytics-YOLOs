#!/usr/bin/env python3
"""
Convert PE .npy files to fp16 under pes_fp16/ and backup original to pes_fp32/.
"""

from pathlib import Path
import argparse
import shutil
import numpy as np
from tqdm import tqdm


def convert_camera(cam_dir: Path) -> None:
    pes_dir = cam_dir / "pes"
    if not pes_dir.exists():
        raise FileNotFoundError(f"Missing {pes_dir}")

    fp32_dir = cam_dir / "pes_fp32"
    fp16_dir = cam_dir / "pes_fp16"

    if not fp32_dir.exists():
        shutil.copytree(pes_dir, fp32_dir)

    fp16_dir.mkdir(parents=True, exist_ok=True)
    files = list(pes_dir.rglob("*.npy"))
    for src in tqdm(files, desc=f"Converting {cam_dir.name}", unit="file"):
        rel = src.relative_to(pes_dir)
        dst = fp16_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        arr = np.load(src)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        np.save(dst, arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Dataset root, e.g. /home/.../MDMT-yolo")
    parser.add_argument("--cameras", nargs="+", default=["camera1", "camera2"])
    args = parser.parse_args()

    base = Path(args.base)
    for cam in args.cameras:
        convert_camera(base / cam)

    print("done")


if __name__ == "__main__":
    main()
