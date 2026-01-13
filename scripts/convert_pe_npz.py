#!/usr/bin/env python3
"""
Convert PE .npy files to compressed .npz for all cameras.
"""

from pathlib import Path
import argparse
import os
import numpy as np
from tqdm import tqdm


def list_npy_files(root: Path):
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(".npy"):
                files.append(Path(dirpath) / name)
    return files


def convert_dir(src_dir: Path, dst_dir: Path, limit: int | None = None) -> int:
    files = list_npy_files(src_dir)
    if limit:
        files = files[:limit]
    for f in tqdm(files, desc=f"Converting {src_dir}", unit="file"):
        rel = f.relative_to(src_dir)
        out = dst_dir / rel.with_suffix(".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        arr = np.load(f)
        np.savez_compressed(out, pe=arr)
    return len(files)


def convert_camera(cam_dir: Path, source: str, out_name: str, limit: int | None) -> int:
    src_dir = cam_dir / source
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing {src_dir}")
    dst_dir = cam_dir / out_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    return convert_dir(src_dir, dst_dir, limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Dataset root, e.g. /home/.../MDMT-yolo")
    parser.add_argument("--cameras", nargs="+", default=["camera1", "camera2"])
    parser.add_argument("--source", type=str, default="pes_fp16", help="Source folder under each camera")
    parser.add_argument("--out-name", type=str, default="pes_npz", help="Output folder name under each camera")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files per camera (0 = all)")
    args = parser.parse_args()

    base = Path(args.base)
    limit = args.limit if args.limit > 0 else None
    for cam in args.cameras:
        cam_dir = base / cam
        n = convert_camera(cam_dir, args.source, args.out_name, limit)
        print(f"{cam}: converted {n} files")


if __name__ == "__main__":
    main()
