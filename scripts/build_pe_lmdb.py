#!/usr/bin/env python3
"""
Build LMDB from PE .npy files and write an index.json alongside.
"""

from pathlib import Path
import argparse
import io
import lmdb
import numpy as np
from tqdm import tqdm
import os


def iter_npy_files(pes_dir: Path):
    for root, _, files in os.walk(pes_dir):
        for name in files:
            if name.endswith(".npy"):
                yield Path(root) / name


def estimate_map_size_and_count(pes_dir: Path, overhead: float = 1.2) -> tuple[int, int]:
    total = 0
    count = 0
    for f in iter_npy_files(pes_dir):
        total += f.stat().st_size
        count += 1
    return int(total * overhead) + (128 << 20), count  # add 128MB slack


def build_lmdb(pes_dir: Path, lmdb_path: Path, commit_interval: int, index_format: str, map_size: int | None) -> None:
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)
    if map_size is None:
        map_size, total_files = estimate_map_size_and_count(pes_dir)
    else:
        total_files = None
    env = lmdb.open(str(lmdb_path), map_size=map_size)

    index_file = None
    if index_format in ("txt", "jsonl"):
        index_path = lmdb_path.with_suffix(f".index.{index_format}")
        index_file = index_path.open("w", encoding="utf-8")

    count = 0
    txn = env.begin(write=True)
    file_iter = iter_npy_files(pes_dir)
    for f in tqdm(file_iter, desc=f"Building {lmdb_path.name}", unit="file", total=total_files):
        rel = f.relative_to(pes_dir).as_posix()
        arr = np.load(f)
        buf = io.BytesIO()
        np.save(buf, arr)
        txn.put(rel.encode("utf-8"), buf.getvalue())
        if index_file:
            index_file.write(rel + "\n")
        count += 1
        if commit_interval > 0 and count % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.sync()
    env.close()
    if index_file:
        index_file.close()
    print(f"Saved {count} entries to {lmdb_path} (map_size={map_size})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Dataset root, e.g. /home/.../MDMT-yolo")
    parser.add_argument("--cameras", nargs="+", default=["camera1", "camera2"])
    parser.add_argument("--source", type=str, default="pes", help="Source folder under each camera (pes or pes_fp16)")
    parser.add_argument("--suffix", type=str, default="fp32", help="LMDB suffix (fp32 or fp16)")
    parser.add_argument("--commit-interval", type=int, default=1000, help="Commit every N entries")
    parser.add_argument(
        "--index-format",
        type=str,
        default="txt",
        choices=["txt", "jsonl", "none"],
        help="Index file format",
    )
    parser.add_argument("--map-size", type=int, default=0, help="LMDB map size in bytes (0 to auto-estimate)")
    args = parser.parse_args()

    base = Path(args.base)
    for cam in args.cameras:
        cam_dir = base / cam
        pes_dir = cam_dir / args.source
        if not pes_dir.exists():
            raise FileNotFoundError(f"Missing {pes_dir}")
        lmdb_path = cam_dir / f"pes_{args.suffix}.lmdb"
        index_format = args.index_format if args.index_format != "none" else ""
        build_lmdb(
            pes_dir,
            lmdb_path,
            commit_interval=args.commit_interval,
            index_format=index_format,
            map_size=args.map_size or None,
        )


if __name__ == "__main__":
    main()
