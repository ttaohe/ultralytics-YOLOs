#!/usr/bin/env python3
"""
Benchmark PE I/O for a sample of files (npy/npz/lmdb).
"""

from pathlib import Path
import argparse
import io
import time
import os
import random
import numpy as np


def list_npy_files(pes_dir: Path):
    files = []
    for root, _, names in os.walk(pes_dir):
        for name in names:
            if name.endswith(".npy"):
                files.append(Path(root) / name)
    return files


def bench_npy(files):
    total_bytes = 0
    t0 = time.perf_counter()
    for f in files:
        arr = np.load(f)
        total_bytes += f.stat().st_size
        _ = arr.shape
    dt = time.perf_counter() - t0
    return dt, total_bytes


def build_npz_sample(files, pes_dir: Path, npz_dir: Path):
    npz_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        rel = f.relative_to(pes_dir)
        npz_path = npz_dir / rel.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.load(f)
        np.savez_compressed(npz_path, pe=arr)


def bench_npz(files, npz_dir: Path):
    total_bytes = 0
    t0 = time.perf_counter()
    for f in files:
        rel = f.relative_to(f.parents[2] / "pes")
        npz_path = npz_dir / rel.with_suffix(".npz")
        arr = np.load(npz_path)["pe"]
        total_bytes += npz_path.stat().st_size
        _ = arr.shape
    dt = time.perf_counter() - t0
    return dt, total_bytes


def bench_lmdb(files, pes_dir: Path, lmdb_path: Path):
    import lmdb  # optional dependency

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, max_readers=256)
    total_bytes = 0
    t0 = time.perf_counter()
    with env.begin(write=False) as txn:
        for f in files:
            rel = f.relative_to(pes_dir).as_posix()
            buf = txn.get(rel.encode("utf-8"))
            if buf is None:
                raise FileNotFoundError(f"Missing key in LMDB: {rel}")
            total_bytes += len(buf)
            arr = np.load(io.BytesIO(buf), allow_pickle=False)
            _ = arr.shape
    dt = time.perf_counter() - t0
    env.close()
    return dt, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pes-dir", type=str, required=True, help="PE directory containing .npy files")
    parser.add_argument("--count", type=int, default=100, help="Number of files to sample")
    parser.add_argument("--mode", choices=["npy", "npz", "lmdb", "all"], default="all")
    parser.add_argument("--npz-dir", type=str, help="Directory containing .npz files (same structure as pes-dir)")
    parser.add_argument("--lmdb", type=str, help="Path to LMDB file")
    parser.add_argument("--sample-dir", type=str, default="sample_pe_bench", help="Dir to store sample npz files")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pes_dir = Path(args.pes_dir)
    all_files = list_npy_files(pes_dir)
    if not all_files:
        raise SystemExit(f"No .npy files found under {pes_dir}")
    rnd = random.Random(args.seed)
    if args.count > len(all_files):
        args.count = len(all_files)
    files = rnd.sample(all_files, args.count)
    if not files:
        raise SystemExit(f"No .npy files found under {pes_dir}")

    if args.mode in ("npy", "all"):
        dt, total_bytes = bench_npy(files)
        per_file = dt / len(files)
        mbps = total_bytes / dt / (1024 * 1024)
        print(f"mode=npy files={len(files)} time={dt:.3f}s per_file={per_file:.4f}s throughput={mbps:.1f} MB/s")

    if args.mode in ("npz", "all"):
        npz_dir = Path(args.npz_dir) if args.npz_dir else Path(args.sample_dir)
        build_npz_sample(files, pes_dir, npz_dir)
        dt, total_bytes = bench_npz(files, npz_dir)
        per_file = dt / len(files)
        mbps = total_bytes / dt / (1024 * 1024)
        print(f"mode=npz files={len(files)} time={dt:.3f}s per_file={per_file:.4f}s throughput={mbps:.1f} MB/s")

    if args.mode in ("lmdb", "all"):
        if not args.lmdb:
            print("mode=lmdb skipped (no --lmdb provided)")
        else:
            dt, total_bytes = bench_lmdb(files, pes_dir, Path(args.lmdb))
            per_file = dt / len(files)
            mbps = total_bytes / dt / (1024 * 1024)
            print(f"mode=lmdb files={len(files)} time={dt:.3f}s per_file={per_file:.4f}s throughput={mbps:.1f} MB/s")


if __name__ == "__main__":
    main()
