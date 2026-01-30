#!/usr/bin/env python3
"""
Generate overlap masks from PE npz files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def infer_const(camera_name: str) -> torch.Tensor:
    if camera_name.endswith("1"):
        return torch.tensor([1.0, 0.0, 1.0, 0.0])
    if camera_name.endswith("2"):
        return torch.tensor([-1.0, 0.0, -1.0, 0.0])
    raise ValueError(f"Unknown camera name for const PE: {camera_name}")


def compute_mask(pe: np.ndarray, const: torch.Tensor, eps: float, ds: int) -> np.ndarray:
    pe_t = torch.from_numpy(pe).float()
    if pe_t.ndim != 3 or pe_t.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected PE shape: {pe_t.shape}")
    if pe_t.shape[-1] == 3:
        pad = torch.zeros((*pe_t.shape[:2], 1), dtype=pe_t.dtype)
        pe_t = torch.cat([pe_t, pad], dim=-1)

    const = const.to(pe_t.device, dtype=pe_t.dtype)
    diff = (pe_t - const).abs().amax(dim=-1)
    valid = diff > eps

    if ds > 1:
        v = valid.float()[None, None, ...]
        v = F.max_pool2d(v, kernel_size=ds, stride=ds, ceil_mode=True)
        valid = v[0, 0] > 0

    return valid.cpu().numpy().astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--camera-name", type=str, default=None)
    ap.add_argument("--ds", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    pe_root = Path(args.pe_root)
    if not pe_root.exists():
        raise FileNotFoundError(pe_root)

    out_root = Path(args.out_root) if args.out_root else pe_root
    out_root.mkdir(parents=True, exist_ok=True)

    cam = args.camera_name
    if cam is None:
        name = pe_root.name
        if "camera1" in name:
            cam = "camera1"
        elif "camera2" in name:
            cam = "camera2"
    if cam is None:
        raise ValueError("--camera-name required when camera not inferable from path")

    const = infer_const(cam)

    npz_files = sorted(pe_root.rglob("*_pe.npz"))
    if not npz_files:
        raise RuntimeError(f"No *_pe.npz found under {pe_root}")

    for p in tqdm(npz_files, desc=f"Generating masks (ds={args.ds})"):
        rel = p.relative_to(pe_root)
        out_dir = out_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / p.name.replace("_pe.npz", f"_overlap_mask_ds{args.ds}.npy")
        if out_path.exists() and not args.overwrite:
            continue

        pe = np.load(p)["pe"]
        mask = compute_mask(pe, const, args.eps, args.ds)
        np.save(out_path, mask)

    print(f"Done. Wrote masks to: {out_root}")


if __name__ == "__main__":
    main()
