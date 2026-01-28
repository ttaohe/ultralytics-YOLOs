#!/usr/bin/env python3
"""
Visualize overlap mask derived from PE npz for a random multiview sample.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def pick_random_image(cam1_dir: Path):
    images = sorted([p for p in cam1_dir.glob("*.jpg")])
    if not images:
        images = sorted([p for p in cam1_dir.glob("*.png")])
    if not images:
        raise FileNotFoundError(f"No images found under {cam1_dir}")
    return random.choice(images)


def map_to_cam2(cam1_path: Path, cam2_dir: Path):
    stem = cam1_path.stem
    if "-1_" in stem:
        stem2 = stem.replace("-1_", "-2_")
        cand = cam2_dir / f"{stem2}{cam1_path.suffix}"
        if cand.exists():
            return cand
    # fallback: try same stem
    cand = cam2_dir / f"{stem}{cam1_path.suffix}"
    if cand.exists():
        return cand
    # last resort: match by suffix after underscore
    if "_" in stem:
        tail = stem.split("_")[-1]
        matches = list(cam2_dir.glob(f"*_{tail}{cam1_path.suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find matching camera2 image for {cam1_path.name}")


def load_pe(pe_root: Path, split: str, stem: str):
    pe_path = pe_root / split / f"{stem}_pe.npz"
    if not pe_path.exists():
        raise FileNotFoundError(f"PE npz not found: {pe_path}")
    pe = np.load(pe_path)["pe"]
    return pe


def pe_to_rgb(pe: np.ndarray) -> np.ndarray:
    """Map PE (H,W,4) to RGB visualization using first 3 channels."""
    if pe.ndim != 3 or pe.shape[2] < 3:
        raise ValueError(f"Unexpected PE shape: {pe.shape}")
    rgb = pe[..., :3].astype(np.float32)
    # Normalize per-channel to [0,255]
    rgb_min = rgb.reshape(-1, 3).min(axis=0)
    rgb_max = rgb.reshape(-1, 3).max(axis=0)
    denom = np.maximum(rgb_max - rgb_min, 1e-6)
    rgb = (rgb - rgb_min) / denom
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ultralytics/cfg/datasets/MDMT_multiview.yaml", help="data yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="split to sample")
    parser.add_argument("--out-dir", default="runs/overlap_mask", help="output directory")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--num", type=int, default=1, help="number of random samples")
    args = parser.parse_args()

    random.seed(args.seed)

    data = yaml.safe_load(Path(args.data).read_text())
    root = Path(data["path"])
    cam_dirs = data.get("camera_dirs", ["camera1", "camera2"])
    cam1_dir = root / cam_dirs[0] / "images" / args.split
    cam2_dir = root / cam_dirs[1] / "images" / args.split

    pe_dirs = data.get("pe_npz_dirs", {})
    pe1_root = Path(pe_dirs.get(cam_dirs[0], root / cam_dirs[0] / "pes_npz"))
    pe2_root = Path(pe_dirs.get(cam_dirs[1], root / cam_dirs[1] / "pes_npz"))

    for _ in range(max(1, args.num)):
        cam1_img = pick_random_image(cam1_dir)
        cam2_img = map_to_cam2(cam1_img, cam2_dir)

        pe1 = load_pe(pe1_root, args.split, cam1_img.stem)
        pe2 = load_pe(pe2_root, args.split, cam2_img.stem)

        # Compute valid/overlap masks from constant PE codes
        const1 = np.array([1.0, 0.0, 1.0, 0.0], dtype=pe1.dtype)
        const2 = np.array([-1.0, 0.0, -1.0, 0.0], dtype=pe2.dtype)
        diff1 = np.max(np.abs(pe1 - const1), axis=-1)
        diff2 = np.max(np.abs(pe2 - const2), axis=-1)
        valid1 = diff1 > 1e-6
        valid2 = diff2 > 1e-6

        img1 = cv2.imread(str(cam1_img))
        img2 = cv2.imread(str(cam2_img))
        if img1 is None or img2 is None:
            raise RuntimeError("Failed to read images.")

        mask1 = (valid1.astype(np.uint8) * 255)
        mask2 = (valid2.astype(np.uint8) * 255)
        mask1_color = cv2.applyColorMap(mask1, cv2.COLORMAP_JET)
        mask2_color = cv2.applyColorMap(mask2, cv2.COLORMAP_JET)
        overlay1 = cv2.addWeighted(img1, 0.6, mask1_color, 0.4, 0)
        overlay2 = cv2.addWeighted(img2, 0.6, mask2_color, 0.4, 0)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{cam1_img.stem}_cam1.jpg"), img1)
        cv2.imwrite(str(out_dir / f"{cam2_img.stem}_cam2.jpg"), img2)
        # PE visualizations
        pe1_rgb = pe_to_rgb(pe1)
        pe2_rgb = pe_to_rgb(pe2)
        cv2.imwrite(str(out_dir / f"{cam1_img.stem}_pe_cam1.jpg"), pe1_rgb)
        cv2.imwrite(str(out_dir / f"{cam2_img.stem}_pe_cam2.jpg"), pe2_rgb)
        # PE overlap area visualization (mask applied)
        pe1_overlap = cv2.bitwise_and(pe1_rgb, pe1_rgb, mask=mask1)
        pe2_overlap = cv2.bitwise_and(pe2_rgb, pe2_rgb, mask=mask2)
        cv2.imwrite(str(out_dir / f"{cam1_img.stem}_pe_cam1_overlap.jpg"), pe1_overlap)
        cv2.imwrite(str(out_dir / f"{cam2_img.stem}_pe_cam2_overlap.jpg"), pe2_overlap)
        cv2.imwrite(str(out_dir / f"{cam1_img.stem}_cam1_mask.png"), mask1)
        cv2.imwrite(str(out_dir / f"{cam2_img.stem}_cam2_mask.png"), mask2)
        cv2.imwrite(str(out_dir / f"{cam1_img.stem}_cam1_overlap.jpg"), overlay1)
        cv2.imwrite(str(out_dir / f"{cam2_img.stem}_cam2_overlap.jpg"), overlay2)

        print(f"cam1: {cam1_img}")
        print(f"cam2: {cam2_img}")
        print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
