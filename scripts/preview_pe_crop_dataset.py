#!/usr/bin/env python3
"""
Preview PE-based consistent crops for a small set of multiview samples.
"""

from pathlib import Path
import argparse
import random
import numpy as np
import cv2


def pe_to_xy_norm(pe):
    y = np.arctan2(pe[..., 0], pe[..., 1]) / (2 * np.pi)
    x = np.arctan2(pe[..., 2], pe[..., 3]) / (2 * np.pi)
    y = (y + 1.0) % 1.0
    x = (x + 1.0) % 1.0
    return x, y


def load_pe(pe_path: Path):
    if pe_path.suffix == ".npz":
        return np.load(pe_path)["pe"]
    return np.load(pe_path)


def find_pe(im_path: Path):
    pe_name_npy = im_path.stem + "_pe.npy"
    pe_name_npz = im_path.stem + "_pe.npz"
    base = im_path.parents[2]
    split = im_path.parent.name
    # Prefer npz, then fp16, then fp32
    candidates = [
        base / "pes_npz" / split / pe_name_npz,
        base / "pes_fp16" / split / pe_name_npy,
        base / "pes" / split / pe_name_npy,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def match_camera2_path(cam1_path: Path):
    name = cam1_path.name
    if "-1_" in name:
        name = name.replace("-1_", "-2_")
    cam2_path = cam1_path.parents[2] / "camera2" / "images" / cam1_path.parent.name / name
    if cam2_path.exists():
        return cam2_path
    return None


def preview_one(img1_path: Path, img2_path: Path, out_dir: Path, crop_size: int, tries: int, eps: float):
    pe1_path = find_pe(img1_path)
    pe2_path = find_pe(img2_path)
    if pe1_path is None or pe2_path is None:
        return False, "missing PE"

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return False, "missing image"

    pe1 = load_pe(pe1_path)
    pe2 = load_pe(pe2_path)
    if pe1.shape[:2] != img1.shape[:2] or pe2.shape[:2] != img2.shape[:2]:
        return False, "shape mismatch"

    const1 = np.array([1, 0, 1, 0], np.float32)
    const2 = np.array([-1, 0, -1, 0], np.float32)
    valid1 = (np.abs(pe1 - const1).max(axis=-1) > eps)
    valid2 = (np.abs(pe2 - const2).max(axis=-1) > eps)

    x1, y1 = pe_to_xy_norm(pe1)
    x2, y2 = pe_to_xy_norm(pe2)

    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    ok = False
    msg = "no crop"

    for _ in range(tries):
        ys, xs = np.where(valid1)
        if len(xs) == 0:
            msg = "no valid1"
            break
        idx = np.random.randint(0, len(xs))
        cx, cy = xs[idx], ys[idx]
        x0 = max(0, cx - crop_size // 2)
        y0 = max(0, cy - crop_size // 2)
        x1p = min(W1, x0 + crop_size)
        y1p = min(H1, y0 + crop_size)
        if (x1p - x0) < crop_size or (y1p - y0) < crop_size:
            continue

        x_min, x_max = x1[y0:y1p, x0:x1p].min(), x1[y0:y1p, x0:x1p].max()
        y_min, y_max = y1[y0:y1p, x0:x1p].min(), y1[y0:y1p, x0:x1p].max()

        mask2 = (x2 >= x_min) & (x2 <= x_max) & (y2 >= y_min) & (y2 <= y_max) & valid2
        ys2, xs2 = np.where(mask2)
        if len(xs2) < 50:
            continue

        x2_0, x2_1 = xs2.min(), xs2.max()
        y2_0, y2_1 = ys2.min(), ys2.max()
        if x2_1 <= x2_0 or y2_1 <= y2_0:
            continue

        o1 = img1.copy()
        o2 = img2.copy()
        cv2.rectangle(o1, (x0, y0), (x1p, y1p), (0, 255, 0), 2)
        cv2.rectangle(o2, (x2_0, y2_1), (x2_1, y2_0), (0, 255, 0), 2)

        crop1 = img1[y0:y1p, x0:x1p].copy()
        crop2 = img2[y2_0:y2_1, x2_0:x2_1].copy()
        mask2_img = (mask2.astype(np.uint8) * 255)

        stem = img1_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay1.png"), o1)
        cv2.imwrite(str(out_dir / f"{stem}_overlay2.png"), o2)
        cv2.imwrite(str(out_dir / f"{stem}_crop1.png"), crop1)
        cv2.imwrite(str(out_dir / f"{stem}_crop2.png"), crop2)
        cv2.imwrite(str(out_dir / f"{stem}_mask2.png"), mask2_img)
        ok = True
        msg = "ok"
        break

    return ok, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/pe_crop_preview")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    base = Path(args.base)
    cam1_dir = base / "camera1" / "images" / args.split
    if not cam1_dir.exists():
        raise FileNotFoundError(cam1_dir)
    imgs = sorted([p for p in cam1_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    rnd = random.Random(args.seed)
    if args.count > len(imgs):
        args.count = len(imgs)
    samples = rnd.sample(imgs, args.count)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in samples:
        cam2 = match_camera2_path(p)
        if cam2 is None:
            print(f"{p.name}: missing camera2")
            continue
        ok, msg = preview_one(p, cam2, out_dir, args.crop, args.tries, args.eps)
        print(f"{p.name}: {msg}")


if __name__ == "__main__":
    main()
