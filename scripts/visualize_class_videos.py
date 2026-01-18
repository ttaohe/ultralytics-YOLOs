#!/usr/bin/env python3
"""
Visualize selected classes as videos by grouping frames from a YOLO dataset split.
Default grouping uses the filename prefix before the first underscore.
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--classes", default="car,bus", help="Comma-separated class names to draw")
    parser.add_argument("--out-dir", default="runs/vis_class_videos", help="Output directory for videos")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for output videos")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, e.g., mp4v, avc1")
    parser.add_argument(
        "--group-by",
        default="prefix",
        choices=["prefix", "stem"],
        help="Grouping key: prefix (before first underscore) or full stem",
    )
    return parser.parse_args()


def group_key(stem: str, mode: str) -> str:
    if mode == "stem":
        return stem
    # prefix before first underscore
    return stem.split("_")[0] if "_" in stem else stem


def frame_index(stem: str) -> int:
    # try to parse trailing digits after last underscore, e.g., 23-1_00000001
    if "_" in stem:
        tail = stem.split("_")[-1]
        if tail.isdigit():
            return int(tail)
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def load_labels(label_path: Path):
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            rows.append((cls, x, y, w, h))
    return rows


def draw_boxes(img, labels, class_set):
    h, w = img.shape[:2]
    for cls, x, y, bw, bh in labels:
        if cls not in class_set:
            continue
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def main():
    args = parse_args()
    data = yaml.safe_load(Path(args.data).read_text())
    root = Path(data["path"])
    names = data.get("names", [])
    name_to_id = {n: i for i, n in enumerate(names)}

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_ids = {name_to_id[c] for c in classes if c in name_to_id}
    if not class_ids:
        raise ValueError(f"No valid class ids found for {classes} in names={names}")

    img_dir = root / data[args.split]
    label_dir = root / "labels" / args.split

    images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    groups = {}
    for p in images:
        key = group_key(p.stem, args.group_by)
        groups.setdefault(key, []).append(p)

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    for key, frames in groups.items():
        frames = sorted(frames, key=lambda p: frame_index(p.stem))
        first = cv2.imread(str(frames[0]))
        if first is None:
            continue
        h, w = first.shape[:2]
        out_path = out_dir / f"{key}.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
        for img_path in frames:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            label_path = label_dir / f"{img_path.stem}.txt"
            labels = load_labels(label_path)
            img = draw_boxes(img, labels, class_ids)
            writer.write(img)
        writer.release()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
