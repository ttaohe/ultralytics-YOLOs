#!/usr/bin/env python3
"""
统计 YOLO 标注的中心点(xc, yc)与宽高(w, h)分布（归一化 0..1）

特性：
- 递归扫描 labels/{split}/ 下所有 .txt（兼容 VID 序列目录）
- 输出数值统计（均值、标准差、分位数）到 JSON/CSV
- 可选绘图：中心点热力图、w/h 直方图、w-h 二维直方图

示例：
python scripts/yolo_bbox_stats.py \
  --labels-root /home/hetao/graduate/data/VisDrone-VID-yolo/labels \
  --splits train val \
  --out-dir /home/hetao/graduate/ultralytics-YOLOs/runs/stats/visdrone_vid \
  --plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def gather_label_files(labels_root: Path, splits: Sequence[str] | None) -> List[Path]:
    files: List[Path] = []
    if splits:
        for sp in splits:
            p = labels_root / sp
            if p.exists():
                files.extend(sorted(p.rglob("*.txt")))
    else:
        if labels_root.exists():
            files.extend(sorted(labels_root.rglob("*.txt")))
    return files


def parse_label_file(txt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取单个 YOLO 标注文件，返回 centers(N,2), sizes(N,2)。若无有效框返回空数组。"""
    centers: List[Tuple[float, float]] = []
    sizes: List[Tuple[float, float]] = []
    content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            # YOLO: cls xc yc w h
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        # 基本有效性过滤（允许 1% 容差）
        if not (-0.01 <= xc <= 1.01 and -0.01 <= yc <= 1.01 and -0.01 <= w <= 1.01 and -0.01 <= h <= 1.01):
            continue
        centers.append((xc, yc))
        sizes.append((w, h))
    if not centers:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.array(centers, dtype=np.float32), np.array(sizes, dtype=np.float32)


def summarize(arr: np.ndarray, name: str) -> Dict[str, float]:
    """对 1D 数组做统计。"""
    if arr.size == 0:
        return {
            "name": name,
            "count": 0,
        }
    q = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "name": name,
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(q[0]),
        "p1": float(q[1]),
        "p5": float(q[2]),
        "p10": float(q[3]),
        "p50": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p99": float(q[7]),
        "max": float(q[8]),
    }


def save_stats_json_csv(out_dir: Path, stats: Dict[str, Dict[str, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_dir / "bbox_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # CSV（宽表）
    keys = sorted({k for v in stats.values() for k in v.keys()})
    with (out_dir / "bbox_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + list(stats.keys()))
        for k in keys:
            row = [k]
            for name in stats.keys():
                row.append(stats[name].get(k, ""))
            writer.writerow(row)


def plot_distributions(out_dir: Path, centers: np.ndarray, sizes: np.ndarray, bins: int = 100) -> None:
    if not _HAS_MPL:
        print("[提示] 未安装 matplotlib，跳过绘图。可通过 pip install matplotlib 启用绘图。")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 中心点热力图
    if centers.size:
        plt.figure(figsize=(6, 5))
        plt.hist2d(centers[:, 0], centers[:, 1], bins=bins, range=[[0, 1], [0, 1]], cmap="magma")
        plt.colorbar(label="count")
        plt.xlabel("x_center")
        plt.ylabel("y_center")
        plt.title("Centers heatmap (normalized)")
        plt.tight_layout()
        plt.savefig(out_dir / "centers_heatmap.png", dpi=200)
        plt.close()

    # w/h 直方图
    if sizes.size:
        plt.figure(figsize=(6, 4))
        plt.hist(sizes[:, 0], bins=bins, range=(0, 1), alpha=0.7, label="w")
        plt.hist(sizes[:, 1], bins=bins, range=(0, 1), alpha=0.7, label="h")
        plt.xlabel("normalized size")
        plt.ylabel("count")
        plt.title("Width/Height hist")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "wh_hist.png", dpi=200)
        plt.close()

        # w-h 二维直方图
        plt.figure(figsize=(5, 5))
        plt.hist2d(sizes[:, 0], sizes[:, 1], bins=bins, range=[[0, 1], [0, 1]], cmap="viridis")
        plt.colorbar(label="count")
        plt.xlabel("w")
        plt.ylabel("h")
        plt.title("W-H heatmap (normalized)")
        plt.tight_layout()
        plt.savefig(out_dir / "wh_heatmap.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 YOLO 标注的中心点与宽高分布")
    parser.add_argument("--labels-root", type=Path, required=True, help="labels 根目录（包含 train/val/test 子目录，内部递归扫描）")
    parser.add_argument("--splits", type=str, nargs="*", default=None, help="指定 split 列表，如 train val test；不填则扫描全部")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录（保存 JSON/CSV 及可选图像）")
    parser.add_argument("--plots", action="store_true", help="是否生成可视化图（需要 matplotlib）")
    parser.add_argument("--limit", type=int, default=0, help="最多读取的标注文件数（0 为不限制）")
    args = parser.parse_args()

    label_files = gather_label_files(args.labels_root, args.splits)
    if args.limit and len(label_files) > args.limit:
        label_files = label_files[: args.limit]
    if not label_files:
        raise SystemExit(f"未找到标注文件：{args.labels_root} splits={args.splits}")

    all_centers: List[np.ndarray] = []
    all_sizes: List[np.ndarray] = []
    for i, f in enumerate(label_files):
        c, s = parse_label_file(f)
        if c.size:
            all_centers.append(c)
        if s.size:
            all_sizes.append(s)

    centers = np.concatenate(all_centers, axis=0) if all_centers else np.zeros((0, 2), dtype=np.float32)
    sizes = np.concatenate(all_sizes, axis=0) if all_sizes else np.zeros((0, 2), dtype=np.float32)

    stats = {
        "xc": summarize(centers[:, 0] if centers.size else np.array([], dtype=np.float32), "xc"),
        "yc": summarize(centers[:, 1] if centers.size else np.array([], dtype=np.float32), "yc"),
        "w": summarize(sizes[:, 0] if sizes.size else np.array([], dtype=np.float32), "w"),
        "h": summarize(sizes[:, 1] if sizes.size else np.array([], dtype=np.float32), "h"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_stats_json_csv(args.out_dir, stats)

    if args.plots:
        plot_distributions(args.out_dir, centers, sizes)

    print(f"完成：共 {len(label_files)} 个标签文件，{int(centers.shape[0])} 个目标。输出位于 {args.out_dir}")


if __name__ == "__main__":
    main()


