#!/usr/bin/env python3
"""
按给定的 imgsz（支持多个）统计在 letterbox 缩放后（不含填充）全数据集的 YOLO 框宽高直方图。

说明：
- 仅按 YOLO 的 letterbox 缩放比例 r = min(imgsz / H, imgsz / W) 缩放框尺寸；填充不影响宽高，因此忽略 padding。
- 支持 VID 目录结构：labels/{split}/sequence/frame.txt 对应 images/{split}/sequence/frame.jpg
- 支持多个 split（train/val/test）

示例：
python scripts/yolo_resize_bbox_hist.py \
  --labels-root /home/hetao/graduate/data/VisDrone-VID-yolo/labels \
  --images-root /home/hetao/graduate/data/VisDrone-VID-yolo/images \
  --splits train val \
  --imgsz 640 1280 \
  --out-dir /home/hetao/graduate/ultralytics-YOLOs/runs/stats/resize_hist \
  --normalized --plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    # 尝试使用 Ultralytics 的 exif_size 与其处理一致
    from ultralytics.data.utils import exif_size as _exif_size  # type: ignore
except Exception:
    _exif_size = None

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def exif_size(img: Image.Image) -> Tuple[int, int]:
    """返回 (width, height)，尽量与 Ultralytics 行为保持一致。"""
    if _exif_size is not None:
        w, h = _exif_size(img)
        return int(w), int(h)
    # 回退
    w, h = img.size
    return int(w), int(h)


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


def label_to_image_path(label_path: Path, labels_root: Path, images_root: Path) -> Path:
    """将 labels 根路径下的 txt 路径映射为 images 根路径下的 jpg 路径（保持子路径一致）。"""
    rel = label_path.relative_to(labels_root)
    return (images_root / rel).with_suffix(".jpg")


def parse_label_file(txt_path: Path) -> np.ndarray:
    """
    读取 YOLO 标注文件，返回 (N, 4) 的数组 [xc, yc, w, h]，均为 0..1 归一化。
    """
    content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        return np.zeros((0, 4), dtype=np.float32)
    rows: List[List[float]] = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        # 基本有效性过滤（允许 1% 容差）
        if not (-0.01 <= xc <= 1.01 and -0.01 <= yc <= 1.01 and -0.01 <= w <= 1.01 and -0.01 <= h <= 1.01):
            continue
        rows.append([xc, yc, w, h])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def summarize_1d(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"count": 0}
    q = np.quantile(arr, [0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
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


def save_stats(out_dir: Path, imgsz: int, w_vals: np.ndarray, h_vals: np.ndarray, normalized: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"imgsz{imgsz}_{'norm' if normalized else 'px'}"
    # JSON
    stats = {"w": summarize_1d(w_vals), "h": summarize_1d(h_vals)}
    (out_dir / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    # CSV（两列）
    with (out_dir / f"stats_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "w", "h"])
        keys = sorted(set(list(stats["w"].keys()) + list(stats["h"].keys())))
        for k in keys:
            writer.writerow([k, stats["w"].get(k, ""), stats["h"].get(k, "")])


def plot_hist(
    out_dir: Path,
    imgsz: int,
    w_vals: np.ndarray,
    h_vals: np.ndarray,
    normalized: bool,
    bins: int,
    edges: List[float] | None = None,
) -> None:
    if not _HAS_MPL:
        print("[提示] 未安装 matplotlib，跳过绘图。pip install matplotlib 可启用绘图。")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"imgsz{imgsz}_{'norm' if normalized else 'px'}"
    plt.figure(figsize=(7, 4))
    hist_kwargs = {}
    if edges is None:
        hist_kwargs["bins"] = bins
        hist_kwargs["range"] = (0, 1 if normalized else imgsz)
    else:
        hist_kwargs["bins"] = edges
    if w_vals.size:
        plt.hist(w_vals, alpha=0.6, label="w", **hist_kwargs)
    if h_vals.size:
        plt.hist(h_vals, alpha=0.6, label="h", **hist_kwargs)
    plt.xlabel("size (normalized)" if normalized else f"size (pixels, imgsz={imgsz})")
    plt.ylabel("count")
    plt.title(f"Width/Height histogram @ imgsz={imgsz}" + ("" if edges is None else " (binned)"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_wh_{tag}{'_binned' if edges is not None else ''}.png", dpi=200)
    plt.close()

def plot_binned_group_pct(
    out_dir: Path,
    imgsz: int,
    edges: List[float],
    w_vals: np.ndarray,
    h_vals: np.ndarray,
    normalized: bool,
) -> None:
    if not _HAS_MPL:
        return
    if len(edges) < 2:
        return
    w_counts, _ = np.histogram(w_vals, bins=edges)
    h_counts, _ = np.histogram(h_vals, bins=edges)
    w_total = max(int(w_vals.size), 1)
    h_total = max(int(h_vals.size), 1)
    w_pct = (w_counts / w_total) * 100.0
    h_pct = (h_counts / h_total) * 100.0

    centers = (np.array(edges[:-1]) + np.array(edges[1:])) / 2.0
    widths = np.array(edges[1:]) - np.array(edges[:-1])
    # 避免过窄可见性问题
    bar_w = widths * 0.4 if widths.size else 0.4

    tag = f"imgsz{imgsz}_{'norm' if normalized else 'px'}"
    plt.figure(figsize=(max(6, min(14, 0.6 * len(centers))), 4))
    plt.bar(centers - bar_w / 2, w_pct, width=bar_w, alpha=0.7, label="w (%)")
    plt.bar(centers + bar_w / 2, h_pct, width=bar_w, alpha=0.7, label="h (%)")
    plt.ylabel("percentage (%)")
    plt.xlabel("size (normalized)" if normalized else f"size (pixels, imgsz={imgsz})")
    plt.title(f"Binned percentage by interval @ imgsz={imgsz}")
    # x 轴刻度显示为 [l, r)
    xticks = []
    for i in range(len(edges) - 1):
        xticks.append(f"[{edges[i]}, {edges[i+1]})")
    plt.xticks(centers, xticks, rotation=40, ha="right")
    plt.tight_layout()
    out_path = out_dir / f"binned_pct_hist_{tag}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="统计不同 imgsz 下 letterbox 缩放后的框宽高直方图")
    parser.add_argument("--labels-root", type=Path, required=True, help="labels 根目录（包含 train/val/test）")
    parser.add_argument("--images-root", type=Path, required=True, help="images 根目录（与 labels 子路径一致）")
    parser.add_argument("--splits", type=str, nargs="*", default=None, help="指定 split 列表：如 train val；为空则扫描全部")
    parser.add_argument("--imgsz", type=int, nargs="+", required=True, help="一个或多个 imgsz（如 640 1280）")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--normalized", action="store_true", help="输出为归一化到 [0,1] 的宽高；默认输出像素单位")
    parser.add_argument("--bins", type=int, default=80, help="直方图分箱数")
    parser.add_argument(
        "--bin-edges",
        type=str,
        default=None,
        help="可选：自定义区间边界，逗号分隔。如 --bin-edges 0,0.02,0.05,0.1,0.2,1 或 0,16,32,64,128,256,512",
    )
    parser.add_argument("--limit", type=int, default=0, help="最多处理的标签文件数（0 表示不限制）")
    parser.add_argument("--verbose", action="store_true", help="打印进度")
    args = parser.parse_args()

    label_files = gather_label_files(args.labels_root, args.splits)
    if args.limit and len(label_files) > args.limit:
        label_files = label_files[: args.limit]
    if not label_files:
        raise SystemExit(f"未找到标注文件：{args.labels_root} splits={args.splits}")

    # 每个 imgsz 分别累积 w/h
    per_imgsz_w: Dict[int, List[float]] = {s: [] for s in args.imgsz}
    per_imgsz_h: Dict[int, List[float]] = {s: [] for s in args.imgsz}

    for idx, lb_path in enumerate(label_files):
        if args.verbose and (idx % 1000 == 0):
            print(f"[{idx}/{len(label_files)}] {lb_path}")
        xywh = parse_label_file(lb_path)  # [xc,yc,w,h] 0..1
        if xywh.size == 0:
            continue
        im_path = label_to_image_path(lb_path, args.labels_root, args.images_root)
        if not im_path.exists():
            # 可能存在极少数遗漏或路径不同步
            continue
        try:
            with Image.open(im_path) as im:
                W, H = exif_size(im)  # width, height
        except Exception:
            continue
        if W <= 0 or H <= 0:
            continue

        # 原图像素尺寸的框宽高
        ws_px = xywh[:, 2] * float(W)
        hs_px = xywh[:, 3] * float(H)

        for s in args.imgsz:
            r = min(float(s) / float(H), float(s) / float(W))
            ws_resized = ws_px * r
            hs_resized = hs_px * r
            if args.normalized:
                ws_resized = ws_resized / float(s)
                hs_resized = hs_resized / float(s)
            per_imgsz_w[s].extend(ws_resized.tolist())
            per_imgsz_h[s].extend(hs_resized.tolist())

    # 保存结果
    for s in args.imgsz:
        w_vals = np.array(per_imgsz_w[s], dtype=np.float32)
        h_vals = np.array(per_imgsz_h[s], dtype=np.float32)
        save_stats(args.out_dir, s, w_vals, h_vals, args.normalized)
        # 直方图：若提供 bin-edges，则使用该自定义区间绘制直方图；否则使用等宽 bins
        plot_hist(
            args.out_dir,
            s,
            w_vals,
            h_vals,
            args.normalized,
            bins=args.bins,
            edges=None,  # 默认先输出一个等宽直方图
        )

        # 区间统计（若指定 bin-edges）
        if args.bin_edges:
            try:
                edges = [float(x) for x in args.bin_edges.split(",") if x.strip() != ""]
                edges = sorted(list(set(edges)))
            except Exception:
                edges = []
            if len(edges) >= 2:
                # 对 w/h 分别做区间统计
                w_counts, _ = np.histogram(w_vals, bins=edges)
                h_counts, _ = np.histogram(h_vals, bins=edges)
                w_total = max(int(w_vals.size), 1)
                h_total = max(int(h_vals.size), 1)
                out_csv = args.out_dir / f"binned_imgsz{s}_{'norm' if args.normalized else 'px'}.csv"
                with out_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["bin_left", "bin_right", "w_count", "w_pct", "h_count", "h_pct"])
                    for i in range(len(edges) - 1):
                        wl = int(w_counts[i])
                        hl = int(h_counts[i])
                        writer.writerow(
                            [
                                edges[i],
                                edges[i + 1],
                                wl,
                                round(100.0 * wl / w_total, 4),
                                hl,
                                round(100.0 * hl / h_total, 4),
                            ]
                        )
                # 使用自定义区间再绘制一张“分区间直方图”（计数），以及“分区间占比柱状图”
                plot_hist(
                    args.out_dir,
                    s,
                    w_vals,
                    h_vals,
                    args.normalized,
                    bins=args.bins,
                    edges=edges,
                )
                plot_binned_group_pct(args.out_dir, s, edges, w_vals, h_vals, args.normalized)
            else:
                print("[提示] --bin-edges 至少需要两个数。已忽略区间统计。")

    print(f"完成：共处理 {len(label_files)} 个标签文件，输出位于 {args.out_dir}")


if __name__ == "__main__":
    main()


