#!/usr/bin/env python3
"""
VisDrone -> YOLO 转换与类别校验脚本

功能：
- 将 VisDrone2019-DET 与 VisDrone2019-VID 的标注转为 YOLO txt（xc yc w h 范围[0,1]）
  - DET：images/{split}, labels/{split}
  - VID：支持序列目录结构，images/{split}/{sequence}/{frame}.jpg 与对应 labels/{split}/{sequence}/{frame}.txt
- 可选择复制或移动图片
- 在转换前后校验数据集类别是否与 VisDrone.yaml 的 names 配置一致

用法示例：
python scripts/visdrone_convert_and_validate.py \
  --source-root /path/to/unzipped \
  --out-root /path/to/VisDrone \
  --yaml ultralytics/cfg/datasets/VisDrone.yaml \
  --copy  # 或 --move

说明：
- 期望的源目录结构为（自动识别 DET 或 VID）：
  - DET：
    {source-root}/VisDrone2019-DET-train/{images,annotations}
    {source-root}/VisDrone2019-DET-val/{images,annotations}
    {source-root}/VisDrone2019-DET-test-dev/{images,annotations}
  - VID：
    {source-root}/VisDrone2019-VID-train/{sequences,annotations}
    {source-root}/VisDrone2019-VID-val/{sequences,annotations}
    {source-root}/VisDrone2019-VID-test-dev/{sequences,annotations}
  若存在其他 split 也会按同名策略尝试处理。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image
import yaml

try:
    from tqdm import tqdm
except Exception:  # 最小依赖降级（无 tqdm 时降级为原生迭代）
    def tqdm(x, **kwargs):  # type: ignore
        return x


SPLIT_MAP = {
    # DET
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
    # VID
    "VisDrone2019-VID-train": "train",
    "VisDrone2019-VID-val": "val",
    "VisDrone2019-VID-test-dev": "test",
}


def load_yaml_names(yaml_path: Path) -> Dict[int, str]:
    """从 VisDrone.yaml 读取 names 配置，返回 {class_id: name}，class_id 从 0 开始。"""
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    names = data.get("names")
    if names is None:
        raise ValueError(f"YAML 中未找到 'names' 字段: {yaml_path}")

    if isinstance(names, dict):
        parsed = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        parsed = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError("'names' 字段应为 list 或 dict")

    if not parsed:
        raise ValueError("'names' 为空")
    # 基本连续性检查（0..N-1）
    expected = set(range(0, len(parsed)))
    if set(parsed.keys()) != expected:
        raise ValueError(f"YAML 类别索引不连续或起点非 0，期望 {sorted(expected)}，实际 {sorted(parsed.keys())}")
    return parsed


def scan_visdrone_categories(annotations_dirs: Iterable[Path]) -> Set[int]:
    """扫描多个 annotations 目录，提取实际出现的原始类别 ID（VisDrone 原始为 1..10，0 表示 ignore）。

    同时兼容：
    - DET 行格式: x,y,w,h,score,object_category
    - VID 行格式: frame_index,target_id,x,y,w,h,score,object_category,truncation,occlusion
    """
    found: Set[int] = set()
    for ann_dir in annotations_dirs:
        if not ann_dir.exists():
            continue
        for txt in ann_dir.glob("*.txt"):
            content = txt.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            for line in content.splitlines():
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                # DET
                if len(parts) >= 6 and len(parts) < 10:
                    if parts[4].strip() == "0":
                        continue
                    try:
                        cls_raw = int(parts[5].strip())
                    except ValueError:
                        continue
                else:
                    # VID
                    if parts[6].strip() == "0":
                        continue
                    try:
                        cls_raw = int(parts[7].strip())
                    except ValueError:
                        continue
                found.add(cls_raw)
    return found


def validate_categories(yaml_names: Dict[int, str], found_raw_ids: Set[int]) -> None:
    """校验数据集中出现的原始类别 ID 是否与 YAML 配置一致。

    VisDrone 原始类别取值应在 [1..len(yaml_names)]，转换后将减 1 映射到 [0..N-1]。
    """
    if not found_raw_ids:
        print("[警告] 未在 annotations 中发现任何有效目标（可能是仅 test 集或标注为空）")
        return

    n_yaml = len(yaml_names)
    expected_raw = set(range(1, n_yaml + 1))
    unexpected = sorted([x for x in found_raw_ids if x not in expected_raw])
    missing = sorted([x for x in expected_raw if x not in found_raw_ids])

    print(f"YAML 类别数: {n_yaml}")
    print(f"数据集中发现的原始类别 ID: {sorted(found_raw_ids)}")
    if unexpected:
        print(f"[错误] 数据集中存在 YAML 未定义的原始类别: {unexpected}")
    if missing:
        print(f"[提示] 数据集中未出现的原始类别: {missing}")

    if unexpected:
        raise ValueError("数据集类别与 YAML 定义不一致（存在越界/未知类）。")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_move_images(source_images_dir: Path, target_images_dir: Path, move: bool) -> None:
    """复制或移动图片，支持：
    - 扁平 images 目录（*.jpg）
    - 序列目录结构（递归复制 sequences/*/*.jpg 或 images/*/*.jpg）
    """
    ensure_dir(target_images_dir)

    # 如果包含子目录，则递归拷贝，保留相对路径结构
    subdirs = [p for p in source_images_dir.iterdir() if p.is_dir()] if source_images_dir.exists() else []
    if subdirs:
        jpgs = list(source_images_dir.rglob("*.jpg"))
        if not jpgs:
            print(f"[警告] 未在 {source_images_dir}（含子目录）找到 .jpg 图片")
        for img in tqdm(jpgs, desc=f"{'Moving' if move else 'Copying'} images (recursive) -> {target_images_dir}"):
            rel = img.relative_to(source_images_dir)
            dst = target_images_dir / rel
            ensure_dir(dst.parent)
            if move:
                img.replace(dst)
            else:
                shutil.copy2(img, dst)
        return

    # 否则按扁平目录处理
    imgs = list(source_images_dir.glob("*.jpg"))
    if not imgs:
        print(f"[警告] 未在 {source_images_dir} 找到 .jpg 图片")
    for img in tqdm(imgs, desc=f"{'Moving' if move else 'Copying'} images -> {target_images_dir}"):
        dst = target_images_dir / img.name
        if move:
            img.replace(dst)
        else:
            shutil.copy2(img, dst)


def _find_frame_image(seq_dir: Path, frame_idx: int) -> Optional[Path]:
    """根据帧号在序列目录中查找图片，尝试 7/6/5 位零填充与非填充命名。"""
    candidates = [
        seq_dir / f"{frame_idx:07d}.jpg",
        seq_dir / f"{frame_idx:06d}.jpg",
        seq_dir / f"{frame_idx:05d}.jpg",
        seq_dir / f"{frame_idx}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    # 宽松匹配：包含 frame_idx 的文件名
    hits = sorted(seq_dir.glob(f"*{frame_idx}*.jpg"))
    return hits[0] if hits else None


def convert_split(source_dir: Path, out_root: Path, split_name: str) -> int:
    """将单个 split 转换为 YOLO 标注，返回转换的标注文件数。

    同时兼容：
    - DET：annotations 下每图一 txt，images 下为平铺图片
    - VID：annotations 下每序列一 txt，sequences 或 images 下为子目录，每帧一图
    """
    source_images_dir = source_dir / "images"
    source_sequences_dir = source_dir / "sequences"
    source_ann_dir = source_dir / "annotations"
    target_images_dir = out_root / "images" / split_name
    target_labels_dir = out_root / "labels" / split_name

    ensure_dir(target_labels_dir)

    count = 0
    ann_files = list(source_ann_dir.glob("*.txt"))
    for ann_file in tqdm(ann_files, desc=f"Converting {split_name}"):
        content = ann_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue

        # 判断是 DET 还是 VID：依据每行字段数量
        first_line = content.splitlines()[0]
        parts0 = first_line.split(",")
        is_vid = len(parts0) >= 10

        if not is_vid:
            # ---- DET 模式 ----
            image_name = ann_file.with_suffix(".jpg").name
            image_path = source_images_dir / image_name
            if not image_path.exists():
                print(f"[跳过] 找不到对应图片: {image_path}")
                continue
            with Image.open(image_path) as im:
                width, height = im.size
            dw, dh = 1.0 / width, 1.0 / height

            yolo_lines: List[str] = []
            for line in content.splitlines():
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                if parts[4].strip() == "0":
                    continue
                try:
                    x, y, w, h = [int(float(v)) for v in parts[0:4]]
                    cls_raw = int(parts[5].strip())
                except ValueError:
                    continue
                cls = cls_raw - 1
                x_center = (x + w / 2.0) * dw
                y_center = (y + h / 2.0) * dh
                w_norm = w * dw
                h_norm = h * dh
                yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            (target_labels_dir / ann_file.name).write_text("".join(yolo_lines), encoding="utf-8")
            count += 1
            continue

        # ---- VID 模式 ----
        # 推断序列目录：优先 sequences，其次 images（包含子目录）
        seq_root = source_sequences_dir if source_sequences_dir.exists() else source_images_dir
        seq_name = ann_file.stem
        seq_dir = seq_root / seq_name
        if not seq_dir.exists():
            print(f"[跳过] 找不到序列目录: {seq_dir}")
            continue

        out_seq_labels = target_labels_dir / seq_name
        ensure_dir(out_seq_labels)

        # 聚合到每帧写一个 yolo 标签文件
        per_frame: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
        for line in content.splitlines():
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                frame_idx = int(float(parts[0].strip()))
                score = parts[6].strip()
                if score == "0":
                    continue
                x = int(float(parts[2].strip()))
                y = int(float(parts[3].strip()))
                w = int(float(parts[4].strip()))
                h = int(float(parts[5].strip()))
                cls_raw = int(parts[7].strip())
            except ValueError:
                continue
            cls = cls_raw - 1
            per_frame.setdefault(frame_idx, []).append((cls, x, y, w, h))

        # 为每一帧生成标签文件
        for frame_idx, objs in sorted(per_frame.items()):
            img_path = _find_frame_image(seq_dir, frame_idx)
            if img_path is None or not img_path.exists():
                print(f"[跳过] 找不到帧图像: {seq_dir} frame={frame_idx}")
                continue
            with Image.open(img_path) as im:
                width, height = im.size
            dw, dh = 1.0 / width, 1.0 / height

            lines: List[str] = []
            for cls, x, y, w, h in objs:
                x_center = (x + w / 2.0) * dw
                y_center = (y + h / 2.0) * dh
                w_norm = w * dw
                h_norm = h * dh
                lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            (out_seq_labels / img_path.with_suffix('.txt').name).write_text("".join(lines), encoding="utf-8")
            count += 1

    return count


def detect_available_splits(source_root: Path) -> List[Tuple[str, Path, str]]:
    """检测可用的 split 目录，返回 (folder_name, folder_path, split_key)。"""
    results: List[Tuple[str, Path, str]] = []
    for folder_name, split_key in SPLIT_MAP.items():
        folder_path = source_root / folder_name
        if (folder_path / "annotations").exists():
            results.append((folder_name, folder_path, split_key))
    # 同时兼容非标准命名（比如只有 train/val/test 的目录名），做一次宽松匹配
    for p in source_root.iterdir() if source_root.exists() else []:
        if not p.is_dir():
            continue
        name = p.name.lower()
        if (p / "annotations").exists():
            if "train" in name and ("visdrone2019-det" not in name and "visdrone2019-vid" not in name):
                results.append((p.name, p, "train"))
            elif "val" in name and ("visdrone2019-det" not in name and "visdrone2019-vid" not in name):
                results.append((p.name, p, "val"))
            elif "test" in name and ("visdrone2019-det" not in name and "visdrone2019-vid" not in name):
                results.append((p.name, p, "test"))
    # 去重（以路径为主）
    unique: Dict[Path, Tuple[str, Path, str]] = {}
    for item in results:
        unique[item[1]] = item
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDrone -> YOLO 转换与校验")
    parser.add_argument("--source-root", type=Path, required=True, help="解压后的 VisDrone 源根目录（包含 VisDrone2019-DET-xxx 或 VisDrone2019-VID-xxx）")
    parser.add_argument("--out-root", type=Path, required=True, help="输出数据集根目录（将生成 images/ 与 labels/ 结构）")
    parser.add_argument("--yaml", type=Path, default=Path("ultralytics/cfg/datasets/VisDrone.yaml"), help="用于校验的 VisDrone.yaml 路径")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--copy", action="store_true", help="复制图片到输出目录（默认）")
    group.add_argument("--move", action="store_true", help="移动图片到输出目录（会修改源目录）")
    parser.add_argument("--validate-only", action="store_true", help="仅做类别校验，不执行转换")

    args = parser.parse_args()

    yaml_names = load_yaml_names(args.yaml)
    print(f"从 YAML 读取到 {len(yaml_names)} 个类别：{[yaml_names[i] for i in range(len(yaml_names))]}")

    available = detect_available_splits(args.source_root)
    if not available:
        raise SystemExit(f"未在 {args.source_root} 下检测到有效的 VisDrone split 目录（需包含 annotations/）")

    # 扫描类别
    ann_dirs = [p / "annotations" for _, p, _ in available]
    found_raw_ids = scan_visdrone_categories(ann_dirs)
    validate_categories(yaml_names, found_raw_ids)

    if args.validate_only:
        print("已完成类别校验（validate-only 模式），未执行转换。")
        return

    # 复制/移动图片，并转换标注
    for folder_name, folder_path, split_key in available:
        # 优先 sequences（针对 VID），否则 images
        source_images_dir = folder_path / "sequences"
        if not source_images_dir.exists():
            source_images_dir = folder_path / "images"
        target_images_dir = args.out_root / "images" / split_key
        if source_images_dir.exists():
            copy_or_move_images(source_images_dir, target_images_dir, move=args.move)
        else:
            print(f"[警告] 未找到图片目录: {source_images_dir}")

        converted = convert_split(folder_path, args.out_root, split_key)
        print(f"{folder_name} -> {split_key}: 写出 {converted} 个 YOLO 标注文件")

    print(f"完成。输出位于: {args.out_root}")


if __name__ == "__main__":
    main()


