#!/usr/bin/env python3
"""
Visualize a few validation batches with pixel-synchronized sliding-window tiles.
Saves per-sample tile images and PE overlays into separate folders.
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics.utils import YAML, colorstr, LOGGER


def _tile_positions(h, w, tile, stride):
    if h <= tile and w <= tile:
        return [(0, 0, h, w)]
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    return [(y, x, y + tile, x + tile) for y in ys for x in xs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--val-tile-size", type=int, default=640)
    parser.add_argument("--val-tile-stride", type=int, default=480)
    parser.add_argument("--val-tile-original", action="store_true", help="Use original-resolution tiles")
    parser.add_argument("--save-dir", type=str, default="runs/vis_val_tiles")
    args = parser.parse_args()

    from ultralytics.models.yolo.multiview.train import MultiviewTrainer

    data_dict = YAML.load(args.data)
    trainer = MultiviewTrainer(
        overrides={
            "model": args.model,
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
        }
    )
    trainer.data = data_dict
    if not hasattr(trainer, "stride"):
        trainer.stride = torch.tensor([32])

    val_path = data_dict.get("val") or data_dict.get("test")
    dataloader = trainer.get_dataloader(val_path, args.batch, rank=-1, mode="val")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(colorstr("bright_green") + f"Saving val tiles to {save_dir}")

    dataset = getattr(dataloader, "dataset", None)
    for bi, batch in enumerate(dataloader):
        if bi >= args.num_batches:
            break
        imgs = batch["img"]  # (N*V, C, H, W)
        coords = batch.get("coords")  # (N, V, H, W, 4)
        im_files = batch.get("im_file", [])

        n = coords.shape[0] if coords is not None else (imgs.shape[0] // 2)
        v = coords.shape[1] if coords is not None else 2
        _, _, h, w = imgs.shape
        tiles = _tile_positions(h, w, args.val_tile_size, args.val_tile_stride)

        for si in range(n):
            scene_dir = save_dir / f"batch{bi}_scene{si}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            # Optional original-resolution inputs
            ori_imgs = None
            ori_coords = None
            if args.val_tile_original:
                ori_imgs = []
                ori_coords = []
                target_h = None
                target_w = None
                for vi in range(v):
                    idx = si * v + vi
                    im_path = im_files[idx]
                    from ultralytics.utils.patches import imread

                    im = imread(im_path)
                    if im is None:
                        continue
                    if im.ndim == 2:
                        im = np.repeat(im[..., None], 3, axis=2)
                    if im.shape[2] == 3:
                        im = im[..., ::-1]
                    if target_h is None:
                        target_h, target_w = im.shape[:2]
                    if im.shape[:2] != (target_h, target_w):
                        im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    ori_imgs.append(im)
                    if dataset is not None and hasattr(dataset, "load_coords"):
                        pe = dataset.load_coords(im_path, (target_h, target_w))
                        ori_coords.append(pe.numpy())
                if ori_imgs:
                    h, w = ori_imgs[0].shape[:2]
                    tiles = _tile_positions(h, w, args.val_tile_size, args.val_tile_stride)

            # Visualize tile windows on original images (or batch images if not using original)
            for vi in range(v):
                if args.val_tile_original and ori_imgs:
                    canvas = ori_imgs[vi].copy()
                else:
                    img = imgs[si * v + vi].cpu().float()
                    if img.max() <= 1:
                        img = img * 255.0
                    canvas = img.permute(1, 2, 0).numpy().astype("uint8")
                for (y0, x0, y1, x1) in tiles:
                    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
                stem = Path(im_files[si * v + vi]).stem if im_files else f"img{si * v + vi}"
                out_path = scene_dir / f"tile_windows_view{vi}_{stem}.jpg"
                cv2.imwrite(str(out_path), canvas)

            for ti, (y0, x0, y1, x1) in enumerate(tiles):
                for vi in range(v):
                    idx = si * v + vi
                    if args.val_tile_original and ori_imgs:
                        img_np = ori_imgs[vi].copy()
                    else:
                        img = imgs[idx].cpu().float()
                        if img.max() <= 1:
                            img = img * 255.0
                        img_np = img.permute(1, 2, 0).numpy().astype("uint8")
                    crop = img_np[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    stem = Path(im_files[idx]).stem if im_files else f"img{idx}"
                    base = scene_dir / f"tile{ti:03d}_view{vi}_{stem}"
                    try:
                        cv2.imwrite(str(base.with_suffix(".jpg")), crop)
                    except Exception:
                        from PIL import Image

                        Image.fromarray(crop).save(base.with_suffix(".jpg"))

                    if coords is None and not (args.val_tile_original and ori_coords):
                        continue
                    if args.val_tile_original and ori_coords:
                        pe = ori_coords[vi]
                    else:
                        pe = coords[si, vi].cpu().numpy()
                    pe_crop = pe[y0:y1, x0:x1, :]
                    pe_vis = pe_crop[..., :3]
                    pmin, pmax = pe_vis.min(), pe_vis.max()
                    if pmax > pmin:
                        pe_vis = (pe_vis - pmin) / (pmax - pmin)
                    pe_vis = (pe_vis * 255.0).astype("uint8")
                    overlay = (0.6 * crop + 0.4 * pe_vis).astype("uint8")
                    try:
                        cv2.imwrite(str(base.with_name(base.name + "_pe_overlay.jpg")), overlay)
                    except Exception:
                        from PIL import Image

                        Image.fromarray(overlay).save(base.with_name(base.name + "_pe_overlay.jpg"))

        LOGGER.info(f"Saved batch {bi} tiles to {save_dir}")


if __name__ == "__main__":
    main()
