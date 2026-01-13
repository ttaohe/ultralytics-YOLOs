#!/usr/bin/env python3
"""
Multiview YOLO inference script.
Runs multiview inference with coords/PE and saves per-view predictions.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.files import increment_path
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xyxy2xywh
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules.multiview_block import MultiviewFusionBlock


def _load_pe_from_disk(im_file: str) -> torch.Tensor | None:
    """Load position encoding from disk, matching MultiviewDataset behavior."""
    im_path = Path(im_file)
    pe_paths = []
    pe_filename = im_path.stem + "_pe.npy"
    pes_dir = im_path.parents[2] / "pes" / im_path.parent.name
    pe_paths.append(pes_dir / pe_filename)
    coords_dir = im_path.parents[2] / "coords" / im_path.parent.name
    pe_paths.append(coords_dir / (im_path.stem + ".npy"))
    for pe_path in pe_paths:
        if pe_path.exists():
            pe = np.load(pe_path)
            if pe.ndim == 3 and pe.shape[2] in (3, 4):
                if pe.shape[2] == 3:
                    pe = np.pad(pe, [(0, 0), (0, 0), (0, 1)], mode="constant")
                return torch.from_numpy(pe).float()
    return None


def _generate_random_pe(h: int, w: int) -> torch.Tensor:
    """Generate random sinusoidal PE as fallback."""
    y_coords = torch.linspace(0, 2 * np.pi, h)
    x_coords = torch.linspace(0, 2 * np.pi, w)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    phase_y = torch.rand(1).item() * 2 * np.pi
    phase_x = torch.rand(1).item() * 2 * np.pi
    pe = torch.stack(
        [torch.sin(yy + phase_y), torch.cos(yy + phase_y), torch.sin(xx + phase_x), torch.cos(xx + phase_x)],
        dim=-1,
    )
    return pe


def _resize_no_pad(img: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize image to fit within imgsz while preserving aspect ratio (no padding)."""
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    r = min(r, 1.0)  # scaleup=False
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def _parse_pairs(path: Path) -> list[list[str]]:
    """Parse a whitespace-delimited pairs file (one multiview group per line)."""
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        pairs.append(parts)
    return pairs


def _save_txt(save_dir: Path, im_path: Path, preds: torch.Tensor, orig_shape: tuple[int, int], save_conf: bool):
    """Save predictions to a YOLO-format text file."""
    txt_path = save_dir / "labels" / f"{im_path.stem}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if preds is None or preds.numel() == 0:
        txt_path.write_text("")
        return
    w, h = orig_shape[1], orig_shape[0]
    xywh = xyxy2xywh(preds[:, :4])
    xywh[:, 0] /= w
    xywh[:, 1] /= h
    xywh[:, 2] /= w
    xywh[:, 3] /= h
    lines = []
    for i in range(preds.shape[0]):
        cls = int(preds[i, 5].item())
        if save_conf:
            line = f"{cls} {preds[i, 4]:.6f} {xywh[i, 0]:.6f} {xywh[i, 1]:.6f} {xywh[i, 2]:.6f} {xywh[i, 3]:.6f}"
        else:
            line = f"{cls} {xywh[i, 0]:.6f} {xywh[i, 1]:.6f} {xywh[i, 2]:.6f} {xywh[i, 3]:.6f}"
        lines.append(line)
    txt_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--pairs", type=str, required=True, help="Path to multiview pairs file (one group per line)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=1, help="Number of scenes per batch")
    parser.add_argument("--num-views", type=int, default=0, help="Number of views per scene (0 to infer)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default="runs/predict_multiview")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--allow-missing-pe", action="store_true")
    parser.add_argument("--export-attn", action="store_true", help="Export multiview attention maps")
    parser.add_argument("--export-attn-batches", type=int, default=1, help="Number of batches to export attention")
    parser.add_argument("--export-attn-max-len", type=int, default=2048, help="Max attention length to export")
    args = parser.parse_args()

    device = select_device(args.device)
    model = YOLO(args.weights).model
    model.to(device).eval()
    if args.half and device.type != "cpu":
        model.half()

    export_modules = []
    if args.export_attn:
        for m in model.modules():
            if isinstance(m, MultiviewFusionBlock):
                m.export_attn = True
                m.export_attn_max_len = args.export_attn_max_len
                export_modules.append(m)

    pairs_path = Path(args.pairs)
    pairs = _parse_pairs(pairs_path)
    if not pairs:
        raise ValueError(f"No valid pairs found in {pairs_path}")

    num_views = args.num_views or len(pairs[0])
    for idx, group in enumerate(pairs):
        if len(group) != num_views:
            raise ValueError(f"Line {idx + 1} has {len(group)} views, expected {num_views}")

    save_dir = increment_path(Path(args.save_dir) / "exp", exist_ok=False)
    LOGGER.info(f"{colorstr('bright_blue')}Saving results to {save_dir}")

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), scaleup=False)

    export_batches_left = args.export_attn_batches
    for i in range(0, len(pairs), args.batch):
        batch_groups = pairs[i : i + args.batch]
        imgs = []
        coords = []
        orig_shapes = []
        im_paths = []

        for group in batch_groups:
            view_imgs = []
            view_coords = []
            view_shapes = []
            view_paths = []
            for p in group:
                im_path = Path(p)
                img0 = cv2.imread(str(im_path))
                if img0 is None:
                    raise FileNotFoundError(f"Image not found: {im_path}")
                orig_shape = img0.shape[:2]

                img = _resize_no_pad(img0, args.imgsz)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res_h, res_w = img.shape[:2]

                pe = _load_pe_from_disk(str(im_path))
                if pe is None:
                    if not args.allow_missing_pe:
                        raise FileNotFoundError(f"PE not found for {im_path}")
                    pe = _generate_random_pe(res_h, res_w)
                if pe.shape[0] != res_h or pe.shape[1] != res_w:
                    pe_np = pe.numpy()
                    pe_np = cv2.resize(pe_np, (res_w, res_h), interpolation=cv2.INTER_LINEAR)
                    pe = torch.from_numpy(pe_np).float()

                img_with_pe = np.concatenate([img.astype(np.float32), pe.numpy()], axis=2)
                img_with_pe = letterbox(image=img_with_pe)
                img_res = img_with_pe[:, :, :3]
                pe_res = img_with_pe[:, :, 3:]

                img_t = torch.from_numpy(img_res.transpose(2, 0, 1)).float() / 255.0
                view_imgs.append(img_t)
                view_coords.append(torch.from_numpy(pe_res).float())
                view_shapes.append(orig_shape)
                view_paths.append(im_path)

            imgs.append(torch.stack(view_imgs))
            coords.append(torch.stack(view_coords).permute(0, 2, 3, 1))
            orig_shapes.append(view_shapes)
            im_paths.append(view_paths)

        imgs_t = torch.stack(imgs).to(device)
        coords_t = torch.stack(coords).to(device)
        if args.half and device.type != "cpu":
            imgs_t = imgs_t.half()
            coords_t = coords_t.half()

        b, v, c, h, w = imgs_t.shape
        preds = model(imgs_t.view(b * v, c, h, w), coords=coords_t)
        if args.export_attn and export_batches_left > 0 and export_modules:
            attn_dir = save_dir / "attn"
            attn_dir.mkdir(parents=True, exist_ok=True)
            for mi, m in enumerate(export_modules):
                attn_maps = m.last_attn_maps
                if attn_maps is None:
                    continue
                for bi in range(attn_maps.shape[0]):
                    for s in range(attn_maps.shape[1]):
                        for t in range(attn_maps.shape[2]):
                            amap = attn_maps[bi, s, t].numpy()
                            amap = amap - amap.min()
                            if amap.max() > 0:
                                amap = amap / amap.max()
                            amap = (amap * 255).astype(np.uint8)
                            heat = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
                            out = attn_dir / f"attn_m{mi}_b{i+bi}_s{s}_t{t}.png"
                            cv2.imwrite(str(out), heat)
            export_batches_left -= 1
            if export_batches_left <= 0:
                for m in export_modules:
                    m.export_attn = False
                    m.last_attn_maps = None
        preds = non_max_suppression(
            preds, conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det
        )

        for bi in range(b):
            for vi in range(v):
                idx = bi * v + vi
                pred = preds[idx]
                if pred is not None and pred.numel():
                    pred[:, :4] = scale_boxes((h, w), pred[:, :4], orig_shapes[bi][vi])
                if args.save_txt:
                    _save_txt(save_dir, im_paths[bi][vi], pred, orig_shapes[bi][vi], args.save_conf)

    LOGGER.info(f"{colorstr('bright_green')}Done.")


if __name__ == "__main__":
    main()
