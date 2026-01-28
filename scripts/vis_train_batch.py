#!/usr/bin/env python3
"""
Visualize a few training batches (after random crop/letterbox) for multiview.
Saves images + a text file with resized shapes to confirm crop behavior.
"""

import argparse
from pathlib import Path

import torch
from ultralytics.utils import YAML, colorstr, LOGGER
from ultralytics.utils.plotting import plot_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="runs/vis_batches")
    parser.add_argument("--random-crop-size", type=int, default=0)
    parser.add_argument("--random-crop-prob", type=float, default=0.0)
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
            "random_crop_size": args.random_crop_size,
            "random_crop_prob": args.random_crop_prob,
        }
    )
    trainer.data = data_dict
    # Ensure random crop params are set on args for transform building
    trainer.args.random_crop_size = args.random_crop_size
    trainer.args.random_crop_prob = args.random_crop_prob
    # Ensure stride is available for build_dataset (trainer.stride may be set during training setup)
    if not hasattr(trainer, "stride"):
        trainer.stride = torch.tensor([32])

    train_path = data_dict.get("train")
    dataloader = trainer.get_dataloader(train_path, args.batch, rank=-1, mode="train")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(colorstr("bright_green") + f"Saving batches to {save_dir}")

    for i, batch in enumerate(dataloader):
        if i >= args.num_batches:
            break
        # plot_images expects img in [0,1] or [0,255]; use as-is
        plot_path = save_dir / f"train_batch{i}.jpg"
        labels = {
            "img": batch["img"],
            "cls": batch["cls"],
            "bboxes": batch["bboxes"],
            "batch_idx": batch["batch_idx"],
        }
        plot_images(
            labels=labels,
            paths=batch["im_file"],
            fname=plot_path,
        )
        # Save per-image PE overlay and original crop window
        coords = batch.get("coords")
        crop_windows = batch.get("crop_window_ori", [])
        if coords is not None:
            n, v = coords.shape[0], coords.shape[1]
            imgs = batch["img"]
            for si in range(n):
                for vi in range(v):
                    idx = si * v + vi
                    img = imgs[idx].cpu().float()
                    if img.max() <= 1:
                        img = img * 255.0
                    img_np = img.permute(1, 2, 0).numpy().astype("uint8")
                    pe = coords[si, vi].cpu().numpy()
                    pe_rgb = pe[..., :3]
                    pmin, pmax = pe_rgb.min(), pe_rgb.max()
                    if pmax > pmin:
                        pe_rgb = (pe_rgb - pmin) / (pmax - pmin)
                    pe_rgb = (pe_rgb * 255.0).astype("uint8")
                    overlay = (0.6 * img_np + 0.4 * pe_rgb).astype("uint8")
                    pe_path = save_dir / f"train_batch{i}_img{idx}_pe_overlay.jpg"
                    from PIL import Image, ImageDraw
                    Image.fromarray(overlay).save(pe_path)

                    # Draw crop window on original image if available
                    if crop_windows:
                        im_path = batch["im_file"][idx]
                        try:
                            from ultralytics.utils.patches import imread
                            orig = imread(im_path)
                            if orig is not None:
                                x0, y0, x1, y1 = crop_windows[idx]
                                orig_pil = Image.fromarray(orig[..., ::-1]) if orig.shape[2] == 3 else Image.fromarray(orig)
                                draw = ImageDraw.Draw(orig_pil)
                                if x1 > x0 and y1 > y0:
                                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                                crop_path = save_dir / f"train_batch{i}_img{idx}_crop_window.jpg"
                                orig_pil.save(crop_path)
                        except Exception:
                            pass
        # save shapes for debugging
        shapes_path = save_dir / f"train_batch{i}_shapes.txt"
        with open(shapes_path, "w", encoding="utf-8") as f:
            f.write(f"imgsz={args.imgsz}\n")
            f.write("resized_shape:\n")
            for p, s in zip(batch["im_file"], batch.get("resized_shape", [])):
                f.write(f"{Path(p).name}\t{s}\n")
        LOGGER.info(f"Saved {plot_path} and {shapes_path}")


if __name__ == "__main__":
    main()
