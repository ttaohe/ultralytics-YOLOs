
import cv2
import torch
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train-video-sparse-modelwdata/yolo12-sam2-sparse-p39/weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='/home/hetao/graduate/data/VisDrone-VID/VisDrone2019-VID-test-dev/sequences', help='root folder containing video sequence folders')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    parser.add_argument('--output', type=str, default='runs/inference_test_dev', help='output directory')
    parser.add_argument('--limit', type=int, default=None, help='limit number of sequences for debug')
    opt = parser.parse_args()

    # Ensure output dir
    out_dir_root = Path(opt.output)
    out_dir_root.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {opt.weights}...")
    try:
        yolo = YOLO(opt.weights)
        model = yolo.model
        model.to(opt.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check State Injection
    supports_state = hasattr(model, 'set_memory') and hasattr(model, 'get_memory')
    if supports_state:
        print("Model supports State Injection.")
    else:
        print("Model does NOT support State Injection. Running in stateless mode (if possible).")

    # Locate Sequences
    source_root = Path(opt.source)
    if not source_root.exists():
        print(f"Error: Source {opt.source} not found.")
        return

    # Check if source is a single sequence or root of sequences
    # Heuristic: if it contains images directly, it's a sequence. If it contains dirs, it's a root.
    first_item = next(source_root.iterdir(), None)
    if first_item and first_item.is_dir():
        sequences = sorted([d for d in source_root.iterdir() if d.is_dir()])
        print(f"Found {len(sequences)} sequences in {source_root}")
    else:
        sequences = [source_root]
        print(f"Treating {source_root} as a single sequence.")

    if opt.limit:
        sequences = sequences[:opt.limit]
        print(f"Limiting to first {opt.limit} sequences.")

    # VisDrone Classes
    names = yolo.names if hasattr(yolo, 'names') else {
        0: 'ignored', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 
        5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 
        10: 'motor', 11: 'others'
    }

    # Preprocessing
    def preprocess(img, device):
        img = cv2.resize(img, (opt.imgsz, opt.imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    total_frames = 0
    
    for seq_idx, seq_dir in enumerate(sequences):
        seq_name = seq_dir.name
        print(f"\nProcessing Sequence {seq_idx+1}/{len(sequences)}: {seq_name}")
        
        # Reset Memory State for new video
        memory_state = None
        
        # Create output dir for this sequence
        seq_out_dir = out_dir_root / seq_name
        seq_out_dir.mkdir(exist_ok=True)
        
        files = sorted(list(seq_dir.glob('*.jpg')) + list(seq_dir.glob('*.png')))
        if not files:
            print("  No images found, skipping.")
            continue
            
        for i, file in enumerate(tqdm(files, desc=f"  Inferring {seq_name}")):
            img_raw = cv2.imread(str(file))
            if img_raw is None: continue
            
            img_tensor = preprocess(img_raw, opt.device)

            if supports_state:
                model.set_memory(memory_state)
            
            with torch.no_grad():
                preds = model(img_tensor)
                
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                
                # Post-process (NMS)
                # Max det increased for VisDrone crowded scenes
                preds = non_max_suppression(preds, opt.conf, 0.45, classes=None, max_det=500)
                
                det = preds[0] 
                
            if supports_state:
                memory_state = model.get_memory()

            # Draw
            annotator = Annotator(img_raw.copy(), line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_raw.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            img_out = annotator.result()
            
            # Save
            save_path = seq_out_dir / file.name
            cv2.imwrite(str(save_path), img_out)
            total_frames += 1

    print(f"\nAll Done. Processed {len(sequences)} sequences, {total_frames} frames.")
    print(f"Results saved to {out_dir_root}")

if __name__ == '__main__':
    main()
