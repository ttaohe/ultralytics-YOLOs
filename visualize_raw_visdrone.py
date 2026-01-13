
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def visualize_raw_visdrone():
    # User's raw data path from screenshot
    RAW_ROOT = Path("/home/hetao/graduate/data/VisDrone-VID/VisDrone2019-VID-train")
    SEQ_DIR = RAW_ROOT / "sequences"
    ANN_DIR = RAW_ROOT / "annotations"
    OUTPUT_DIR = Path("runs/visualize/visdrone_raw_official")
    
    # VisDrone Class Map (Standard)
    # 0: ignored region, 1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van, 
    # 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
    CLASS_NAMES = {
        0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 
        5: "van", 6: "truck", 7: "tricycle", 8: "awning-tri", 9: "bus", 
        10: "motor", 11: "others"
    }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading raw data from: {RAW_ROOT}")
    
    if not SEQ_DIR.exists() or not ANN_DIR.exists():
        print(f"Error: Annotations or Sequences dir not found at {RAW_ROOT}")
        print("Please verify the path.")
        return

    # Get all sequences
    # sequences are folders inside SEQ_DIR
    seqs = sorted([d for d in SEQ_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(seqs)} sequences.")
    
    # Limit number of sequences to process to save time during debug
    # seqs = seqs[:1] 
    
    for seq_path in tqdm(seqs, desc="Processing Sequences"):
        seq_name = seq_path.name
        ann_file = ANN_DIR / f"{seq_name}.txt"
        
        if not ann_file.exists():
            print(f"Warning: Annotation file {ann_file} not found for sequence {seq_name}")
            continue
            
        # Parse Annotations
        # Format: frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion
        anns = {} # frame_idx -> list of boxes
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                # First parts are ints
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                w = int(parts[4])
                h = int(parts[5])
                score = float(parts[6])
                cls_id = int(parts[7])
                trunc = int(parts[8])
                occ = int(parts[9])
                
                # Filter 'ignored' or 'others' if desired? 
                # Usually we want to see everything raw.
                
                if frame_idx not in anns:
                    anns[frame_idx] = []
                anns[frame_idx].append({
                    'bbox': [x, y, w, h],
                    'cls': cls_id,
                    'id': target_id,
                    'score': score
                })
        
        # Visualize Frames
        # Frame images in VisDrone usually: 0000001.jpg, etc.
        # But verify naming convention in folder
        frames = sorted(list(seq_path.glob("*.jpg")))
        
        # Only visualize first 10 frames of each sequence to be quick
        MAX_FRAMES_PER_SEQ = 5 
        
        for i, frame_path in enumerate(frames):
            if i >= MAX_FRAMES_PER_SEQ: break
            
            # Parse frame index from filename?
            # e.g. 0000001.jpg -> 1
            try:
                frame_idx = int(frame_path.stem)
            except ValueError:
                # If naming is different, might need adjustment
                print(f"Skipping {frame_path}, cannot parse frame index")
                continue
            
            img = cv2.imread(str(frame_path))
            if img is None: continue
            
            # Draw Anns
            if frame_idx in anns:
                for ann in anns[frame_idx]:
                    x, y, w, h = ann['bbox']
                    cls_id = ann['cls']
                    tid = ann['id']
                    
                    # Colors
                    c = cls_id
                    color = ((c * 50) % 255, (c * 80) % 255, (c * 130) % 255)
                    
                    # Draw Box
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    
                    # Label
                    label_text = f"{CLASS_NAMES.get(cls_id, str(cls_id))} ID:{tid}"
                    cv2.putText(img, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save
            out_name = f"{seq_name}_{frame_path.name}"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), img)

    print(f"Done. Saved raw visualizations to {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_raw_visdrone()
