
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configuration
SOURCE_DATASET = "/home/hetao/graduate/data/VisDrone-VID-yolo-cp"
TARGET_DATASET = "/home/hetao/graduate/data/VisDrone-VID-yolo-masked"
# Raw annotations are split by split? 
# Usually raw annotations are in specific folders. 
# Based on previous yaml: /home/hetao/graduate/data/VisDrone-VID/VisDrone2019-VID-train/annotations
# We need to handle train/val/test splits if they have different raw annotation paths.
# Let's assume standard VisDrone structure or just use the one known path for training?
# The user only cared about masking training data? "Mask train data ignored parts".
# But validation should also be masked to match distribution? User said "Test time... ignore".
# Let's safe-guard: mask ALL if possible, or at least train.
# The raw_annotations_root in yaml was specific to 'train'.
# If validation has its own annotations, we need that path too.
# Let's inspect where val annotations are.
# VisDrone2019-VID-val/annotations?

RAW_ANNS_TRAIN = "/home/hetao/graduate/data/VisDrone-VID/VisDrone2019-VID-train/annotations"
RAW_ANNS_VAL = "/home/hetao/graduate/data/VisDrone-VID/VisDrone2019-VID-val/annotations"
# RAW_ANNS_TEST = ... (Test usually has no annotations? Or dev?)
# We will focus on TRAIN and VAL for now.

IGNORED_CLASS_IDS = [0, 11] # 0: Ignored, 11: Others

def load_ignored_boxes(ann_path):
    """Load ignored boxes from a raw VisDrone annotation file."""
    boxes_by_frame = {}
    if not os.path.exists(ann_path):
        return boxes_by_frame
        
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts: continue
            try:
                # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                f_idx = int(parts[0])
                cls_id = int(parts[7])
                
                if cls_id in IGNORED_CLASS_IDS:
                    x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                    if w > 0 and h > 0:
                        if f_idx not in boxes_by_frame:
                            boxes_by_frame[f_idx] = []
                        boxes_by_frame[f_idx].append([x, y, x+w, y+h])
            except ValueError:
                continue
    return boxes_by_frame

def process_single_image(args):
    """Read image, apply mask, save to target."""
    src_img_path, dst_img_path, boxes = args
    
    # Read image
    img = cv2.imread(str(src_img_path))
    if img is None:
        print(f"Warning: Failed to read {src_img_path}")
        return
        
    # Apply mask
    if boxes:
        h_img, w_img = img.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box
            # Clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = 114 # Gray
                
    # Save
    cv2.imwrite(str(dst_img_path), img)

def main():
    print(f"Cloning dataset from {SOURCE_DATASET} to {TARGET_DATASET}...")
    
    # 1. Copy everything first (rsync-like) or just specific folders?
    # To be safe and clean: Copy labels and folder structure, process images.
    
    ignore_patterns = shutil.ignore_patterns("*.jpg", "*.jpeg", "*.png") # Don't copy images
    
    if os.path.exists(TARGET_DATASET):
        print(f"Target directory {TARGET_DATASET} exists. Deleting...")
        shutil.rmtree(TARGET_DATASET)
        
    print("Copying directory structure and non-image files (labels)...")
    shutil.copytree(SOURCE_DATASET, TARGET_DATASET, ignore=ignore_patterns)
    
    # 2. Process Images
    splits = ['train', 'val', 'test']
    
    # Pre-load annotations mapping
    # Map sequence name to annotation Dict
    # Sequence name is parent folder of image. 
    # Annotation file is {seq_name}.txt
    
    raw_ann_roots = {
        'train': RAW_ANNS_TRAIN,
        'val': RAW_ANNS_VAL,
        # 'test': ...
    }
    
    # We load annotations into memory (filename -> boxes)
    # But files are distributed in subfolders per sequence.
    
    # Actually, simpler: Iterate input images. For each image, identify seq and frame.
    # Look up annotation.
    
    # Let's collect all image tasks
    tasks = []
    
    for split in splits:
        src_split_dir = Path(SOURCE_DATASET) / 'images' / split
        dst_split_dir = Path(TARGET_DATASET) / 'images' / split
        
        if not src_split_dir.exists():
            continue
            
        # Ensure dst exists (should be created by copytree but images might be ignored so empty dirs might assume)
        # copytree ignore patterns might skip the files but should keep dirs? 
        # Actually ignore patterns apply to directory names too if they match? No, usually fine.
        # But copytree with ignore *.jpg will create the directory structure (dirs not ignored).
        
        print(f"Scanning {split} images...")
        
        # Load all annotations for this split
        raw_root = raw_ann_roots.get(split)
        split_anns = {} # seq_name -> { frame_idx -> boxes }
        
        if raw_root and os.path.exists(raw_root):
             print(f"Loading raw annotations for {split} from {raw_root}...")
             ann_files = list(Path(raw_root).glob("*.txt"))
             for ann_file in tqdm(ann_files):
                 seq_name = ann_file.stem
                 split_anns[seq_name] = load_ignored_boxes(str(ann_file))
        
        # Walk source images
        # Structure: images/train/uav000013_00000_v/0000001.jpg
        
        # glob all images
        images = list(src_split_dir.rglob("*.jpg")) # Assuming jpg
        
        for img_path in images:
            # Replicate path in dest
            rel_path = img_path.relative_to(src_split_dir)
            dst_path = dst_split_dir / rel_path
            
            # Ensure parent dir exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Identify boxes
            seq_name = img_path.parent.name
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                frame_idx = -1
                
            boxes = []
            if seq_name in split_anns and frame_idx in split_anns[seq_name]:
                boxes = split_anns[seq_name][frame_idx]
                
            tasks.append((img_path, dst_path, boxes))
            
    print(f"Processing {len(tasks)} images...")
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
        
    print("Done! Masked dataset created at:", TARGET_DATASET)

if __name__ == "__main__":
    main()
