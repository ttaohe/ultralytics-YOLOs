
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
from visualization.visualize_attention import AttentionVisualizer

def preprocess(img, imgsz, device):
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train-video-sparse/yolo12-sam2-sparse-p33/weights/best.pt')
    parser.add_argument('--source', type=str, default='/home/hetao/graduate/data/VisDrone-VID-yolo/images/test/uav0000009_03358_v/')
    parser.add_argument('--imgsz', type=int, default=1920)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='visualization/output')
    parser.add_argument('--max-frames', type=int, default=10)
    opt = parser.parse_args()

    # Setup Output
    output_dir = Path(opt.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    print(f"Loading model from {opt.weights}...")
    yolo = YOLO(opt.weights)
    model = yolo.model
    model.to(opt.device)
    model.eval()

    # Helper: Load Images
    source_path = Path(opt.source)
    if source_path.is_dir():
        files = sorted(list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')))
    else:
        print("Source must be a directory of images.")
        return

    files = files[:opt.max_frames]
    print(f"Processing {len(files)} frames...")

    # Stateful components
    memory_state = None
    image_buffer = [] # To store historic raw images for visualization
    
    # Initialize Visualizer
    visualizer = AttentionVisualizer(model)

    with visualizer:
        for i, file in enumerate(files):
            img_raw = cv2.imread(str(file))
            if img_raw is None: continue
            
            # State Injection
            model.set_memory(memory_state)
            
            # Forward Pass using YOLO engine (handles NMS, etc.)
            # yolo.predict returns a list of Results objects
            # Use rect=False to ensure square inference (simpler for visualizer grid mapping)
            results = yolo.predict(img_raw, imgsz=opt.imgsz, device=opt.device, verbose=False, embed=None, rect=False)
            
            # Logic: We can only visualize IF we have memory
            if i > 0 and len(image_buffer) > 0:
                # Get Detections (Results list of length 1 for batch 1)
                det_result = results[0]
                
                # Check if we have any boxes
                if det_result.boxes is not None and len(det_result.boxes) > 0:
                    # Pick the highest confidence box
                    # boxes.conf is Tensor
                    best_idx = det_result.boxes.conf.argmax()
                    box = det_result.boxes.xywhn[best_idx].cpu().numpy() # [x_center, y_center, w, h] normalized
                    
                    # box is [x_center, y_center, w, h] normalized
                    query_box = box # Pass full box for sampling
                    
                    print(f"Frame {i}: Visualizing attention for Object at {query_box[:2]}")
                    
                    # Visualize
                    # Pass currently stored history images corresponding to memory
                    mem_imgs = image_buffer[-8:] 
                    
                    vis_curr, vis_mems_no_rope, vis_mems_rope = visualizer.visualize(
                        img_raw, mem_imgs, query_box=query_box, imgsz=opt.imgsz
                    )
                    
                    if vis_curr is not None:
                        # Draw the detection box on current frame for confirmation
                        H, W = vis_curr.shape[:2]
                        # query_box is [x_center, y_center, w, h] normalized
                        cx, cy = int(query_box[0]*W), int(query_box[1]*H)
                        bw, bh = int(query_box[2]*W), int(query_box[3]*H)
                        x1, y1 = cx - bw//2, cy - bh//2
                        x2, y2 = cx + bw//2, cy + bh//2
                        cv2.rectangle(vis_curr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Save Montage
                        # Layout: [Memory Frames Row]
                        #         [Current Frame]
                        
                        # Resize mems to small
                        thumb_h = 240
                        scale = thumb_h / vis_curr.shape[0]
                        vis_curr_small = cv2.resize(vis_curr, (0,0), fx=scale, fy=scale)
                        
                        row_mem = []
                        # 默认用 With-RoPE 版本做缩略展示
                        for m in vis_mems_rope:
                            m_small = cv2.resize(m, (0,0), fx=scale, fy=scale)
                            row_mem.append(m_small)
                        
                        if row_mem:
                            top_row = np.hstack(row_mem)
                            
                            cv2.imwrite(str(output_dir / f"frame_{i:03d}_curr.jpg"), vis_curr)
                            
                            # 分别保存 No-RoPE / With-RoPE 的可视化结果
                            for midx, (m_img_no, m_img_rope) in enumerate(zip(vis_mems_no_rope, vis_mems_rope)):
                                frame_real_idx = i - len(vis_mems_rope) + midx
                                cv2.imwrite(
                                    str(output_dir / f"frame_{i:03d}_link_to_{frame_real_idx:03d}_norpe.jpg"),
                                    m_img_no,
                                )
                                cv2.imwrite(
                                    str(output_dir / f"frame_{i:03d}_link_to_{frame_real_idx:03d}_rope.jpg"),
                                    m_img_rope,
                                )
                            print(f"Saved visualization for frame {i}")
                else:
                    print(f"Frame {i}: No detections found.")

            # State Extraction & Buffer Update
            memory_state = model.get_memory()
            image_buffer.append(img_raw)
            
            # Keep buffer size reasonable (aligned with max_memory slightly loosely)
            if len(image_buffer) > 16:
                image_buffer.pop(0)

    print(f"Done. Check results in {opt.output}")

if __name__ == '__main__':
    main()
