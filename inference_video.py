
import cv2
import torch
import numpy as np

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train-video-sparse/yolo12-sam2-sparse-p32/weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='dataset/visdrone_video/test/uav0000305_00000_v', help='video source or folder')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')
    opt = parser.parse_args()

    from ultralytics import YOLO
    
    # Load Model using standard factory
    print(f"Loading model from {opt.weights}...")
    yolo = YOLO(opt.weights)
    model = yolo.model # Get the underlying nn.Module (YOLOVideo)
    model.to(opt.device)
    
    # Check if State Injection methods exist
    if not hasattr(model, 'set_memory') or not hasattr(model, 'get_memory'):
        print("Error: Model does not support State Injection (set_memory/get_memory)!")
        return

    # Simulate Stateful Inference
    # In a real server, 'memory_state' would be stored in a Cache matched to a Request ID
    memory_state = None 
    
    # Helper to process a folder of images as a video stream
    source = Path(opt.source)
    if source.is_dir():
        files = sorted(list(source.glob('*.jpg')) + list(source.glob('*.png')))
    else:
        print(f"Error: Source {opt.source} is not a directory. This script assumes image folder for simplicity.")
        return

    # Preprocessing
    def preprocess(img, device):
        img = cv2.resize(img, (opt.imgsz, opt.imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    print(f"Inference on {len(files)} frames from {opt.source}")
    
    for i, file in enumerate(files):
        img_raw = cv2.imread(str(file))
        if img_raw is None: continue
        
        # Preprocess
        img = preprocess(img_raw, opt.device)

        # [State Injection] Inject previous memory state
        # In first frame, memory_state is None, model handles it as empty bank.
        model.set_memory(memory_state)
        
        # Inference
        # Note: model() return tuple (pred, proto) on training or (pred) on val depending on logic, 
        # but yolo output is usually (pred). 
        # Base DetectionModel forward returns x
        with torch.no_grad():
            results = model(img)
        
        # [State Extraction] Retrieve updated memory state
        memory_state = model.get_memory()
        if i % 100 == 0:
            mem_size = 0
            if memory_state:
                # memory_state is List[Tensor], index 0 is the bank
                # bank[0] is typically [B, N, C]
                if isinstance(memory_state, list) and len(memory_state) > 0:
                     mem_tensor = memory_state[0]
                     mem_size = mem_tensor.numel() * 4 / 1024 / 1024 # MB
            print(f"frame {i}: Processed. Memory Size in State: {mem_size:.2f} MB")

    print("Inference Finished. State Injection mechanism verified.")

if __name__ == '__main__':
    main()
