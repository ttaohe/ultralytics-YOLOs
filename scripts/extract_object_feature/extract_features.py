import argparse
import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from collections import defaultdict

class FeatureExtractor:
    def __init__(self, model_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"Loading model from {model_path} to {device}...")
        self.model = YOLO(model_path)
        # Move to device
        self.model.to(device)
        
        self.features = []
        self.hook_handle = None
        self.detect_head = None
        self.strides = None
        
        self._register_hook()
        print("Model loaded and hook registered.")

    def _register_hook(self):
        # Access the underlying PyTorch model
        # YOLO object wrapper -> DetectionModel -> nn.Sequential
        # Usually self.model.model is the DetectionModel
        
        # Find the Detect head
        m = None
        # self.model.model might be the DetectionModel
        # traverse modules to find Detect
        for module in self.model.model.modules():
            if isinstance(module, Detect):
                m = module
                break
        
        if m is None:
            raise ValueError("Could not find Detect head in model.")
        
        self.detect_head = m
        # Move strides to CPU/numpy for easy calculation, or keep on device
        self.strides = m.stride
        if hasattr(self.strides, 'cpu'):
            self.strides = self.strides.cpu().numpy()
        elif isinstance(self.strides, (int, float)):
             self.strides = [self.strides]
        
        print(f"Found Detect head with strides: {self.strides}")
        
        # Register hook on forward
        # The input to Detect.forward is a list of feature maps [P3, P4, P5]
        def hook(module, input, output):
            # input is a tuple of arguments to forward. input[0] is x (list of tensors)
            x = input[0]
            # Store detached features
            self.features = [feat.detach() for feat in x]

        self.hook_handle = m.register_forward_hook(hook)

    def load_gt(self, gt_path):
        """
        Load GT from file.
        Assumes format: frame_id, id, x1, y1, x2, y2
        Returns dict: {frame_idx: [(id, x1, y1, x2, y2), ...]}
        """
        gt_data = defaultdict(list)
        print(f"Loading GT from {gt_path}...")
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    # try space separated
                    parts = line.strip().split()
                
                if len(parts) < 6:
                    continue
                
                # Parse
                try:
                    frame_idx = int(parts[0])
                    obj_id = int(parts[1])
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    x2 = float(parts[4])
                    y2 = float(parts[5])
                    
                    gt_data[frame_idx].append((obj_id, x1, y1, x2, y2))
                except ValueError:
                    continue
        return gt_data

    def process_video(self, video_path, gt_path, output_path, img_size=(640, 640)):
        gt_data = self.load_gt(gt_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        all_extracted_data = [] # list of tuples (frame_idx, obj_id, feature_vector)
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video {video_path} ({total_frames} frames)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process if we have GT for this frame
            current_gt = gt_data.get(frame_idx, [])
            if not current_gt:
                frame_idx += 1
                continue

            # Preprocess image (Letterbox)
            h0, w0 = frame.shape[:2]
            h, w = img_size
            r = min(h / h0, w / w0)
            padw, padh = (w - w0 * r) / 2, (h - h0 * r) / 2
            
            if r != 1:
                interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
                img = cv2.resize(frame, (int(w0 * r), int(h0 * r)), interpolation=interp)
            else:
                img = frame
            
            # Padding
            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
            # Convert to tensor
            # HWC to CHW, BGR to RGB
            img_tensor = img[:, :, ::-1].transpose(2, 0, 1) 
            img_tensor = np.ascontiguousarray(img_tensor)
            img_tensor = torch.from_numpy(img_tensor).to(self.device).float()
            img_tensor /= 255.0
            img_tensor = img_tensor.unsqueeze(0) # Add batch dim
            
            # Forward pass
            with torch.no_grad():
                self.model.model(img_tensor)
            
            # self.features is now populated with [P3, P4, P5] features
            # Extract feature vectors for each GT object
            for obj_data in current_gt:
                obj_id, x1, y1, x2, y2 = obj_data
                
                # Center in original image
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Map to resized/padded image
                cx_new = cx * r + left # left is padw integer
                cy_new = cy * r + top  # top is padh integer
                
                # Extract from each scale
                vectors = []
                for i, fmap in enumerate(self.features):
                    stride = self.strides[i]
                    
                    # Map to feature map grid
                    gx = cx_new / stride
                    gy = cy_new / stride
                    
                    # Nearest neighbor interpolation (taking the grid cell value)
                    ix = int(round(gx))
                    iy = int(round(gy))
                    
                    # Bounds check
                    _, C, fh, fw = fmap.shape
                    ix = max(0, min(ix, fw - 1))
                    iy = max(0, min(iy, fh - 1))
                    
                    # Extract vector
                    # fmap is (1, C, H, W)
                    vec = fmap[0, :, iy, ix].cpu().numpy()
                    vectors.append(vec)
                
                # Concatenate vectors from all scales
                full_vec = np.concatenate(vectors)
                all_extracted_data.append((frame_idx, obj_id, full_vec))
            
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
                
        cap.release()
        
        # Save results
        print("Saving results...")
        X = []
        ids = []
        frames = []
        
        for f_idx, o_id, vec in all_extracted_data:
            X.append(vec)
            ids.append(o_id)
            frames.append(f_idx)
            
        if len(X) == 0:
            print("No features extracted. Check GT format or video alignment.")
            return

        X = np.array(X)
        ids = np.array(ids)
        frames = np.array(frames)
        
        np.savez(output_path, features=X, ids=ids, frames=frames)
        print(f"Saved {len(X)} feature vectors to {output_path}")
        print("You can load it using: data = np.load('filename.npz'); features = data['features']")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract YOLO features for specific targets in video.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT file (frame_id, id, x1, y1, x2, y2)")
    parser.add_argument("--output", type=str, default="features.npz", help="Output .npz file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.model)
    extractor.process_video(args.video, args.gt, args.output, img_size=(args.imgsz, args.imgsz))

