import sys; sys.modules['ultralytics.nn.tasks'] = None;

import torch
import cv2
import numpy as np
import os
import sys

# Hack to avoid circular import by not importing the whole world
# We just need VisDroneVideoDataset and check_det_dataset
# which shouldn't trigger model loading if we are careful.
# But check_det_dataset imports nn.autobackend...
# Let's mock check_det_dataset return value since we know the path.

def mock_get_dataset_info(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

from ultralytics.data.video_dataset import VisDroneVideoDataset
from ultralytics.utils import LOGGER

# Mock Args
class Args:
    def __init__(self):
        self.task = 'detect'
        self.imgsz = 640
        self.rect = False
        self.single_cls = False
        self.stride = 32
        self.fraction = 1.0
        self.data = '/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/datasets/VisDrone-Video.yaml'

def visualize_augmentation():
    img_path = '/home/hetao/graduate/data/VisDrone-VID-yolo/images/train'
    
    # Initialize Dataset with Augmentation ENABLED
    # Create a mock cfg object because we are manually calling it
    # data = check_det_dataset('/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/datasets/VisDrone-vid.yaml')
    data = mock_get_dataset_info('/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/datasets/VisDrone-vid.yaml')
    
    # Define a mock Config object that works as both object (cfg.task) and dict (cfg['key'])
    # And ensures attribute assignment updates dict
    class MockCfg(dict):
        def __getattr__(self, key):
            return self.get(key, None)
        def __setattr__(self, key, value):
            self[key] = value

    cfg = MockCfg()
    cfg.task = 'detect'
    cfg.imgsz = 640
    cfg.rect = False
    cfg.single_cls = False
    cfg.stride = 32
    # Add augmentation params needed by v8_transforms
    cfg.mosaic = 1.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0
    cfg.copy_paste_mode = 'flip' # Fix AssertionError
    cfg.cutmix = 0.0
    cfg.degrees = 0.0
    
    print(f"DEBUG: cfg keys: {cfg.keys()}")
    print(f"DEBUG: cfg.mosaic: {cfg.mosaic}")
    print(f"DEBUG: cfg['mosaic']: {cfg['mosaic']}")
    cfg.translate = 0.1
    cfg.scale = 0.5
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.flipud = 0.0
    cfg.fliplr = 0.5
    cfg.bgr = 0.0
    cfg.hsv_h = 0.015
    cfg.hsv_s = 0.7
    cfg.hsv_v = 0.4
    
    # Initialize Dataset with Augmentation ENABLED
    # We must pass 'data' because YOLODataset uses it
    dataset = VisDroneVideoDataset(
        img_path=img_path,
        imgsz=640,
        batch_size=4,
        augment=True,
        hyp=cfg, # Pass our robust mock cfg
        rect=False,
        cache=False,
        stride=32,
        data=data # PASS DATA DICT
    )
    
    # Force Mosaic to be active (usually controlled by Loader)
    # We need to manually simulate the loader's behavior or just check get_image_and_label?
    # Actually, Mosaic is applied in v8_transforms which is called by the Loader's collate_fn or dataset wrapper.
    # Wait, VisDroneVideoDataset is just a Dataset. The transforms are applied in `build_transforms` in `ultralytics/data/build.py`.
    # To properly test this without rebuilding the whole pipeline, we should manually construct the transform.
    
    print("WARNING: Dataset.getitem only returns raw images (mostly). Transforms are usually external.")
    print("Checking if VisDroneVideoDataset applies transforms internally...")
    # It does NOT. It returns raw labels. The transforms are applied later.
    
    # We need to instantiate the transforms as the Trainer does.
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.cfg import get_cfg
    
    # Let's try to use the build_yolo_dataset function which sets up everything
    cfg = get_cfg(cfg='/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/models/12/yolo12-video-sparse-p3.yaml')
    cfg.data = '/home/hetao/graduate/ultralytics-YOLOs/ultralytics/cfg/datasets/VisDrone-Video.yaml'
    cfg.imgsz = 640
    cfg.mosaic = 1.0 # Force Mosaic
    cfg.mixup = 0.0
    cfg.degrees = 10.0
    cfg.translate = 0.1
    cfg.scale = 0.5
    
    print("Building Dataset with Transforms... (Manual Mode)")
    # Note: We use 'train' mode to get augmentations
    # dataset = build_yolo_dataset(cfg, img_path, batch=4, data={'names': {0:'car'}}, mode='train', rect=False, stride=32)
    
    # This returns a YOLODataset (or subclass) with .transforms attribute set
    
    output_dir = "visualization_aug_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Checking raw get_image_and_label(0) output...")
    raw_label = dataset.get_image_and_label(0)
    raw_img = raw_label['img']
    print(f"Raw Image Shape: {raw_img.shape}")
    
    print("Testing Mosaic transform in isolation...")
    from ultralytics.data.augment import Mosaic, RandomPerspective, LetterBox
    
    mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
    # Get a fresh raw label
    label = dataset.get_image_and_label(0)
    print(f"Input to Mosaic Shape: {label['img'].shape}")
    
    out_label = mosaic(label)
    print(f"Output from Mosaic Shape: {out_label['img'].shape}")
    
    if out_label['img'].shape[2] == 6:
        print("Mosaic SUCCESS: Preserved 6 channels.")
    else:
        print("Mosaic FAILED: Dropped channels.")

    # If Mosaic works, maybe RandomPerspective?
    print("Testing RandomPerspective...")
    affine = RandomPerspective(degrees=0.0, translate=0.1, scale=0.5, pre_transform=None) # Mosaic output is large, so pre_transform not needed?
    # Wait, RandomPerspective handles resizing to target size based on border.
    # Mosaic4 returns 2*imgsz.
    # We need to simulate v8_transforms chain.
    
    # In v8_transforms: Compose([mosaic, affine])
    # affine is initialized with pre_transform=LetterBox if stretch=False.
    # But Mosaic runs first.
    # If Mosaic runs, 'rect_shape' should be None.
    
    label_m = mosaic(dataset.get_image_and_label(0))
    # output of Mosaic is 1280x1280 (if imgsz=640)
    
    # RandomPerspective usually takes 640x640 input?
    # No, it adapts.
    out_affine = affine(label_m)
    print(f"Output from Affine Shape: {out_affine['img'].shape}")
    
    
    # Proceed to batch loop if we want...
    print("Running Ablation on Transforms to find 3-channel culprit...")
    
    # helper to check output
    def check_transform_chain(chain_name, transform_list, index=0):
        # We need to manually compose them because dataset.transforms expects a callable that takes dict
        # Ultralytics Compose is:
        # class Compose:
        #    def __init__(self, transforms):
        #        self.transforms = transforms
        #    def __call__(self, data):
        #        for t in self.transforms:
        #             data = t(data)
        #        return data
        
        # We can simulate this
        try:
            # We must use fresh label for each test
            label = dataset.get_image_and_label(index)
            
            for t in transform_list:
                label = t(label)
                
            img = label['img']
            # Format returns Tensor, others return dict
            # Wait, Format returns DICT with 'img' as Tensor.
            
            if isinstance(img, torch.Tensor):
                c = img.shape[0]
            else:
                c = img.shape[2]
                
            print(f"  [{chain_name}] Output Channels: {c}")
            return c == 6
        except Exception as e:
            print(f"  [{chain_name}] CRASHED: {e}")
            return False

    full_transforms = dataset.transforms.transforms
    # Flatten the nested Compose (Mosaic+Affine is in pre_transform Compose)
    # The printed list was: [Compose(Mosaic, CP, Affine), MixUp, CutMix, Albu, HSV, Flip, Flip, Format]
    
    flat_list = []
    # Inspect first element
    if hasattr(full_transforms[0], 'transforms'):
        flat_list.extend(full_transforms[0].transforms)
        flat_list.extend(full_transforms[1:])
    else:
        flat_list = full_transforms

    print(f"Flattened Transforms list len: {len(flat_list)}")
    
    current_chain = []
    for i, t in enumerate(flat_list):
        t_name = str(t).split('.')[-1].split(' object')[0]
        current_chain.append(t)
        if not check_transform_chain(f"+{t_name}", current_chain):
            print(f"CULPRIT FOUND: {t_name}")
            break
            
    print("Ablation Complete.")
    
    # Restore visuals
    # If we found culprit, we might have crashed.
    # But for the User's visualization request, we need a working chain.
    # We can reconstruct a safe chain (just Mosaic + Affine + Format?)
    
    print("Generating visualization with Safe Chain (Mosaic+Affine+Format)...")
    safe_chain = [flat_list[0], flat_list[2], flat_list[-1]] # Mosaic, Affine, Format (skip CopyPaste and others)
    # Note: flat_list indices depend on flattening logic. 
    # [Mosaic, CopyPaste, Affine, Mixup, Cutmix, Albu, HSV, Flip, Flip, Format]
    # 0: Mosaic
    # 1: CopyPaste
    # 2: Affine
    # ...
    # -1: Format
    
    # Let's verify safe chain
    if check_transform_chain("SafeChain", safe_chain):
        # Apply this chain to 5 samples and save
        # from ultralytics.data.augment import Compose
        
        class DebugCompose:
            def __init__(self, transforms):
                self.transforms = transforms
            def __call__(self, data):
                print("DebugCompose Start")
                for i, t in enumerate(self.transforms):
                    t_name = str(t).split('.')[-1].split(' object')[0]
                    data = t(data)
                    img = data['img']
                    if isinstance(img, torch.Tensor):
                         shape = img.shape
                    else:
                         shape = img.shape
                    print(f"  After {t_name}: {shape}")
                return data

        dataset.transforms = DebugCompose(safe_chain)
        
        for i in range(5):
            print(f"Saving safe sample {i}...")
            batch = dataset[i]
            imgtext = batch['img']
            print(f"  Batch Tensor Shape: {imgtext.shape}")
            
            # VisDroneVideoDataset splits them into 'img' and 'history_img'
            if 'history_img' in batch:
                hist_text = batch['history_img']
                print(f"  History Tensor Shape: {hist_text.shape}")
                # Stack them back for visualization
                img_all = torch.cat([imgtext, hist_text], dim=0) # (6, H, W)
                img = img_all
            else:
                img = imgtext

            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            
            # Now img should be 6 channels if history exists
            
            if img.shape[2] == 6:
                img = (img * 255).astype(np.uint8)
                curr = img[..., 0:3]
                hist = img[..., 3:6]
                
                # Draw Grid
                grid_step = 64
                H, W = curr.shape[:2]
                color = (0, 255, 0)
                for x in range(0, W, grid_step):
                    cv2.line(curr, (x, 0), (x, H), color, 1)
                    cv2.line(hist, (x, 0), (x, H), color, 1)
                for y in range(0, H, grid_step):
                    cv2.line(curr, (0, y), (W, y), color, 1)
                    cv2.line(hist, (0, y), (W, y), color, 1)
                    
                combined = np.hstack([curr, hist])
                cv2.imwrite(f"{output_dir}/aug_sample_{i}_fixed.jpg", combined)
                print(f"FAILED: Saved {output_dir}/aug_sample_{i}_fixed.jpg")
            else:
                 print(f"Safe chain failed?? Shape: {img.shape}")

if __name__ == "__main__":
    visualize_augmentation()
