import torch
import numpy as np
from pathlib import Path
import cv2
from .dataset import YOLODataset

class MultiviewDataset(YOLODataset):
    """
    Dataset for multiview object detection.
    Loads multiple views of the same scene and their corresponding geographic coordinates.
    """
    def __init__(self, *args, num_views=1, **kwargs):
        self.num_views = num_views
        super().__init__(*args, **kwargs)
        
    def __len__(self):
        # Assume images are ordered: Scene1_View1, Scene1_View2, ..., Scene2_View1, ...
        return len(self.im_files) // self.num_views

    def load_coords(self, im_file, shape):
        """
        Load geographic coordinates for an image.
        Assumes coords are in a 'coords' directory parallel to 'images', 
        with .npy extension.
        
        Args:
            im_file (str): Path to image file.
            shape (tuple): (h, w) of the image.
            
        Returns:
            torch.Tensor: (h, w, 3) coordinates.
        """
        im_path = Path(im_file)
        # images/train/img1.jpg -> coords/train/img1.npy
        coord_path = im_path.parents[1] / 'coords' / im_path.parent.name / (im_path.stem + '.npy')
        
        if coord_path.exists():
            try:
                coords = np.load(coord_path) # Expecting (H, W, 3)
                # Resize if necessary to match image shape
                if coords.shape[:2] != shape:
                    coords = cv2.resize(coords, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                return torch.from_numpy(coords).float()
            except Exception as e:
                print(f"Error loading coords {coord_path}: {e}")
                return torch.zeros(shape[0], shape[1], 3)
        else:
            # Fallback: Generate dummy coords or return zeros
            # print(f"Warning: Coords not found for {im_file}, using zeros.")
            return torch.zeros(shape[0], shape[1], 3)

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
        - img: (V, C, H, W)
        - coords: (V, H, W, 3)
        - cls: (N, 1) flattened
        - bboxes: (N, 4) flattened
        - batch_idx: (N, 1) mapping to view index in the batch (0..V-1)
        """
        start_idx = index * self.num_views
        views_data = []
        
        # Load all views for this scene
        for v in range(self.num_views):
            # YOLODataset.__getitem__ returns a dict for a single image
            data = super().__getitem__(start_idx + v)
            views_data.append(data)
            
        # Stack images: (V, C, H, W)
        imgs = [d['img'] for d in views_data]
        imgs_stack = torch.stack(imgs)
        
        # Load Coords
        # Note: 'img' in data is already resized/augmented. 
        # We need to load coords and apply same resize? 
        # YOLODataset doesn't easily expose the raw resize params in __getitem__ output 
        # without some hacking. 
        # For now, we load coords matching the *output* image size.
        _, h, w = imgs_stack.shape[1:]
        coords_list = []
        for v in range(self.num_views):
            im_file = self.im_files[start_idx + v]
            c = self.load_coords(im_file, (h, w))
            coords_list.append(c)
        coords_stack = torch.stack(coords_list) # (V, H, W, 3)

        # Aggregate labels
        # We need to adjust batch_idx to be unique across views if we treat them as separate images in a batch,
        # OR we keep them as one sample.
        # Standard YOLO batching expects flat list of images.
        # But here one "sample" is V images.
        # We will return a structure that collate_fn can handle.
        
        data = {
            'img': imgs_stack,
            'coords': coords_stack,
            'cls': [d['cls'] for d in views_data],
            'bboxes': [d['bboxes'] for d in views_data],
            'im_file': [d['im_file'] for d in views_data],
            'ori_shape': [d['ori_shape'] for d in views_data],
            'resized_shape': [d['resized_shape'] for d in views_data],
        }
        return data

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for MultiviewDataset.
        Batch is a list of N samples, each containing V views.
        Output should be compatible with YOLO model input.
        
        Args:
            batch: List of dicts from __getitem__
            
        Returns:
            dict with:
                'img': (N*V, C, H, W)
                'coords': (N, V, H, W, 3)
                'batch_idx': (M) - maps targets to image index in (N*V)
                'cls': (M, 1)
                'bboxes': (M, 4)
        """
        new_batch = {}
        imgs = []
        coords = []
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        
        for i, sample in enumerate(batch):
            # sample['img'] is (V, C, H, W)
            imgs.append(sample['img'])
            coords.append(sample['coords'])
            
            # Process labels
            # We have V lists of labels
            for v in range(len(sample['cls'])):
                # Image index in the flattened batch
                img_idx = i * len(sample['cls']) + v
                
                c = sample['cls'][v]
                b = sample['bboxes'][v]
                
                if c.shape[0] > 0:
                    all_cls.append(c)
                    all_bboxes.append(b)
                    all_batch_idx.append(torch.full((c.shape[0],), img_idx))
        
        # Stack images: (N, V, C, H, W) -> (N*V, C, H, W)
        imgs_tensor = torch.cat(imgs, dim=0)
        
        # Stack coords: (N, V, H, W, 3)
        coords_tensor = torch.stack(coords, dim=0)
        
        # Concatenate labels
        if all_cls:
            new_batch['cls'] = torch.cat(all_cls, dim=0)
            new_batch['bboxes'] = torch.cat(all_bboxes, dim=0)
            new_batch['batch_idx'] = torch.cat(all_batch_idx, dim=0)
        else:
            new_batch['cls'] = torch.zeros((0, 1))
            new_batch['bboxes'] = torch.zeros((0, 4))
            new_batch['batch_idx'] = torch.zeros((0))

        new_batch['img'] = imgs_tensor
        new_batch['coords'] = coords_tensor
        
        return new_batch
