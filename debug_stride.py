import numpy as np
import logging

# Mock Logger
class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warning(self, msg):
        print(f"[WARN] {msg}")

LOGGER = Logger()

class MockDataset:
    def __init__(self, n=20, vid_stride=5):
        self.prefix = "MockDataset: "
        self.im_files = [f"img_{i}.jpg" for i in range(n)]
        self.labels = [{"id": i, "file": f"img_{i}.jpg"} for i in range(n)]
        # Create 2 videos
        # Video 0: 0-9
        # Video 1: 10-19
        self.video_indices = np.array([0]*10 + [1]*10)
        self.vid_stride = vid_stride
        self.augment = True
        
        print("Initial Dataset:")
        for i in range(n):
            print(f"  {i}: {self.im_files[i]} | Label ID: {self.labels[i]['id']} | Vid: {self.video_indices[i]}")

    def _filter_dataset_by_stride(self):
        """
        Filter dataset to only keep every Nth frame (vid_stride) for each video.
        This modifies self.im_files, self.labels, self.video_indices, etc. in place.
        """
        keep_indices = []
        
        # We need to process each video separately
        # self.video_indices is already populated by _get_video_indices in __init__
        
        # Find boundaries of each video
        # Since self.video_indices is a numpy array of video IDs corresponding to self.im_files
        unique_vids = np.unique(self.video_indices)
        
        for vid in unique_vids:
            # Get all indices for this video
            # Note: np.where returns a tuple
            indices = np.where(self.video_indices == vid)[0]
            indices = sorted(indices) # Ensure sorted order
            
            # Select every Nth frame
            # Example: indices=[0, 1, 2, 3, 4, 5], stride=5 -> [0, 5]
            selected = indices[::self.vid_stride]
            keep_indices.extend(selected)
            
        keep_indices = sorted(keep_indices)
        
        # Apply filtering
        n_before = len(self.im_files)
        
        # Filter im_files
        self.im_files = [self.im_files[i] for i in keep_indices]
        
        # Filter labels
        self.labels = [self.labels[i] for i in keep_indices]
        
        # Filter video_indices (re-compute or filter)
        # It's safer to re-compute or filter array
        self.video_indices = self.video_indices[keep_indices]
             
        self.ni = len(self.im_files)
        
        LOGGER.info(f"{self.prefix}Sparse Video Sampling enabled: "
                    f"vid_stride={self.vid_stride}. "
                    f"Reduced dataset from {n_before} to {self.ni} images.")

    def verify(self):
        print("\nFiltered Dataset:")
        for i in range(len(self.im_files)):
            label_id = self.labels[i]['id']
            img_name = self.im_files[i]
            expected_name = f"img_{label_id}.jpg"
            status = "OK" if img_name == expected_name else "MISMATCH"
            print(f"  {i}: {img_name} | Label ID: {label_id} | Vid: {self.video_indices[i]} -> {status}")

if __name__ == "__main__":
    ds = MockDataset(n=20, vid_stride=5)
    ds._filter_dataset_by_stride()
    ds.verify()
