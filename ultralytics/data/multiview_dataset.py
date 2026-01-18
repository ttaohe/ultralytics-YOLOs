import torch
import numpy as np
from pathlib import Path
import os
import cv2
import warnings
from .dataset import YOLODataset
from ultralytics.utils import LOGGER

class MultiviewDataset(YOLODataset):
    """
    Dataset for multiview object detection.
    Loads multiple views of the same scene and their corresponding position encodings.

    Supports two modes:
    1. Interleaved: Images are in one directory, ordered as view1, view2, view1, view2, ...
    2. Separate: Images are in separate directories (camera1/, camera2/), specified via camera_dirs

    Optimizations:
    - LRU cache for recently used PEs (size-limited, default 1000 entries ~ 5-10MB)
    - On-demand loading with efficient disk I/O
    """
    def __init__(self, *args, num_views=1, camera_dirs=None, pe_cache_size=1000, pe_mmap=False, pe_lmdb=None,
                 pe_disable=False, **kwargs):
        self.num_views = num_views
        self._missing_pe_warned = False  # Flag to warn only once
        self.camera_dirs = camera_dirs  # List of camera subdirectories (e.g., ['camera1', 'camera2'])
        self._separate_mode = camera_dirs is not None
        self._pe_cache_size = pe_cache_size  # Unused (PE cache disabled)
        self._pe_mmap = pe_mmap
        self._pe_lmdb = pe_lmdb
        self._pe_lmdb_envs = {}
        self._pe_disable = pe_disable

        # For separate mode, we need to handle im_files differently
        if self._separate_mode:
            # Set camera_dirs before calling super().__init__
            self._camera_dirs = camera_dirs
        else:
            self._camera_dirs = None

        super().__init__(*args, **kwargs)

        # In separate mode, rebuild im_files to interleave images from all cameras
        if self._separate_mode:
            self._build_separate_mode_im_files()

        # PE cache disabled to avoid prefetch overhead/stalls.

    def build_transforms(self, hyp=None):
        """
        Override to disable MixUp, CutMix, CopyPaste for multiview training.
        These augmentations don't work with our PE+image concatenation (7 channels).
        """
        from ultralytics.data.augment import Compose, LetterBox, Format, RandomFlip

        if hyp is None:
            hyp = self.args

        if self.augment:
            # Only safe geometric augmentations for multiview.
            # HSV is applied separately to RGB channels before PE concatenation.
            transforms = Compose([
                LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
                RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=[]),
                RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=[]),
            ])
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def _load_pe_from_disk(self, im_file):
        """
        Load position encoding from disk.

        Args:
            im_file (str): Path to image file.

        Returns:
            torch.Tensor or None: (H, W, 4) position encoding, or None if not found.
        """
        im_path = Path(im_file)

        # Enforce pes_npz-only loading
        pe = self._load_pe_from_lmdb(im_path)
        if pe is not None:
            return pe

        pe_filename = im_path.stem + "_pe.npy"
        pes_npz_dir = im_path.parents[2] / 'pes_npz' / im_path.parent.name
        pe_path = pes_npz_dir / pe_filename.replace("_pe.npy", "_pe.npz")
        if not pe_path.exists():
            raise FileNotFoundError(f"PE npz not found: {pe_path}")
        try:
            pe = np.load(pe_path)["pe"]
            # Expecting (H, W, 4) for sin/cos encoding
            if pe.ndim == 3 and pe.shape[2] in [3, 4]:
                if pe.shape[2] == 3:
                    pe = np.pad(pe, [(0, 0), (0, 0), (0, 1)], mode='constant')
                return torch.from_numpy(pe).float()
        except Exception as e:
            raise RuntimeError(f"Error loading PE from {pe_path}: {e}") from e

        return None

    def _lmdb_path_for_image(self, im_path: Path) -> Path | None:
        if not self._pe_lmdb:
            return None
        cam_dir = im_path.parents[2]
        if isinstance(self._pe_lmdb, str):
            if self._pe_lmdb in ("fp16", "fp32"):
                return cam_dir / f"pes_{self._pe_lmdb}.lmdb"
            return Path(self._pe_lmdb)
        return None

    def _load_pe_from_lmdb(self, im_path: Path):
        lmdb_path = self._lmdb_path_for_image(im_path)
        if not lmdb_path:
            return None
        if not lmdb_path.exists():
            return None

        try:
            import lmdb
            import io
        except Exception:
            return None

        lmdb_key = os.path.join(im_path.parent.name, f"{im_path.stem}_pe.npy")
        env = self._pe_lmdb_envs.get(str(lmdb_path))
        if env is None:
            env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=256,
            )
            self._pe_lmdb_envs[str(lmdb_path)] = env

        with env.begin(write=False) as txn:
            buf = txn.get(lmdb_key.encode("utf-8"))
            if buf is None:
                return None
            pe = np.load(io.BytesIO(buf), allow_pickle=False)
            return torch.from_numpy(pe).float()

    def _build_separate_mode_im_files(self):
        """
        Build interleaved im_files list from separate camera directories.
        Assumes matching filenames across cameras (e.g., 23-1_XXXXXX.jpg and 23-2_XXXXXX.jpg)

        Uses cache to avoid rebuilding on every initialization.
        """
        from tqdm import tqdm
        import pickle
        import random
        from ultralytics.data.utils import get_hash

        # Get base path from data config or infer from im_files
        # im_files[0] is like: /home/hetao/graduate/data/MDMT-yolo/camera1/images/train/xxx.jpg
        # We need: /home/hetao/graduate/data/MDMT-yolo
        if self.im_files:
            # Use data['path'] if available, otherwise infer from im_files
            if hasattr(self, 'data') and self.data and 'path' in self.data:
                base_path = Path(self.data['path'])
            else:
                # Infer: go up 3 levels from camera1/images/train/xxx.jpg
                # parent: train, parent[1]: images, parent[2]: camera1, parent[3]: MDMT-yolo
                base_path = Path(self.im_files[0]).parents[3]
        else:
            base_path = Path.cwd()

        mode = Path(self.im_files[0]).parent.name if self.im_files else "train"
        # e.g., train

        # Fractional subset support (use separate cache to avoid overwriting full cache)
        fraction = float(getattr(self, "fraction", 1.0) or 1.0)
        fraction = min(max(fraction, 0.0), 1.0)

        def _apply_fraction(im_files, label_files):
            if fraction >= 1.0 or fraction <= 0.0:
                return im_files, label_files
            n_total = len(im_files)
            n_keep = max(1, int(n_total * fraction))
            rng = random.Random(0)
            indices = list(range(n_total))
            rng.shuffle(indices)
            indices = indices[:n_keep]
            im_files = [im_files[i] for i in indices]
            label_files = [label_files[i] for i in indices]
            return im_files, label_files

        # Define cache paths for multiview file list
        base_cache_key = f"{'_'.join(self._camera_dirs)}_{self.num_views}views"
        cache_key = base_cache_key
        if 0.0 < fraction < 1.0:
            frac_tag = f"frac{fraction:.4f}".replace(".", "p")
            cache_key = f"{cache_key}_{frac_tag}"
        cache_path = base_path / f".{mode}_{cache_key}_multiview_cache.pkl"
        full_cache_path = base_path / f".{mode}_{base_cache_key}_multiview_cache.pkl"

        # Try to load from cache first
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # Verify cache is still valid
                cached_hash = get_hash([str(base_path / cam_dir / "images" / mode) for cam_dir in self._camera_dirs])
                if cached_data.get('hash') == cached_hash:
                    self.im_files = cached_data['im_files']
                    self.label_files = cached_data['label_files']
                    LOGGER.info(f"{self.prefix}Loaded {len(self.im_files)} multiview images from cache {cache_path.name}")
                    # Still need to update labels, ni, etc.
                    self._rebuild_labels_after_file_load()
                    return
                else:
                    LOGGER.info(f"{self.prefix}Multiview cache outdated, rebuilding...")
            except Exception as e:
                LOGGER.warning(f"{self.prefix}Error loading multiview cache: {e}, rebuilding...")

        # If fractional cache missing, try to reuse full cache then slice
        if 0.0 < fraction < 1.0 and full_cache_path.exists():
            try:
                with open(full_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                cached_hash = get_hash([str(base_path / cam_dir / "images" / mode) for cam_dir in self._camera_dirs])
                if cached_data.get('hash') == cached_hash:
                    im_files = cached_data['im_files']
                    label_files = cached_data['label_files']
                    im_files, label_files = _apply_fraction(im_files, label_files)
                    self.im_files = im_files
                    self.label_files = label_files
                    LOGGER.info(
                        f"{self.prefix}Loaded {len(self.im_files)} multiview images from full cache and "
                        f"applied fraction={fraction}"
                    )
                    # Save fractional cache for reuse
                    try:
                        cached_data = {
                            'hash': cached_hash,
                            'im_files': self.im_files,
                            'label_files': self.label_files,
                            'num_views': self.num_views,
                            'camera_dirs': self._camera_dirs,
                        }
                        with open(cache_path, 'wb') as f:
                            pickle.dump(cached_data, f)
                        LOGGER.info(f"{self.prefix}Saved multiview cache to {cache_path.name}")
                    except Exception as e:
                        LOGGER.warning(f"{self.prefix}Failed to save multiview cache: {e}")
                    self._rebuild_labels_after_file_load()
                    return
            except Exception as e:
                LOGGER.warning(f"{self.prefix}Error loading full multiview cache: {e}, rebuilding...")

        # Cache miss or invalid, build from scratch
        LOGGER.info(f"{self.prefix}Building multiview dataset from scratch...")

        self.im_files = []
        self.label_files = []

        # Get files from first camera to establish order
        cam0_path = base_path / self._camera_dirs[0] / "images" / mode
        cam0_labels = base_path / self._camera_dirs[0] / "labels" / mode

        if not cam0_path.exists():
            raise ValueError(f"Camera image path not found: {cam0_path}")

        # Get all image files from camera 0
        cam0_images = sorted(cam0_path.glob("*.jpg")) + sorted(cam0_path.glob("*.png"))

        for cam0_img in tqdm(cam0_images, desc=f"Building {mode} multiview dataset"):
            stem = cam0_img.stem  # e.g., 23-1_00000001

            # Extract scene prefix and sequence number
            # E.g., "23-1_00000001" -> ("23-", "00000001")
            scene_prefix, seq_num = self._get_base_filename(stem)

            # Build interleaved list
            scene_images = []
            scene_labels = []

            for cam_dir in self._camera_dirs:
                # Find corresponding image in this camera's directory
                cam_path = base_path / cam_dir / "images" / mode
                cam_labels_path = base_path / cam_dir / "labels" / mode

                # Try to find matching image using scene prefix and sequence
                cam_img = self._find_matching_image(cam_path, scene_prefix, seq_num)
                if cam_img is None:
                    break  # Skip this scene if any view is missing

                scene_images.append(str(cam_img))

                # Find corresponding label
                label_file = cam_labels_path / f"{cam_img.stem}.txt"
                scene_labels.append(str(label_file) if label_file.exists() else "")

            # Only add if all views have images
            if len(scene_images) == self.num_views:
                self.im_files.extend(scene_images)
                self.label_files.extend(scene_labels)

        # Apply fraction after building full list
        self.im_files, self.label_files = _apply_fraction(self.im_files, self.label_files)

        # Save to cache (fractional cache is separate from full cache)
        try:
            cache_hash = get_hash([str(base_path / cam_dir / "images" / mode) for cam_dir in self._camera_dirs])
            cached_data = {
                'hash': cache_hash,
                'im_files': self.im_files,
                'label_files': self.label_files,
                'num_views': self.num_views,
                'camera_dirs': self._camera_dirs,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            LOGGER.info(f"{self.prefix}Saved multiview cache to {cache_path.name}")
        except Exception as e:
            LOGGER.warning(f"{self.prefix}Failed to save multiview cache: {e}")

        # Rebuild labels and other data
        self._rebuild_labels_after_file_load()

    def _rebuild_labels_after_file_load(self):
        """Rebuild labels, ni, and cache-related lists after im_files are set."""
        # Update self.ni (number of images)
        # This must be set before rebuilding ims and npy_files
        self.ni = len(self.im_files)

        # Rebuild self.labels list to match the new im_files
        # Use cache_labels mechanism for proper label parsing
        from ultralytics.data.utils import img2label_paths, verify_image_label
        from ultralytics.data.utils import NUM_THREADS
        from itertools import repeat
        from ultralytics.utils import TQDM
        from multiprocessing.pool import ThreadPool

        # Rebuild label_files based on new im_files
        self.label_files = img2label_paths(self.im_files)

        # Use verify_image_label to properly parse labels with all required fields
        nkpt, ndim = self.data.get("kpt_shape", (0, 0)) if hasattr(self, 'data') and self.data else (0, 0)
        self.labels = []
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(False),  # use_keypoints
                    repeat(len(self.data["names"]) if hasattr(self, 'data') and self.data and "names" in self.data else 4),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(False),  # single_cls
                ),
            )
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in results:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    self.labels.append({
                        "im_file": im_file,
                        "shape": shape,
                        "cls": lb[:, 0:1],  # n, 1
                        "bboxes": lb[:, 1:],  # n, 4
                        "segments": segments,
                        "keypoints": keypoint,
                        "normalized": True,
                        "bbox_format": "xywh",
                    })

        LOGGER.info(f"{self.prefix}Rebuilt labels for {nf} images, {nm + ne} backgrounds, {nc} corrupt")

        # Rebuild cache-related lists to match new im_files
        self.ims = [None] * self.ni
        self.im_hw0 = [None] * self.ni
        self.im_hw = [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

        # Re-apply cache if enabled
        # The cache attribute is set by BaseDataset.__init__ before we rebuild im_files
        if hasattr(self, 'cache') and self.cache:
            LOGGER.info(f"{self.prefix}Rebuilding cache for {self.ni} multiview images...")
            if self.cache == "ram" and self.check_cache_ram():
                self.cache_images()
            elif self.cache == "disk" and self.check_cache_disk():
                self.cache_images()

    def _parse_label_file(self, label_file):
        """
        Parse YOLO format label file.

        Args:
            label_file: Path to label file

        Returns:
            cls: (N, 1) class indices
            bboxes: (N, 4) bounding boxes in xywh format
        """
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                return np.array([], dtype=np.float32).reshape(0, 1), np.array([], dtype=np.float32).reshape(0, 4)

            data = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        data.append([float(x) for x in parts])

            if not data:
                return np.array([], dtype=np.float32).reshape(0, 1), np.array([], dtype=np.float32).reshape(0, 4)

            data = np.array(data, dtype=np.float32)
            cls = data[:, 0:1].reshape(-1, 1)  # (N, 1)
            bboxes = data[:, 1:5]  # (N, 4)
            return cls, bboxes

        except Exception as e:
            warnings.warn(f"Error parsing label file {label_file}: {e}")
            return np.array([], dtype=np.float32).reshape(0, 1), np.array([], dtype=np.float32).reshape(0, 4)

    def _get_base_filename(self, stem, camera_id=None):
        """
        Extract base filename that can be used to match across cameras.
        E.g., "23-1_00000001" -> "23_00000001" (for matching with "23-2_00000001")

        Args:
            stem: Image filename stem (e.g., "23-1_00000001")
            camera_id: Current camera ID (e.g., "1" in "23-1_00000001"), used for constructing pattern

        Returns:
            Tuple of (scene_prefix, sequence_number)
            E.g., ("23-", "00000001") for "23-1_00000001"
        """
        # Format: XX-N_XXXXXX where N is camera ID
        # We need to extract the scene prefix (XX-) and sequence (XXXXXX)
        parts = stem.split('-')
        if len(parts) >= 2:
            # parts[0] = "23", parts[1] = "1_00000001"
            scene_id = parts[0]
            rest = parts[1]
            # rest format: "N_XXXXXX"
            sub_parts = rest.split('_')
            if len(sub_parts) >= 2:
                seq_num = sub_parts[-1]
                return f"{scene_id}-", seq_num

        # Fallback: return original stem and None
        return stem, None

    def _find_matching_image(self, cam_path, scene_prefix, seq_num):
        """
        Find image file in cam_path that matches the scene and sequence.
        E.g., scene_prefix="23-", seq_num="00000001" matches "23-2_00000001.jpg"

        The target camera ID is embedded in cam_path (e.g., camera2), so we just
        need to find any file matching {scene_prefix}*_{seq_num}.jpg
        """
        # Pattern: {scene_prefix}*_{seq_num}.jpg
        # E.g., "23-*_00000001.jpg" should match "23-2_00000001.jpg"
        for ext in ['.jpg', '.png', '.jpeg']:
            pattern = f"{scene_prefix}*_{seq_num}{ext}"
            candidates = list(cam_path.glob(pattern))
            if candidates:
                # Expect exactly one match per camera directory
                return candidates[0]

        return None
        
    def __len__(self):
        # Assume images are ordered: Scene1_View1, Scene1_View2, ..., Scene2_View1, ...
        return len(self.im_files) // self.num_views

    def load_coords(self, im_file, shape):
        """
        Load position encoding for an image using LRU cache.
        Falls back to disk loading and random generation if needed.

        Args:
            im_file (str): Path to image file.
            shape (tuple): (h, w) of the image.

        Returns:
            torch.Tensor: (h, w, 4) position encoding (sin_y, cos_y, sin_x, cos_x).
        """
        h, w = shape

        # Load from disk (no cache)
        pe = self._load_pe_from_disk(im_file)
        if pe is not None:
            # Resize if necessary
            if pe.shape[:2] != shape:
                pe_numpy = pe.numpy()
                pe_resized = cv2.resize(pe_numpy, (w, h), interpolation=cv2.INTER_LINEAR)
                return torch.from_numpy(pe_resized).float()
            return pe

        # No PE file found, use random sinusoidal position encoding
        if not self._missing_pe_warned:
            warnings.warn(
                f"Position encoding file not found for {im_file}. "
                f"Using random sinusoidal PE as fallback. "
                f"This warning will only show once."
            )
            self._missing_pe_warned = True

        return self._generate_random_pe(h, w)

    def _generate_random_pe(self, h, w):
        """
        Generate random sinusoidal position encoding as fallback.

        Args:
            h, w: Image dimensions

        Returns:
            torch.Tensor: (h, w, 4) random PE
        """
        # Create normalized coordinates
        y_coords = torch.linspace(0, 2 * np.pi, h)
        x_coords = torch.linspace(0, 2 * np.pi, w)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Sinusoidal encoding with some random phase offset
        phase_y = torch.rand(1).item() * 2 * np.pi
        phase_x = torch.rand(1).item() * 2 * np.pi

        pe = torch.stack([
            torch.sin(yy + phase_y),
            torch.cos(yy + phase_y),
            torch.sin(xx + phase_x),
            torch.cos(xx + phase_x)
        ], dim=-1)  # (h, w, 4)

        return pe

    def __getitem__(self, index):
        """
        Returns a dictionary containing:
        - img: (V, C, H, W)
        - coords: (V, H, W, 4) position encoding (sin_y, cos_y, sin_x, cos_x)
        - cls: (N, 1) flattened
        - bboxes: (N, 4) flattened
        - batch_idx: (N, 1) mapping to view index in the batch (0..V-1)

        Note: For multiview data, we apply consistent augmentations across views
        by using the same random seed for each view.
        """
        import random

        start_idx = index * self.num_views

        # For multiview, we need consistent augmentations across views
        # We'll process views with the same random seed
        views_data = []
        base_seed = random.randint(0, 2**32 - 1)
        torch_state = torch.random.get_rng_state()
        py_state = random.getstate()
        np_state = np.random.get_state()

        for v in range(self.num_views):
            # Set same seed for each view to get consistent augmentations
            torch.manual_seed(base_seed)
            random.seed(base_seed)
            np.random.seed(base_seed)

            # Get label data (this calls get_image_and_label which loads the image)
            label = super().get_image_and_label(start_idx + v)

            img = label['img']  # (H, W, 3) numpy array - already resized by load_image
            if self.augment:
                # Apply HSV only to RGB channels; keep PE unchanged.
                if hasattr(self, "args") and (self.args.hsv_h or self.args.hsv_s or self.args.hsv_v):
                    from ultralytics.data.augment import RandomHSV

                    img = RandomHSV(hgain=self.args.hsv_h, sgain=self.args.hsv_s, vgain=self.args.hsv_v)(
                        {"img": img}
                    )["img"]

            if self._pe_disable:
                # Skip PE loading entirely (ablation)
                label['img'] = img
                label = self.transforms(label)
                # Format converts to (C, H, W)
                _, h_new, w_new = label['img'].shape
                label['pe'] = torch.zeros((4, h_new, w_new), dtype=label['img'].dtype)
            else:
                # Load PE at the RESIZED image size (not original size!)
                # Because load_image already resized the image before transforms
                im_file = self.im_files[start_idx + v]
                res_h, res_w = label['resized_shape']  # (H, W) - the size AFTER load_image resize
                pe = self.load_coords(im_file, (res_h, res_w))  # (H, W, 4)

                # Concatenate PE with image: (H, W, 3+4) -> (H, W, 7)
                # This ensures PE and image go through the SAME transforms
                img_with_pe = np.concatenate([img, pe.numpy()], axis=2)  # (H, W, 7)

                # Replace img in label with concatenated version
                label['img'] = img_with_pe

                # Apply transforms (will apply to both image and PE channels)
                label = self.transforms(label)

                # Split back into image and PE
                transformed_img_with_pe = label['img']  # (C, H_new, W_new) where C=7 after transforms
                # Format converts to (C, H, W), we need to split channels
                label['img'] = transformed_img_with_pe[:3]  # First 3 channels: RGB
                label['pe'] = transformed_img_with_pe[3:]  # Last 4 channels: PE

            views_data.append(label)

        # Restore randomness for next sample
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)

        # Stack images: (V, C, H, W)
        imgs = [d['img'] for d in views_data]
        imgs_stack = torch.stack(imgs)

        # Stack coords: (V, C, H, W) where C=4, then permute to (V, H, W, 4)
        coords_list = [d['pe'] for d in views_data]
        coords_stack = torch.stack(coords_list).permute(0, 2, 3, 1)  # (V, H, W, 4)

        # Aggregate labels
        # Note: ratio_pad may be removed by augmentations (e.g., Mosaic)
        # For training, we can compute it from ori_shape and resized_shape
        ratio_pads = []
        for i, d in enumerate(views_data):
            if 'ratio_pad' in d:
                ratio_pads.append(d['ratio_pad'])
            else:
                # Compute ratio_pad from ori_shape and resized_shape
                ori_h, ori_w = d['ori_shape']
                res_h, res_w = d['resized_shape']
                ratio = (res_h / ori_h, res_w / ori_w)
                # Padding is unknown without LetterBox info, use (0, 0) as fallback
                ratio_pads.append((ratio, (0, 0)))

        data = {
            'img': imgs_stack,
            'coords': coords_stack,
            'cls': [d['cls'] for d in views_data],
            'bboxes': [d['bboxes'] for d in views_data],
            'im_file': [d['im_file'] for d in views_data],
            'ori_shape': [d['ori_shape'] for d in views_data],
            'resized_shape': [d['resized_shape'] for d in views_data],
            'ratio_pad': ratio_pads,
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
                'coords': (N, V, H, W, 4) position encoding
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

        # Stack coords: (N, V, H, W, 4)
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

        # Add im_file for plotting/training sample visualization
        # Flatten the list of lists: batch samples x V views
        im_files = []
        ori_shapes = []
        resized_shapes = []
        ratio_pads = []
        for sample in batch:
            im_files.extend(sample['im_file'])
            ori_shapes.extend(sample['ori_shape'])
            resized_shapes.extend(sample['resized_shape'])
            ratio_pads.extend(sample['ratio_pad'])
        new_batch['im_file'] = im_files
        new_batch['ori_shape'] = ori_shapes
        new_batch['resized_shape'] = resized_shapes
        new_batch['ratio_pad'] = ratio_pads

        return new_batch
