# Ultralytics Multiview Validator
from ultralytics.models.yolo.detect import DetectionValidator
from pathlib import Path

import cv2
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import LOGGER, colorstr, callbacks, emojis
from ultralytics.utils.patches import imread
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import attempt_compile, select_device, unwrap_model
import numpy as np
import torch
from ultralytics.utils import ops


class MultiviewValidator(DetectionValidator):
    """
    A custom validator for YOLO Multiview models.
    Computes metrics separately for each camera view.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize multiview validator with per-view metrics tracking."""
        super().__init__(dataloader, save_dir, args, _callbacks)

        # Get num_views from data config
        self.num_views = getattr(args, 'num_views', None) or 2
        # Force-enable first-batch val visualization by default
        self._force_val_plot = True
        def _force_plot_first(v):
            if getattr(v, "_force_val_plot", False) and v.batch_i == 0:
                v.args.plots = True
        self.callbacks["on_val_batch_start"].append(_force_plot_first)
        # Only print per-class details during standalone test/val when requested
        if not hasattr(self.args, "verbose_test_only"):
            self.args.verbose_test_only = False

        # Create separate metrics for each view
        self.view_metrics = [DetMetrics() for _ in range(self.num_views)]
        self.view_names = [f"camera{i+1}" for i in range(self.num_views)]
        # Optional overlap/non-overlap metrics
        self.overlap_eval = bool(getattr(self.args, "overlap_eval", False))
        if self.overlap_eval:
            self.overlap_metrics = [DetMetrics() for _ in range(self.num_views)]
            self.nonoverlap_metrics = [DetMetrics() for _ in range(self.num_views)]

    def init_metrics(self, model):
        """Initialize metrics for all views."""
        super().init_metrics(model)

        # Initialize each view's metrics
        for i, metrics in enumerate(self.view_metrics):
            metrics.names = model.names
            metrics.nc = len(model.names)
        if self.overlap_eval:
            for m in self.overlap_metrics + self.nonoverlap_metrics:
                m.names = model.names
                m.nc = len(model.names)

    @staticmethod
    def _build_overlap_mask(coords_view, const_pe_eps=1e-6):
        const0 = coords_view.new_tensor([1.0, 0.0, 1.0, 0.0])
        const1 = coords_view.new_tensor([-1.0, 0.0, -1.0, 0.0])
        diff0 = (coords_view - const0).abs().amax(dim=-1)
        diff1 = (coords_view - const1).abs().amax(dim=-1)
        valid0 = diff0 > const_pe_eps
        valid1 = diff1 > const_pe_eps
        return valid0, valid1

    @staticmethod
    def _filter_by_mask_boxes_xyxy(boxes, mask):
        if boxes.numel() == 0:
            return boxes, torch.zeros((0,), dtype=torch.bool, device=boxes.device)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = ((x1 + x2) * 0.5).round().long()
        cy = ((y1 + y2) * 0.5).round().long()
        h, w = mask.shape
        cx = cx.clamp(0, w - 1)
        cy = cy.clamp(0, h - 1)
        keep = mask[cy, cx]
        return boxes[keep], keep

    def preprocess(self, batch: dict):
        """Preprocess batch and align dtype with model (handles AMP/FP16)."""
        batch = super().preprocess(batch)
        try:
            dtype = next(self.model.parameters()).dtype
        except Exception:
            return batch
        batch["img"] = batch["img"].to(dtype)
        if "coords" in batch:
            batch["coords"] = batch["coords"].to(dtype)
        if getattr(self.args, "debug_dtype", False) and not getattr(self, "_dtype_logged", False):
            self._dtype_logged = True
            img_dtype = batch["img"].dtype
            coords_dtype = getattr(batch.get("coords"), "dtype", None)
            LOGGER.info(
                f"[debug_dtype] model={dtype} img={img_dtype} coords={coords_dtype} half_arg={self.args.half}"
            )
        return batch

    def _tile_positions(self, h, w, tile, stride):
        if h <= tile and w <= tile:
            return [(0, 0, h, w)]
        ys = list(range(0, max(h - tile, 0) + 1, stride))
        xs = list(range(0, max(w - tile, 0) + 1, stride))
        if ys[-1] != h - tile:
            ys.append(h - tile)
        if xs[-1] != w - tile:
            xs.append(w - tile)
        return [(y, x, y + tile, x + tile) for y in ys for x in xs]

    def _tile_one_scene(self, imgs, coords, tile, stride):
        # imgs: (V, C, H, W), coords: (1, V, H, W, 4) or None
        _, _, h, w = imgs.shape
        windows = []
        for y1, x1, y2, x2 in self._tile_positions(h, w, tile, stride):
            wimgs = imgs[:, :, y1:y2, x1:x2]
            wcoords = None
            if coords is not None:
                wcoords = coords[:, :, y1:y2, x1:x2, :]
            windows.append((wimgs, wcoords, (x1, y1)))
        return windows

    def _merge_tile_preds(self, preds_list, offsets):
        merged = []
        for pred, (x0, y0) in zip(preds_list, offsets):
            if pred is None or pred.numel() == 0:
                continue
            p = pred.clone()
            p[:, [0, 2]] += x0
            p[:, [1, 3]] += y0
            merged.append(p)
        if merged:
            return torch.cat(merged, dim=0)
        return torch.zeros((0, 6), device=preds_list[0].device if preds_list else "cpu")

    @staticmethod
    def _map_to_letterbox(preds, ratio_pad):
        # ratio_pad: ((r_w, r_h), (padw, padh)) or (r, (padw, padh))
        if preds.numel() == 0:
            return preds
        ratio, pad = ratio_pad
        if isinstance(ratio, (list, tuple)):
            r_w, r_h = ratio
        else:
            r_w = r_h = float(ratio)
        padw, padh = pad
        preds[:, [0, 2]] = preds[:, [0, 2]] * r_w + padw
        preds[:, [1, 3]] = preds[:, [1, 3]] * r_h + padh
        return preds

    def _nms_merged(self, merged, conf, iou, max_det):
        if merged.numel() == 0:
            return merged
        scores = merged[:, 4]
        keep = scores >= conf
        merged = merged[keep]
        if merged.numel() == 0:
            return merged
        boxes = merged[:, :4]
        scores = merged[:, 4]
        cls = merged[:, 5]
        keep_idx = TorchNMS.batched_nms(boxes, scores, cls, iou)
        return merged[keep_idx][:max_det]

    def _tiled_forward(self, imgs, coords=None):
        tile = int(getattr(self.args, "val_tile_size", 640))
        stride = int(getattr(self.args, "val_tile_stride", 480))
        disable_fusion = bool(getattr(self.args, "val_tile_disable_fusion", False))
        # imgs: (V, C, H, W), coords: (1, V, H, W, 4) or None
        windows = self._tile_one_scene(imgs, coords, tile, stride)
        preds_tiles = []
        offsets = []
        for wimgs, wcoords, (x0, y0) in windows:
            wcoords_in = None
            if not disable_fusion:
                wcoords_in = wcoords
            with torch.no_grad():
                raw = self.model(wimgs, coords=wcoords_in)
            preds_tiles.append(self.postprocess(raw))
            offsets.append((x0, y0))
        return preds_tiles, offsets

    def _save_tiled_vis(self, imgs_scene, coords_scene, offsets, batch, bi):
        if not getattr(self.args, "plots", False):
            return
        if not hasattr(self, "_val_tile_vis_count"):
            self._val_tile_vis_count = 0
        if self._val_tile_vis_count >= 12:
            return
        out_dir = self.save_dir / "val_tiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs_np = imgs_scene.detach().cpu().float().numpy()
        if imgs_np.max() <= 1.0:
            imgs_np = (imgs_np * 255).astype(np.uint8)
        coords_np = None
        if coords_scene is not None:
            coords_np = coords_scene.detach().cpu().float().numpy()[0]  # (V, H, W, 4)

        V = imgs_np.shape[0]
        tile = int(getattr(self.args, "val_tile_size", 640))
        h, w = imgs_np.shape[2], imgs_np.shape[3]
        for ti, (x0, y0) in enumerate(offsets):
            if self._val_tile_vis_count >= 12:
                break
            for vi in range(V):
                img = imgs_np[vi].transpose(1, 2, 0)
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                crop = img[y0:y1, x0:x1]
                # If crop out of bounds, skip
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                stem = Path(batch["im_file"][bi * V + vi]).stem
                base = out_dir / f"val_b{bi}_t{ti}_v{vi}_{stem}"
                cv2.imwrite(str(base.with_suffix(".jpg")), crop)
                if coords_np is None:
                    continue
                pe = coords_np[vi, y0:y1, x0:x1, :]
                pe_norm = ((pe + 1) / 2 * 255).astype(np.uint8)
                pe_vis = np.stack([pe_norm[:, :, 0], pe_norm[:, :, 1], pe_norm[:, :, 2]], axis=2)
                if pe_vis.shape[:2] != crop.shape[:2]:
                    pe_vis = cv2.resize(pe_vis, (crop.shape[1], crop.shape[0]))
                overlay = cv2.addWeighted(crop, 0.6, pe_vis, 0.4, 0)
                cv2.imwrite(str(base.with_name(base.name + "_pe_overlay.jpg")), overlay)
                self._val_tile_vis_count += 1

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.
        Routes predictions to the appropriate view based on batch_idx.

        Args:
            preds: List of predictions from the model.
            batch: Batch data containing ground truth.
        """
        # Get total number of samples in this batch
        batch_size = len(preds)

        batch_coords = batch.get("coords", None)
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0

            # Determine which view this sample belongs to
            # batch_idx should have been set in dataset to track original order
            view_idx = si % self.num_views

            # Update both overall metrics and view-specific metrics
            stat = {
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
            }

            # Update overall metrics (for backward compatibility)
            self.metrics.update_stats(stat)

            # Update view-specific metrics
            self.view_metrics[view_idx].update_stats(stat)

            # Optional overlap / non-overlap metrics
            if self.overlap_eval and batch_coords is not None:
                batch_idx = si // self.num_views
                coords_view = batch_coords[batch_idx, view_idx]  # (H, W, 4)
                valid0, valid1 = self._build_overlap_mask(coords_view)
                overlap_mask = valid0 if view_idx == 0 else valid1
                nonoverlap_mask = ~overlap_mask

                # Filter GT
                gt_boxes = pbatch["bboxes"]
                gt_boxes_overlap, keep_gt_overlap = self._filter_by_mask_boxes_xyxy(gt_boxes, overlap_mask)
                gt_boxes_non, keep_gt_non = self._filter_by_mask_boxes_xyxy(gt_boxes, nonoverlap_mask)
                gt_cls_overlap = pbatch["cls"][keep_gt_overlap] if keep_gt_overlap.numel() else pbatch["cls"][:0]
                gt_cls_non = pbatch["cls"][keep_gt_non] if keep_gt_non.numel() else pbatch["cls"][:0]

                # Filter preds
                pred_boxes = predn["bboxes"]
                pred_boxes_overlap, keep_pred_overlap = self._filter_by_mask_boxes_xyxy(pred_boxes, overlap_mask)
                pred_boxes_non, keep_pred_non = self._filter_by_mask_boxes_xyxy(pred_boxes, nonoverlap_mask)
                pred_conf_overlap = predn["conf"][keep_pred_overlap] if keep_pred_overlap.numel() else predn["conf"][:0]
                pred_cls_overlap = predn["cls"][keep_pred_overlap] if keep_pred_overlap.numel() else predn["cls"][:0]
                pred_conf_non = predn["conf"][keep_pred_non] if keep_pred_non.numel() else predn["conf"][:0]
                pred_cls_non = predn["cls"][keep_pred_non] if keep_pred_non.numel() else predn["cls"][:0]

                pbatch_overlap = dict(pbatch)
                pbatch_overlap["bboxes"] = gt_boxes_overlap
                pbatch_overlap["cls"] = gt_cls_overlap
                pbatch_non = dict(pbatch)
                pbatch_non["bboxes"] = gt_boxes_non
                pbatch_non["cls"] = gt_cls_non

                pred_overlap = {"bboxes": pred_boxes_overlap, "conf": pred_conf_overlap, "cls": pred_cls_overlap}
                pred_non = {"bboxes": pred_boxes_non, "conf": pred_conf_non, "cls": pred_cls_non}

                cls_o = pbatch_overlap["cls"].cpu().numpy()
                no_pred_o = pred_overlap["cls"].shape[0] == 0
                stat_o = {
                    **self._process_batch(pred_overlap, pbatch_overlap),
                    "target_cls": cls_o,
                    "target_img": np.unique(cls_o),
                    "conf": np.zeros(0) if no_pred_o else pred_overlap["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred_o else pred_overlap["cls"].cpu().numpy(),
                }
                self.overlap_metrics[view_idx].update_stats(stat_o)

                cls_n = pbatch_non["cls"].cpu().numpy()
                no_pred_n = pred_non["cls"].shape[0] == 0
                stat_n = {
                    **self._process_batch(pred_non, pbatch_non),
                    "target_cls": cls_n,
                    "target_img": np.unique(cls_n),
                    "conf": np.zeros(0) if no_pred_n else pred_non["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred_n else pred_non["cls"].cpu().numpy(),
                }
                self.nonoverlap_metrics[view_idx].update_stats(stat_n)

            # Evaluate with confusion matrix (overall only)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save predictions (same for all views)
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{pbatch['im_file'].stem}.txt",
                )

    def finalize_metrics(self):
        """Finalize metrics for all views."""
        # Call parent to finalize overall metrics
        super().finalize_metrics()

        # Finalize each view's metrics
        for i, view_metric in enumerate(self.view_metrics):
            view_metric.speed = self.speed
            view_metric.save_dir = self.save_dir / f"view_{i+1}"

    def get_stats(self):
        """
        Calculate and return metrics statistics for all views.
        Returns combined stats with view-specific metrics.
        """
        # Process overall metrics
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()

        # Process each view's metrics
        view_stats = {}
        for i, view_metric in enumerate(self.view_metrics):
            view_dir = self.save_dir / f"view_{i+1}"
            view_dir.mkdir(parents=True, exist_ok=True)
            view_metric.process(save_dir=view_dir, plot=self.args.plots, on_plot=self.on_plot)
            view_metric.clear_stats()

            # Store view-specific results
            view_results = view_metric.results_dict
            for key, value in view_results.items():
                view_stats[f"{key}_view{i+1}"] = value

        # Combine overall and view-specific stats
        stats = self.metrics.results_dict
        stats.update(view_stats)

        # Add overlap/non-overlap stats if enabled
        if self.overlap_eval:
            for i in range(self.num_views):
                vdir = self.save_dir / f"view_{i+1}"
                vdir.mkdir(parents=True, exist_ok=True)
                self.overlap_metrics[i].process(save_dir=vdir / "overlap", plot=False, on_plot=self.on_plot)
                self.nonoverlap_metrics[i].process(save_dir=vdir / "nonoverlap", plot=False, on_plot=self.on_plot)
                for k, v in self.overlap_metrics[i].results_dict.items():
                    stats[f"{k}_overlap_view{i+1}"] = v
                for k, v in self.nonoverlap_metrics[i].results_dict.items():
                    stats[f"{k}_nonoverlap_view{i+1}"] = v
                self.overlap_metrics[i].clear_stats()
                self.nonoverlap_metrics[i].clear_stats()

        return stats

    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        if not getattr(self.args, "val_tile", False):
            return super().__call__(trainer, model)

        # Use standard BaseValidator setup
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod
            model = model.half() if self.args.half else model.float()
            try:
                self.args.half = next(model.parameters()).dtype == torch.float16
            except Exception:
                pass
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            from ultralytics.nn.autobackend import AutoBackend
            from ultralytics.data.utils import check_det_dataset, check_cls_dataset

            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.model = model
            self.device = model.device
            self.args.half = model.fp16
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")
            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))
            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        from ultralytics.utils import TQDM
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []
        # Always allow first-batch visualization for val
        self.args.plots = True

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
                if isinstance(model, torch.nn.Module):
                    try:
                        model_dtype = next(model.parameters()).dtype
                    except Exception:
                        model_dtype = None
                    if model_dtype is not None:
                        if isinstance(batch.get("img"), torch.Tensor) and batch["img"].dtype != model_dtype:
                            batch["img"] = batch["img"].to(model_dtype)
                        if isinstance(batch.get("coords"), torch.Tensor) and batch["coords"].dtype != model_dtype:
                            batch["coords"] = batch["coords"].to(model_dtype)

            with dt[1]:
                preds = []
                imgs = batch["img"]
                coords = batch.get("coords")
                use_original = bool(getattr(self.args, "val_tile_original", False))
                dataset = getattr(self.dataloader, "dataset", None)
                V = self.num_views
                B = imgs.shape[0] // V
                for bi in range(B):
                    if use_original:
                        imgs_scene_list = []
                        coords_list = []
                        target_h = None
                        target_w = None
                        for vi in range(V):
                            idx = bi * V + vi
                            im_path = batch["im_file"][idx]
                            im = imread(im_path, flags=getattr(dataset, "cv2_flag", 1))
                            if im is None:
                                im = np.zeros((self.args.imgsz, self.args.imgsz, 3), dtype=np.uint8)
                            if im.ndim == 2:
                                im = np.repeat(im[..., None], 3, axis=2)
                            if im.shape[2] == 3:
                                im = im[..., ::-1]  # BGR->RGB
                            if target_h is None:
                                target_h, target_w = im.shape[:2]
                            if im.shape[:2] != (target_h, target_w):
                                im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                            imgs_scene_list.append(im)
                            if coords is not None and dataset is not None and hasattr(dataset, "load_coords"):
                                pe = dataset.load_coords(im_path, (target_h, target_w))
                                coords_list.append(pe)
                        imgs_scene = torch.stack(
                            [torch.from_numpy(x).permute(2, 0, 1) for x in imgs_scene_list]
                        ).to(self.device)
                        imgs_scene = (imgs_scene.half() if self.args.half else imgs_scene.float()) / 255.0
                        coords_scene = None
                        if coords_list:
                            coords_scene = torch.stack(coords_list).to(self.device)
                            coords_scene = coords_scene.unsqueeze(0)
                            if self.args.half:
                                coords_scene = coords_scene.half()
                    else:
                        imgs_scene = imgs[bi * V : (bi + 1) * V]
                        coords_scene = None
                        if coords is not None:
                            coords_scene = coords[bi : bi + 1]
                    tile_preds_list, offsets = self._tiled_forward(imgs_scene, coords_scene)
                    if batch_i == 0:
                        self._save_tiled_vis(imgs_scene, coords_scene, offsets, batch, bi)
                    per_view_preds = [[] for _ in range(V)]
                    for tile_preds, (x0, y0) in zip(tile_preds_list, offsets):
                        for vi in range(V):
                            p = tile_preds[vi]
                            if p["bboxes"].numel() == 0:
                                continue
                            merged = torch.cat(
                                [p["bboxes"], p["conf"].unsqueeze(1), p["cls"].unsqueeze(1)], dim=1
                            )
                            merged[:, [0, 2]] += x0
                            merged[:, [1, 3]] += y0
                            per_view_preds[vi].append(merged)

                    for vi in range(V):
                        merged = (
                            torch.cat(per_view_preds[vi], dim=0)
                            if per_view_preds[vi]
                            else torch.zeros((0, 6), device=imgs.device)
                        )
                        merged = self._nms_merged(merged, self.args.conf, self.args.iou, self.args.max_det)
                        if use_original:
                            ratio_pad = batch["ratio_pad"][bi * V + vi]
                            merged = self._map_to_letterbox(merged, ratio_pad)
                        preds.append(
                            {
                                "bboxes": merged[:, :4],
                                "conf": merged[:, 4],
                                "cls": merged[:, 5],
                                "extra": merged.new_zeros((merged.shape[0], 0)),
                            }
                        )

            with dt[2]:
                # Skip loss on tiled val; preds are post-processed per tile.
                if self.training:
                    self.loss += 0.0

            with dt[3]:
                # Already post-processed with NMS
                pass

            self.update_metrics(preds, batch)
            if batch_i == 0:
                try:
                    h, w = batch["img"].shape[2:]
                    ori0 = batch.get("ori_shape", [None])[0]
                    LOGGER.info(f"{colorstr('bright_blue')}val batch0 img size: {w}x{h}, ori_shape: {ori0}")
                except Exception:
                    pass
            if self.args.plots and batch_i < 10:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        return stats

    def print_results(self):
        """Print training/validation set metrics per class and per view."""
        # Print overall results first
        super().print_results()

        # Print per-view results after the "all" line
        # Only print summary metrics (not per-class) unless verbose mode
        if self.seen > 0 and len(self.view_metrics[0].stats) > 0:
            pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.view_metrics[0].keys)

            for i, (view_name, view_metric) in enumerate(zip(self.view_names, self.view_metrics)):
                LOGGER.info(
                    pf % (
                        view_name,
                        self.seen // self.num_views,  # Approximate images per view
                        view_metric.nt_per_class.sum(),
                        *view_metric.mean_results(),
                    )
                )

            # Print per-class results for each view only in verbose mode
            if self.args.verbose and (not self.args.verbose_test_only or not self.training):
                for i, (view_name, view_metric) in enumerate(zip(self.view_names, self.view_metrics)):
                    LOGGER.info(f"\n{view_name} per-class results:")
                    for j, c in enumerate(view_metric.ap_class_index):
                        LOGGER.info(
                            pf
                            % (
                                self.names[c],
                                view_metric.nt_per_image[c],
                                view_metric.nt_per_class[c],
                                *view_metric.class_result(j),
                            )
                        )
        if self.overlap_eval and self.seen > 0:
            pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.overlap_metrics[0].keys)
            for i in range(self.num_views):
                om = self.overlap_metrics[i]
                nm = self.nonoverlap_metrics[i]
                if len(om.stats):
                    LOGGER.info(
                        pf
                        % (
                            f"camera{i+1}_overlap",
                            self.seen // self.num_views,
                            om.nt_per_class.sum(),
                            *om.mean_results(),
                        )
                    )
                if len(nm.stats):
                    LOGGER.info(
                        pf
                        % (
                            f"camera{i+1}_nonoverlap",
                            self.seen // self.num_views,
                            nm.nt_per_class.sum(),
                            *nm.mean_results(),
                        )
                    )

    def plot_val_samples(self, batch, ni):
        """Plot validation samples and PE overlays."""
        if getattr(self, "_val_first_plot_done", False):
            return
        super().plot_val_samples(batch, ni)
        try:
            h, w = batch["img"].shape[2:]
            ori0 = batch.get("ori_shape", [None])[0]
            LOGGER.info(f"{colorstr('bright_blue')}val batch0 img size: {w}x{h}, ori_shape: {ori0}")
        except Exception:
            pass
        if "coords" not in batch:
            return
        images = batch["img"]  # (N*V, C, H, W)
        coords = batch["coords"]  # (N, V, H, W, 4)
        im_files = batch.get("im_file", [])

        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().float().numpy()

        bs, _, h, w = images.shape
        max_subplots = 16
        bs = min(bs, max_subplots)
        ns = int(np.ceil(bs**0.5))

        if np.max(images[0]) <= 1:
            images = (images * 255).astype(np.uint8)

        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
        coords_flat = coords.reshape(-1, h, w, 4)
        num_views = coords.shape[1] if coords.ndim == 5 else 2

        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))
            img = images[i].transpose(1, 2, 0)
            pe = coords_flat[i]
            pe_norm = ((pe + 1) / 2 * 255).astype(np.uint8)
            pe_vis = np.stack(
                [pe_norm[:, :, 0], pe_norm[:, :, 1], pe_norm[:, :, 2]], axis=2
            )
            if pe_vis.shape[:2] != img.shape[:2]:
                pe_vis = cv2.resize(pe_vis, (img.shape[1], img.shape[0]))
            blended = cv2.addWeighted(img, 0.6, pe_vis, 0.4, 0)
            mosaic[y : y + h, x : x + w, :] = blended
            if i < len(im_files):
                filename = Path(im_files[i]).name[:30]
                cv2.putText(mosaic, filename, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.rectangle(mosaic, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)
            view_idx = i % num_views
            cv2.putText(mosaic, f"View {view_idx}", (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        pe_fname = self.save_dir / f"val_batch{ni}_pe.jpg"
        cv2.imwrite(str(pe_fname), mosaic)
        LOGGER.info(f"{colorstr('bold')}Saved val PE visualization to {pe_fname.name}")
        # Disable further plotting after first batch to avoid overhead
        self._val_first_plot_done = True
        self.args.plots = False
        self._force_val_plot = False
