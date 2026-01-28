# Ultralytics Multiview Validator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import LOGGER, colorstr
import numpy as np
import torch


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
