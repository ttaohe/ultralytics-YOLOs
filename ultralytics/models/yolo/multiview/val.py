# Ultralytics Multiview Validator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils import LOGGER, colorstr
import numpy as np


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

    def init_metrics(self, model):
        """Initialize metrics for all views."""
        super().init_metrics(model)

        # Initialize each view's metrics
        for i, metrics in enumerate(self.view_metrics):
            metrics.names = model.names
            metrics.nc = len(model.names)

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
            view_metric.process(save_dir=view_dir, plot=self.args.plots, on_plot=self.on_plot)
            view_metric.clear_stats()

            # Store view-specific results
            view_results = view_metric.results_dict
            for key, value in view_results.items():
                view_stats[f"{key}_view{i+1}"] = value

        # Combine overall and view-specific stats
        stats = self.metrics.results_dict
        stats.update(view_stats)

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
