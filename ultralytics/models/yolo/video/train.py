from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.video.model import YOLOVideo
from ultralytics.utils import RANK

class VideoTrainer(DetectionTrainer):
    """
    Trainer for YOLO Video models with memory attention.
    """
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLOVideo model."""
        model = YOLOVideo(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

