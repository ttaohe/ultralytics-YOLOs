# Ultralytics Multiview Model
from ultralytics.nn.tasks import DetectionModel
import torch

class YOLOMultiview(DetectionModel):
    """
    YOLOMultiview model that extends DetectionModel to handle multiview inputs.
    """
    def __init__(self, cfg="yolo12-multiview.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

    def forward(self, x, *args, **kwargs):
        """
        x can be (B, V, C, H, W) or (B*V, C, H, W)
        """
        if isinstance(x, dict):
            return super().forward(x, *args, **kwargs)

        # Ensure input dtype matches model weights (avoids FP16/FP32 mismatch in val)
        try:
            dtype = next(self.parameters()).dtype
            if isinstance(x, torch.Tensor) and x.dtype != dtype:
                x = x.to(dtype)
        except Exception:
            pass

        if x.dim() == 5:
            B, V, C, H, W = x.shape
            # Reshape to (B*V, C, H, W) for backbone
            x = x.view(B*V, C, H, W)
        
        return super().forward(x, *args, **kwargs)
