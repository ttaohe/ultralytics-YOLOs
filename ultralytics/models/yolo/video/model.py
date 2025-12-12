# Ultralytics Video Model
from ultralytics.nn.modules import YOLOMemoryAttention, SparseMemoryAttention
from ultralytics.nn.tasks import DetectionModel
import torch

class YOLOVideo(DetectionModel):
    """
    YOLOVideo model that extends DetectionModel to handle video sequences
    and memory attention.
    """
    def __init__(self, cfg="yolo12-video.yaml", ch=3, nc=None, verbose=True):
        self.time_steps = 1
        super().__init__(cfg, ch, nc, verbose)
        self.time_steps = 1 # Default

    def forward(self, x, *args, **kwargs):
        """
        x can be (B, T, C, H, W) or (B*T, C, H, W)
        """
        if isinstance(x, dict):
            return super().forward(x, *args, **kwargs)

        if x.dim() == 5:
            B, T, C, H, W = x.shape
            self.time_steps = T
            # Reshape to (B*T, C, H, W) for backbone
            x = x.view(B*T, C, H, W)
        else:
            # Assume x is (N, C, H, W). 
            # If training, we rely on external setting of time_steps
            # or we assume N = B*T
            pass
            
        # Update time_steps in all YOLOMemoryAttention layers
        for m in self.model.modules():
            if isinstance(m, YOLOMemoryAttention):
                m.time_steps = self.time_steps
        
        return super().forward(x, *args, **kwargs)

    def reset_memory(self):
        for m in self.model.modules():
            if isinstance(m, YOLOMemoryAttention):
                m.reset_memory()

    def set_memory(self, memory):
        """
        [State Injection]
        Inject external memory state into the attention layer.
        """
        for m in self.model.modules():
            if isinstance(m, YOLOMemoryAttention):
                m.set_memory(memory)
                break # Assuming model has one memory attention layer for now

    def get_memory(self):
        """
        [State Extraction]
        Retrieve external memory state from the attention layer.
        """
        for m in self.model.modules():
            if isinstance(m, YOLOMemoryAttention):
                return m.get_memory()
        return None

