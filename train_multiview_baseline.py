import argparse
import os
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import YAML


warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")
print(f"[LAUNCH] train_multiview_baseline.py pid={os.getpid()}")


def train():
    """
    标准 Ultralytics 训练脚本（非 video-mode）。

    示例：
      python train_multiview_baseline.py \
        --model ultralytics/cfg/models/v10/yolov10l.yaml \
        --data ultralytics/cfg/datasets/MDMT_camera1.yaml \
        --epochs 100 --imgsz 640 --batch 16 --device 0
    """
    parser = argparse.ArgumentParser(description="Train a YOLO model (standard Ultralytics example style)")
    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/v10/yolov10l.yaml",
        help="模型：*.yaml 或 *.pt（例如 yolov10n.pt 或自定义yaml）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ultralytics/cfg/datasets/MOMT_camera1.yaml",
        help="数据集：data.yaml 路径（普通检测数据集，不要 video_mode）",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0", help="例如 '0'/'0,1'/'cpu'")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=str, default="runs/train_multiview")
    parser.add_argument("--name", type=str, default="yolov10-multiview-baseline")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从断点恢复：last.pt 或 best.pt 的路径。设置后将调用 model.train(resume=True)。",
    )
    opt = parser.parse_args()

    # --- Sanity checks (avoid "wrong run/log confusion") ---
    data_path = Path(opt.data).expanduser()
    # Resolve relative path against current working directory (Ultralytics expects this behavior)
    data_path_abs = (Path.cwd() / data_path).resolve() if not data_path.is_absolute() else data_path.resolve()
    print(f"[ARGS] cwd={Path.cwd()}")
    print(f"[ARGS] model={opt.model}")
    print(f"[ARGS] data={opt.data} -> {data_path_abs}")
    if not data_path_abs.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path_abs}")
    data_dict = YAML.load(str(data_path_abs))
    print(f"[DATA] nc={data_dict.get('nc')} names={data_dict.get('names')}")

    # Resume: 不传入可能冲突的超参，完全交给 checkpoint 里的 train_args
    if opt.resume:
        resume_path = Path(opt.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"[RESUME] Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
        return

    # Fresh training
    model = YOLO(opt.model)
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
    )


if __name__ == "__main__":
    train()

