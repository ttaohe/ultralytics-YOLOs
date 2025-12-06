import torch
import numpy as np
import yaml
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou, compute_ap
from ultralytics.utils.ops import xywhn2xyxy
from ultralytics.utils.nms import non_max_suppression
from ultralytics.data.build import build_dataloader
from ultralytics.data.video_dataset import VisDroneVideoDataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import colorstr

def compute_ap_per_size(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, gt_areas, iou_thres=0.5):
    """
    Compute AP for a specific size category.
    
    Args:
        pred_boxes: [N, 4] (x1, y1, x2, y2)
        pred_scores: [N]
        pred_cls: [N]
        gt_boxes: [M, 4] (x1, y1, x2, y2)
        gt_cls: [M]
        gt_areas: [M] - area of each GT box
        iou_thres: IoU threshold for matching
        
    Returns:
        ap: Average Precision
        precision: Precision curve
        recall: Recall curve
    """
    if len(gt_boxes) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    
    if len(pred_boxes) == 0:
        return 0.0, np.array([0.0]), np.array([0.0])
    
    # Sort predictions by confidence (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_cls = pred_cls[sorted_indices]
    
    # Compute IoU matrix
    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes)).numpy()  # [N, M]
    
    # Match predictions to ground truth
    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    for i in range(len(pred_boxes)):
        # Find best matching GT (highest IoU)
        iou = ious[i]
        best_iou_idx = np.argmax(iou)
        best_iou = iou[best_iou_idx]
        
        # Check if match is valid
        if best_iou >= iou_thres and not gt_matched[best_iou_idx] and pred_cls[i] == gt_cls[best_iou_idx]:
            tp[i] = True
            gt_matched[best_iou_idx] = True
        else:
            fp[i] = True
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute Precision and Recall
    n_gt = len(gt_boxes)
    recall = tp_cumsum / (n_gt + 1e-16)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
    
    # Compute AP using COCO method (101-point interpolation)
    ap, mpre, mrec = compute_ap(recall, precision)
    
    return ap, precision, recall

def analyze_visdrone_performance():
    # 1. 配置
    base_dir = Path('runs/train-video')
    runs = sorted(list(base_dir.glob('*video*')))
    model_path = None
    if runs:
        latest_run = runs[-1]
        best_path = latest_run / 'weights/best.pt'
        last_path = latest_run / 'weights/last.pt'
        if best_path.exists():
            model_path = str(best_path)
        elif last_path.exists():
            model_path = str(last_path)
            
    if not model_path:
        model_path = 'runs/train-video/yolo12-sam2-video5/weights/last.pt'

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.model.cuda()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 2. 构建数据集
    data_cfg = 'ultralytics/cfg/datasets/VisDrone-vid.yaml'
    with open(data_cfg) as f:
        data = yaml.safe_load(f)
    
    dataset_root = data.get('path')
    val_path = data.get('val')
    if not os.path.isabs(val_path):
        val_path = os.path.join(dataset_root, val_path)

    print(f"Validating on: {val_path}")
    
    args = IterableSimpleNamespace(
        imgsz=800,
        rect=False,
        cache=False,
        single_cls=False,
        task='detect',
        classes=None,
        fraction=1.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
    )
    
    dataset = VisDroneVideoDataset(
        img_path=val_path,
        imgsz=800,
        batch_size=1,
        augment=False,
        hyp=args,
        rect=False,
        stride=32,
        prefix=colorstr(f"val: "),
        data=data
    )
    
    batch_size = 4
    workers = 4
    dataloader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False, rank=-1)
    
    # 3. 收集所有预测和 GT
    all_predictions = {'boxes': [], 'scores': [], 'cls': []}
    all_groundtruth = {'boxes': [], 'cls': [], 'areas': []}

    print("Starting inference and collecting predictions...")
    model.model.eval()
    
    for batch in tqdm(dataloader):

        
        imgs = batch['img'].to(device).float() / 255.0
        hist_imgs = batch['history_img']
        if isinstance(hist_imgs, (list, tuple)):
             hist_imgs = torch.stack(hist_imgs)
        hist_imgs = hist_imgs.to(device).float() / 255.0
        
        inp = torch.stack([hist_imgs, imgs], dim=1).view(-1, *imgs.shape[1:])
        
        with torch.no_grad():
            preds = model.model(inp)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            curr_preds = preds[1::2]
            post_preds = non_max_suppression(curr_preds, conf_thres=0.001, iou_thres=0.6, max_det=300)
            
        # 处理每个样本
        for i, det in enumerate(post_preds):
            idx_mask = batch['batch_idx'] == i
            if not idx_mask.any():
                gts = torch.zeros((0, 5))
            else:
                cls = batch['cls'][idx_mask].view(-1, 1)
                bboxes = batch['bboxes'][idx_mask]
                h, w = imgs.shape[2:]
                bboxes = xywhn2xyxy(bboxes, w=w, h=h)
                gts = torch.cat([cls, bboxes], dim=1).to(device)
            
            # 处理预测
            if len(det) > 0:
                det_boxes = det[:, :4].cpu().numpy()  # [N, 4]
                det_scores = det[:, 4].cpu().numpy()   # [N]
                det_cls = det[:, 5].cpu().numpy()      # [N]
                
                # 处理 GT
                if len(gts) > 0:
                    gt_cls = gts[:, 0].cpu().numpy()      # [M]
                    gt_boxes = gts[:, 1:].cpu().numpy()   # [M, 4]
                    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                    
                    # 收集所有 GT
                    all_groundtruth['boxes'].append(gt_boxes)
                    all_groundtruth['cls'].append(gt_cls)
                    all_groundtruth['areas'].append(gt_areas)
                
                # 收集所有预测
                all_predictions['boxes'].append(det_boxes)
                all_predictions['scores'].append(det_scores)
                all_predictions['cls'].append(det_cls)
    
    # 4. 合并所有数据
    print("\nComputing mAP50 by size category...")
    
    # 合并所有预测
    all_pred_boxes = np.concatenate(all_predictions['boxes'], axis=0) if all_predictions['boxes'] else np.zeros((0, 4))
    all_pred_scores = np.concatenate(all_predictions['scores'], axis=0) if all_predictions['scores'] else np.zeros(0)
    all_pred_cls = np.concatenate(all_predictions['cls'], axis=0) if all_predictions['cls'] else np.zeros(0, dtype=int)
    
    # 合并所有 GT
    all_gt_boxes = np.concatenate(all_groundtruth['boxes'], axis=0) if all_groundtruth['boxes'] else np.zeros((0, 4))
    all_gt_cls = np.concatenate(all_groundtruth['cls'], axis=0) if all_groundtruth['cls'] else np.zeros(0, dtype=int)
    all_gt_areas = np.concatenate(all_groundtruth['areas'], axis=0) if all_groundtruth['areas'] else np.zeros(0)
    
    # 5. 按尺寸分组计算 mAP50
    results = {}
    
    for size_cat in ['small', 'medium', 'large', 'all']:
        if size_cat == 'all':
            gt_mask = np.ones(len(all_gt_boxes), dtype=bool)
        else:
            if size_cat == 'small':
                gt_mask = all_gt_areas < 32**2
            elif size_cat == 'medium':
                gt_mask = (all_gt_areas >= 32**2) & (all_gt_areas <= 96**2)
            else:  # large
                gt_mask = all_gt_areas > 96**2
        
        if not gt_mask.any():
            results[size_cat] = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0, 'gt_count': 0}
            continue
        
        gt_boxes = all_gt_boxes[gt_mask]
        gt_cls = all_gt_cls[gt_mask]
        gt_areas = all_gt_areas[gt_mask]
        
        # 使用所有预测，但只匹配到该尺寸的 GT
        if len(all_pred_boxes) == 0:
            results[size_cat] = {'mAP50': 0.0, 'precision': 0.0, 'recall': 0.0, 'gt_count': len(gt_boxes)}
            continue
        
        # 计算 AP
        ap50, precision, recall = compute_ap_per_size(
            all_pred_boxes, all_pred_scores, all_pred_cls, gt_boxes, gt_cls, gt_areas, iou_thres=0.5
        )
        
        # 计算平均 Precision 和 Recall（在最佳 F1 点）
        if len(precision) > 0 and len(recall) > 0:
            f1 = 2 * precision * recall / (precision + recall + 1e-16)
            best_idx = np.argmax(f1)
            avg_precision = precision[best_idx]
            avg_recall = recall[best_idx]
        else:
            avg_precision = 0.0
            avg_recall = 0.0
        
        results[size_cat] = {
            'mAP50': ap50,
            'precision': avg_precision,
            'recall': avg_recall,
            'gt_count': len(gt_boxes)
        }
    
    # 5. 打印报告
    print("\n" + "="*70)
    print(f"{'Size':<10} | {'mAP50':<10} | {'Precision':<10} | {'Recall':<10} | {'GT Count':<10}")
    print("-" * 70)
    
    for size_cat in ['small', 'medium', 'large', 'all']:
        r = results[size_cat]
        print(f"{size_cat.upper():<10} | {r['mAP50']:.4f}     | {r['precision']:.2%}     | {r['recall']:.2%}     | {r['gt_count']:<10}")
    print("="*70)
    print("\n指标说明:")
    print("1. mAP50: 标准 COCO 评估方法，IoU 阈值 0.5 下的平均精度（AP）")
    print("2. Precision: 在最佳 F1 点时的查准率（预测正确的比例）")
    print("3. Recall: 在最佳 F1 点时的查全率（找到的真实目标比例）")
    print("4. 此评估方法更接近训练日志中的 mAP50 计算方式")

if __name__ == '__main__':
    analyze_visdrone_performance()
