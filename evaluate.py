import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


import numpy as np

def compute_metrics(mask_pred, mask_true, num_classes):
    """
    mask_pred/mask_true: (B, H, W), 类型为long, 值域为类别标签
    返回: dict, 包含每个指标的统计
    """
    metrics = {}
    mask_pred = mask_pred.cpu().numpy().astype(np.int32)
    mask_true = mask_true.cpu().numpy().astype(np.int32)

    # Flatten
    mask_pred_flat = mask_pred.flatten()
    mask_true_flat = mask_true.flatten()
    valid = mask_true_flat > 0

    # IoU
    ious = []
    precisions = []
    recalls = []
    f1s = []
    for cls in range(1, num_classes):  
        pred_cls = (mask_pred_flat == cls)
        true_cls = (mask_true_flat == cls)
        inter = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        iou = inter / (union + 1e-7)
        ious.append(iou)
        # precision, recall, f1
        tp = inter
        fp = pred_cls.sum() - inter
        fn = true_cls.sum() - inter
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        precisions.append(precision)
        recalls.append(recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1s.append(f1)

    pa = (mask_pred_flat == mask_true_flat).sum() / len(mask_true_flat)

    metrics['iou'] = np.mean(ious)
    metrics['precision'] = np.mean(precisions)
    metrics['recall'] = np.mean(recalls)
    metrics['f1'] = np.mean(f1s)
    metrics['pa'] = pa
    return metrics


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    total_dice = 0
    metrics_total = {'iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'pa': 0}

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)
            if net.n_classes == 1:
                mask_pred_label = (F.sigmoid(mask_pred) > 0.5).long().squeeze(1)
            else:
                mask_pred_label = mask_pred.argmax(dim=1)

            # dice
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                total_dice += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_onehot = F.one_hot(mask_pred_label, net.n_classes).permute(0, 3, 1, 2).float()
                total_dice += multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)
            
            metrics = compute_metrics(mask_pred_label, mask_true, net.n_classes)
            for k in metrics_total:
                metrics_total[k] += metrics[k]

    net.train()
    num_batches = max(num_val_batches, 1)
    averaged_metrics = {k: v / num_batches for k, v in metrics_total.items()}
    return total_dice / num_batches, averaged_metrics
