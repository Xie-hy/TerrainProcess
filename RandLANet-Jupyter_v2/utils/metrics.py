import numpy as np
import torch

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious


def compute_metrics(predicts, labels, num_classes):
    """
    计算 Accuracy 和 mIoU
    :param predicts: 模型输出 [B, C, H, W]（未归一化的 logits）
    :param labels: 真实标签 [B, H, W]（整数型类别索引）
    :param num_classes: 类别数（含背景）
    :return: accuracy, mIoU
    """
    # 将 predicts 转换为预测标签（0~C-1的整数）
    pred_labels = torch.argmax(predicts, dim=1)  # [B, H, W]

    # 计算 Accuracy
    correct = (pred_labels == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total

    # 计算 mIoU
    iou_per_class = []
    for c in range(num_classes):
        # 统计当前类别的 TP, FP, FN
        true_positive = ((pred_labels == c) & (labels == c)).sum().item()
        false_positive = ((pred_labels == c) & (labels != c)).sum().item()
        false_negative = ((pred_labels != c) & (labels == c)).sum().item()

        # 避免分母为0
        union = true_positive + false_positive + false_negative
        if union == 0:
            iou_c = 0.0  # 若该类别不存在，IoU=0
        else:
            iou_c = true_positive / union
        iou_per_class.append(iou_c)

    # 计算各类别IoU的平均值（mIoU）
    mIoU = np.mean(iou_per_class)

    return accuracy, mIoU