import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        # preds: (B, C, H, W) raw logits
        # targets: (B, H, W) class indices
        preds_softmax = F.softmax(preds, dim=1)
        iou_per_class = []
        for cls in range(self.num_classes):
            pred_cls = preds_softmax[:, cls, :, :]   # (B,H,W)
            target_cls = (targets == cls).float()    # (B,H,W)

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_per_class.append(iou)

        mean_iou = torch.mean(torch.stack(iou_per_class))
        iou_loss = 1 - mean_iou
        return iou_loss
