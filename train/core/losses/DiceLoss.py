import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, reduction='mean'):
        """
        num_classes: total number of classes
        smooth: small constant to avoid division by zero
        reduction: 'mean', 'sum', or 'none'
        """
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits of shape (N, C, H, W)
        targets: ground truth labels of shape (N, H, W), with values in [0, C-1]
        """
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)  # shape: (N, C, H, W)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # shape: (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()       # shape: (N, C, H, W)

        # Flatten inputs and targets
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)       # shape: (N, C, H*W)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum(dim=2)              # shape: (N, C)
        total = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)            # shape: (N, C)

        # Compute Dice score
        dice_score = (2.0 * intersection + self.smooth) / (total + self.smooth)  # shape: (N, C)
        loss = 1.0 - dice_score  # shape: (N, C)

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: (N, C)
