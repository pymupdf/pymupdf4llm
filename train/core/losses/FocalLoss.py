import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: focusing parameter (commonly 2.0)
        alpha: None or tensor/list of shape (num_classes,) or scalar (class balancing)
        reduction: 'none', 'mean', 'sum'
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            # store alpha as tensor for device transfer later
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits with shape (N, C) or (N, C, ...)
        targets: long tensor with shape (N, ...) (class indices)
        Note: This implementation handles the common (N, C) and (N,) case.
              For extra spatial dims (e.g., segmentation), flattening is needed.
        """
        # compute log probabilities
        logpt = F.log_softmax(inputs, dim=1)  # (N, C, ...)
        pt = torch.exp(logpt)                  # (N, C, ...)

        # flatten to (N*..., C) for gather if necessary
        if inputs.dim() > 2:
            # move class dim to last, then flatten
            perm = list(range(0, inputs.dim()))
            perm = perm[0:1] + perm[2:] + [1]  # bring class dim to end
            logpt_flat = logpt.permute(*perm).contiguous().view(-1, inputs.size(1))
            pt_flat = pt.permute(*perm).contiguous().view(-1, inputs.size(1))
            targets_flat = targets.view(-1, 1)
        else:
            logpt_flat = logpt
            pt_flat = pt
            targets_flat = targets.view(-1, 1)

        # gather log-prob and prob corresponding to target class
        logpt_target = logpt_flat.gather(1, targets_flat).squeeze(1)  # (N*...)
        pt_target = pt_flat.gather(1, targets_flat).squeeze(1)        # (N*...)

        # handle alpha (class weights)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                alpha = self.alpha.to(inputs.device)
            else:
                alpha = self.alpha
            if alpha.dim() == 0:
                at = alpha
            else:
                at = alpha.gather(0, targets_flat.squeeze(1))
        else:
            at = 1.0

        # focal loss formula
        loss = - at * (1 - pt_target) ** self.gamma * logpt_target

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # reshape back to original target shape if needed
            if inputs.dim() > 2:
                return loss.view(*targets.shape)
            return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        gamma: focusing parameter
        alpha: weighting factor for positive class (scalar)
        reduction: 'none', 'mean', 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits, shape (N, ...) or (N,1)
        targets: binary labels (0 or 1), same shape as inputs (after squeeze if needed)
        """
        # compute binary cross entropy with logits (stable)
        bce = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        # p_t is the model's estimated probability for the true class
        p_t = torch.exp(-bce)
        # focal loss term
        loss = self.alpha * (1 - p_t) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def test_with_ce():
    logits = torch.tensor([[1.0, 2.0, 0.1], [0.5, 0.2, 3.0]], requires_grad=True)  # shape: [batch_size, num_classes]
    targets = torch.tensor([1, 2])  # 정답 클래스 인덱스

    # 손실 함수 인스턴스 생성
    ce_loss_fn = nn.CrossEntropyLoss()
    focal_loss_fn = FocalLoss(gamma=0.1)

    # 손실 값 계산
    ce_loss_value = ce_loss_fn(logits, targets)
    focal_loss_value = focal_loss_fn(logits, targets)

    # 결과 출력
    print(f"Cross Entropy Loss: {ce_loss_value.item():.6f}")
    print(f"Focal Loss: {focal_loss_value.item():.6f}")
    print(f"Loss Difference: {focal_loss_value.item() - ce_loss_value.item():.6f}")


if __name__ == "__main__":
    test_with_ce()
