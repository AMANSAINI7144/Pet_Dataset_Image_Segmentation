import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Dice Loss
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        inputs = torch.softmax(inputs, dim=1)

        intersection = torch.sum(inputs * targets_onehot)
        union = torch.sum(inputs) + torch.sum(targets_onehot)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


# ---------------------------
# Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------
# Combined Loss (CrossEntropy + Dice)
# ---------------------------
class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.w_ce * ce_loss + self.w_dice * dice_loss
