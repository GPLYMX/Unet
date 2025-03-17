import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DynamicWeightedLoss(nn.Module):
    def __init__(self, weight, alpha=1.0, beta=0.7, gamma=2.0, upper=1000, lower=200, weight_upper=2.5, weight_lower=20.0):
        super(DynamicWeightedLoss, self).__init__()
        self.base_class_weights = weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.upper = upper
        self.lower = lower
        self.weight_upper = weight_upper
        self.weight_lower = weight_lower

    def forward(self, outputs, targets):
        batch_size, num_classes, height, width = outputs.shape
        
        # Calculate base cross entropy loss
        ce_loss = F.cross_entropy(outputs, targets, weight=self.base_class_weights, reduction='none')
        
        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Calculate dynamic weights based on each impurity region's area
        dynamic_weights = torch.ones_like(ce_loss)
        for b in range(batch_size):
            impurity_mask = (targets[b] == 2).cpu().numpy()  # Assuming the impurity class index is 2
            labeled_array, num_features = label(impurity_mask)
            for i in range(1, num_features + 1):
                region_mask = labeled_array == i
                region_area = region_mask.sum()
                if region_area > self.upper:
                    region_weight = self.weight_upper
                elif region_area < self.lower:
                    region_weight = self.weight_lower
                else:
                    # Linearly interpolate weight between upper and lower
                    region_weight = self.weight_upper + (self.weight_lower - self.weight_upper) * (self.upper - region_area) / (self.upper - self.lower)
                dynamic_weights[b][region_mask] *= region_weight
        
        # Apply dynamic weights to the cross entropy loss
        weighted_ce_loss = ce_loss * dynamic_weights
        weighted_ce_loss = weighted_ce_loss.mean()
        
        # Dice Loss
        smooth = 1.0
        outputs = F.softmax(outputs, dim=1)
        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + smooth)
        dice_loss = 1 - dice.mean()

        # Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # Combined Loss
        combined_loss = self.beta * weighted_ce_loss + (1 - self.beta) * dice_loss + focal_loss
        return combined_loss

## 示例权重：背景、茶叶、杂质
#base_class_weights = torch.tensor([0.5, 0.5, 2.0])  # 根据类别频率和重要性调整权重
#
## 创建 DynamicWeightedLoss 实例
#loss_fn = DynamicWeightedLoss(base_class_weights=base_class_weights, alpha=1.0, beta=0.5, gamma=2.0, upper=500, lower=50, weight_upper=0.5, weight_lower=2.0)
#
## 示例数据
#outputs = torch.randn(4, 3, 256, 256)  # (batch_size, num_classes, height, width)
#targets = torch.randint(0, 3, (4, 256, 256))  # (batch_size, height, width)
#
## 计算损失
#loss = loss_fn(outputs, targets)
#print(loss)