import torch.nn as nn
import torch


class AbsWeightedLoss(nn.Module):
    def __init__(self, peak_weight=0.25):
        super(AbsWeightedLoss, self).__init__()
        assert peak_weight >= 0 and peak_weight <= 1
        self.peak_weight = peak_weight

    def forward(self, output, target):
        # Calculate L1 loss
        l1_loss = torch.abs(output - target)

        # Normalize target to have maximum absolute value of 1
        max_abs_target = torch.max(torch.abs(target))
        normalized_target = torch.abs(target) / max_abs_target

        # Calculate weights using alpha and the normalized absolute values of the target
        weights = (1 - self.peak_weight) + self.peak_weight * normalized_target

        # Apply weights to the L1 loss
        weighted_l1_loss = l1_loss * weights

        # Sum the weighted L1 loss
        loss = weighted_l1_loss.mean()

        return loss
