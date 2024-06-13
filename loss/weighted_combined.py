# カスタム損失関数の定義
import numpy as np
import torch
from loss.wavelet_transform import wavelet_transform
from loss.weighted import WeightedLoss
import torch.nn as nn


class WeightedCombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, peak_weight=None, scales=np.arange(1, 37)):
        super(WeightedCombinedLoss, self).__init__()
        weighted_loss_args = {}

        if peak_weight is not None:
            weighted_loss_args["peak_weight"] = peak_weight

        self.wl1_loss = WeightedLoss(*weighted_loss_args)
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.widths = torch.tensor(scales, dtype=torch.float32)

    def forward(self, outputs, targets):
        # 波形のL1Lossを計算
        wl1_loss_waveform = self.wl1_loss(outputs, targets)

        # Wavelet変換を実行
        outputs_cwt = wavelet_transform(outputs.detach(), self.widths)
        targets_cwt = wavelet_transform(targets.detach(), self.widths)

        # 周波数成分のL1Lossを計算
        l1_loss_cwt = self.l1_loss(outputs_cwt[0], targets_cwt[0])

        # 総合損失を計算
        total_loss = self.alpha * wl1_loss_waveform + (1 - self.alpha) * l1_loss_cwt
        return total_loss
