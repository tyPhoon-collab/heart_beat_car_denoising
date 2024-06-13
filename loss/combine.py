import torch
import torch.nn as nn
import numpy as np

from loss.wavelet_transform import wavelet_transform


# カスタム損失関数の定義
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=np.arange(1, 37)):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.widths = torch.tensor(scales, dtype=torch.float32)

    def forward(self, outputs, targets):
        # 波形のL1Lossを計算
        l1_loss_waveform = self.l1_loss(outputs, targets)

        # Wavelet変換を実行
        outputs_cwt = wavelet_transform(outputs.detach(), self.widths)
        targets_cwt = wavelet_transform(targets.detach(), self.widths)

        # 周波数成分のL1Lossを計算
        l1_loss_cwt = self.l1_loss(outputs_cwt[0], targets_cwt[0])

        # 総合損失を計算
        total_loss = self.alpha * l1_loss_waveform + (1 - self.alpha) * l1_loss_cwt
        return total_loss
