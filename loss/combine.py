import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ptwt
import pywt


# GPU上でWavelet変換を行う関数
def wavelet_transform(signal: torch.Tensor, widths, wavelet_name="morl"):
    wavelet = pywt.ContinuousWavelet(wavelet_name)  # type: ignore
    cwt_coeffs = ptwt.cwt(signal, widths, wavelet)
    return cwt_coeffs


# カスタム損失関数の定義
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, widths=np.arange(1, 37)):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.widths = torch.tensor(widths, dtype=torch.float32)

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


if __name__ == "__main__":
    # モデルの定義（例として単純な線形モデル）
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(5120, 5120)  # 入力5120次元、出力5120次元

        def forward(self, x):
            return self.fc(x)

    # モデル、損失関数、最適化手法の初期化
    model = SimpleModel()
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # ダミーデータの作成
    inputs = torch.randn(1, 1, 5120)
    targets = torch.randn(1, 1, 5120)

    # 学習ループ
    for epoch in range(100):  # 例として100エポック
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)  # モデルの出力
        loss = criterion(outputs, targets)  # 損失の計算
        loss.backward()  # 勾配の計算
        optimizer.step()  # パラメータの更新

        if (epoch + 1) % 10 == 0:  # 10エポックごとに損失を出力
            print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
