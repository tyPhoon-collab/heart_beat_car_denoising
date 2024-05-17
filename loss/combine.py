import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.signal import cwt, morlet


# Wavelet変換を行う関数
def wavelet_transform(signal, widths):
    wavelet = morlet
    cwt_matr = cwt(signal, wavelet, widths)
    return cwt_matr


# カスタム損失関数の定義
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, widths=np.arange(1, 37)):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.widths = widths

    def forward(self, outputs, targets):
        # 波形のL1Lossを計算
        l1_loss_waveform = self.l1_loss(outputs, targets)

        # Wavelet変換を実行
        # 勾配計算を行わないようにするため、detach()メソッドを使用
        outputs_np = outputs.detach().cpu().numpy().squeeze()
        targets_np = targets.detach().cpu().numpy().squeeze()

        cwt_outputs = wavelet_transform(outputs_np, self.widths)
        cwt_targets = wavelet_transform(targets_np, self.widths)

        # CWTの結果をPyTorchテンソルに変換
        cwt_outputs_tensor = torch.tensor(
            cwt_outputs,
            dtype=torch.float32,
            device=outputs.device,
        )
        cwt_targets_tensor = torch.tensor(
            cwt_targets,
            dtype=torch.float32,
            device=outputs.device,
        )

        # 周波数成分のL1Lossを計算
        l1_loss_cwt = self.l1_loss(cwt_outputs_tensor, cwt_targets_tensor)

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
    inputs = torch.randn(1, 1, 5120)  # バッチサイズ1、チャネル1、入力次元5120
    targets = torch.randn(1, 1, 5120)  # バッチサイズ1、チャネル1、ターゲット次元5120

    # 学習ループ
    for epoch in range(100):  # 例として100エポック
        optimizer.zero_grad()  # 勾配の初期化
        outputs = model(inputs)  # モデルの出力
        loss = criterion(outputs, targets)  # 損失の計算
        loss.backward()  # 勾配の計算
        optimizer.step()  # パラメータの更新

        if (epoch + 1) % 10 == 0:  # 10エポックごとに損失を出力
            print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")
