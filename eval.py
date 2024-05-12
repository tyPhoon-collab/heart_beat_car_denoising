import torch
from torch.utils.data import DataLoader
from dataset.dataset import NoisyHeartbeatDataset
from dataset.randomizer import NumpyRandomShuffleRandomizer
from dataset.sampling_rate_converter import ScipySamplingRateConverter
from models.wave_u_net import WaveUNet
from utils.visualize import plot_three_signals

# 推論用のデータセット設定
test_dataset = NoisyHeartbeatDataset(
    clean_file_path="data/Stop.mat",
    noisy_file_path="data/100km.mat",
    sampling_rate_converter=ScipySamplingRateConverter(
        input_rate=32000, output_rate=1024
    ),
    randomizer=NumpyRandomShuffleRandomizer(),
    train=False,
)

# DataLoaderの設定。推論ではshuffleは不要です。
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# モデルのインスタンスを作成し、訓練済みの重みをロードします。
model = WaveUNet()
model.load_state_dict(torch.load("output/checkpoint/model_weights_epoch_5.pth"))
model.eval()  # 評価モードに設定

# 推論の実行
with torch.no_grad():  # 勾配計算を無効化
    for noisy, clean in test_dataloader:
        outputs = model(noisy)

        plot_three_signals(
            noisy[0][0],
            clean[0][0],
            outputs[0][0],
            "Noisy",
            "Clean",
            "Output",
            filename="eval.png",
        )
        break  # ここではデモンストレーションのため、最初のバッチのみ処理
