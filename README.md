# Heart beat car denoising

## 実行環境

Docker環境で行うのが良いですが、念の為、使用しているGPUとコンテナの環境を以下に記載します

```bash
$ python --version
Python 3.10.11

$ nvidia-smi
Tue Sep 17 17:25:40 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   52C    P8              34W / 420W |  14002MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       928      G   /usr/lib/xorg/Xorg                            4MiB |
|    0   N/A  N/A     42211      C   /opt/conda/bin/python                     13370MiB |
|    0   N/A  N/A    100278      C   /opt/conda/bin/python                       612MiB |
+---------------------------------------------------------------------------------------+

```

`requirements.txt`も用意してあります

## セットアップ

### GitHubリポジトリから

[GitHub リポジトリ](https://github.com/tyPhoon-collab/heart_beat_car_denoising)

1. clone
2. .containerのDockerfileを元にコンテナを起動
3. 以下のファイルを追加する

- .env（オプション）
  - ロガー用の変数を置く
    - [サンプル](#envファイルのサンプル)
- dataフォルダ
  - 杉浦先生が用意したMAT形式のデータを置く
    - data/
      - 100km.mat
      - Stop.mat
  - gitで管理していないため、必要に応じてファイルを追加する

## 重要なファイル

- `cli.py`
  - CLIの実装
- `models/*`
  - モデルの実装
  - 現在メインで運用しているのは`wave_u_net_enhance_transformer.py`
<!-- - `eval.py`
  - 杉浦先生から指示があった評価ケース(30種類)を実行する -->
- `train.py`
  - シンプルな訓練用ファイル
- `entire_eval.py`
  - ノイズの信号を全でデノイジングする
- `.container/`
  - Dockerの定義

## 実行コマンド例

### シンプルな例

まずはこれが動くか確認してください

#### 訓練

```bash
python train.py
```

output/checkpoint/<日付>/model_weights_best.pthなどが生成されます

#### 評価

```bash
python entire_eval.py
```

output/html/entire_eval.htmlなどが生成されます。ブラウザで開くとインタラクティブに結果を見ることができます

### CLIを使用する場合

もし柔軟にプロパティを変更したい場合は、ある程度以下のCLIで対応可能です

```bash
# オプションの確認
python cli.py train --help

python cli.py train \
  --model WaveUNetEnhanceTransformer \
  --loss-fn WeightedLoss \
  --batch-size 64 \
  --learning-rate 0.000025 \
  --epoch-size 100 \
  --with-progressive-gain \
  --progressive-epoch-to 50 \
  --gain 1 \
  --pretrained-weights-path output/checkpoint/path_to_weights.pth
```

```bash
python cli.py eval --help

python cli.py eval \
  --model WaveUNetEnhanceTransformer \
  --loss-fn WeightedLoss \
  --batch-size 64 \
  --gain 1 \
  --weights-path output/checkpoint/path_to_weights.pth
```

### bashファイルを使用する場合

CLIをラップしたもの。少しだけコマンドが短くなる程度

```bash
# オプションの確認
bash train_and_eval.sh
# Usage: train_and_eval.sh <ID> <MODEL> <LOSS_FN> <BATCH_SIZE> [--gain VALUE] [--stride-samples VALUE] [--split-samples VALUE] [<ANOTHER_TRAINING_OPTIONS>]

# 以下、使用例
# IDは任意のもの、重みの保存時のフォルダ名になる
bash train_and_eval.sh run_sample WaveUNetEnhanceTransformer WeightedLoss 64 --gain 1 \
  --learning-rate 0.000025 --epoch-size 100 --progressive-epoch-to 50 --with-progressive-gain \
  --pretrained-weights-path output/checkpoint/path_to_weights.pth
```

## .envファイルのサンプル

このディレクトリの直下に.envファイルを作成して、以下の環境変数を定義する。CLIなどで指定しないグローバルな変数を管理している

Discordやneptuneなどの統合の際に使用中。基本的にすべて0で構いません

`STDOUT_LOGGING=1`はあると便利です

```bash
ONLY_FIRST_BATCH=0
STDOUT_LOGGING=1
REMOTE_LOGGING=0
DISCORD_LOGGING=0
NEPTUNE_LOGGING=0
NEPTUNE_SAVE_MODEL_STATE=0
# DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx.../xxx..."
# NEPTUNE_PROJECT_NAME="xxx/yyy"
# NEPTUNE_API_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXX...=="
```
