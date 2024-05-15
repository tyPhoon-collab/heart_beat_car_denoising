# Heart beat car denoising

## セットアップ

1. cloneする
2. 計算機サーバーで動かす。Larkを見てセットアップする。.containerは作成済み
3. 以下のファイルを追加する

- .env
  - ロガー用の変数を置く
    - [サンプル](#envファイルのサンプル)
    - LOGGING
    - ONLY_FIRST_BATCH
    - DISCORD_WEBHOOK_URL
    - NEPTUNE_PROJECT_NAME
    - NEPTUNE_API_TOKEN
  - なくても動く
- dataフォルダ
  - 杉浦先生が用意したMAT形式のデータを置く
    - data/
      - 100km.mat
      - Stop.mat
  - gitで管理していないため、必要に応じてファイルを追加する
  - Larkの心拍推定からダウンロードする

## ファイル名

[このリポジトリ](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/README.md)に則ったファイル名にしている。inferenceではなく、evalとしている。

## 実行コマンド例

```bash
python cli.py train --model Conv1DAutoencoder --loss-fn SmoothL1Loss \
--checkpoint-dir "output/checkpoint/AE_smooth_l1_loss"
```

```bash
python cli.py eval --model Conv1DAutoencoder --loss-fn SmoothL1Loss \
--weights-path "output/checkpoint/AE_smooth_l1_loss/2024-05-15_05-36/model_weights_epoch_5.pth" \
--figure-filename "AE.png" \
--clean-audio-filename "AE_clean.wav" \
--noisy-audio-filename "AE_noisy.wav" \
--audio-filename "AE_output.wav"
```

## .envファイルのサンプル

```bash
LOGGING=1
ONLY_FIRST_BATCH=0
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx.../xxx..."
NEPTUNE_PROJECT_NAME="xxx/yyy"
NEPTUNE_API_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXX...=="
```
