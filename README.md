# Heart beat car denoising

## セットアップ

1. cloneする
2. 計算機サーバーで動かす。Larkを見てセットアップする。.containerは作成済み
3. 以下のファイルを追加する

- .env
  - ロガー用の変数を置く
    - [サンプル](#envファイルのサンプル)
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

とくに理由はない

## 実行コマンド例

```bash
bash train_and_eval.sh PSAE PixelShuffleConv1DAutoencoder CombinedLoss
```

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
ONLY_FIRST_BATCH=0
STDOUT_LOGGING=1
REMOTE_LOGGING=1
DISCORD_LOGGING=1
NEPTUNE_LOGGING=1
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx.../xxx..."
NEPTUNE_PROJECT_NAME="xxx/yyy"
NEPTUNE_API_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXX...=="
```

## Note

### 5/2

- インストラクション
  - ルールベースのみでは厳しいので、ノイズ除去部分をNNで行う
- 実装方針の決定
  - サンプルベースでランダム化した100km.matとStop.matを足し合わせたものを雑音とする

### 5/9

- DataSetとDataLoaderの仮実装
- WaveUNetの仮実装

### 5/10

- Autoencoderの仮実装

### 5/16

- WaveUNetによる雑音除去の検証
- Autoencoderによる雑音除去の検証
- 今後の方針
  - PixelShuffleベースのAutoEncoderの実装
    - skipコネクションが良くない可能性がある
  - TransformerBlockを使用した上記のモデルの実装
    - nheadが重要
    - 500サンプル程度ずらした6セットを1バッチとする
  - Diffusionモデルも実装してみる
  - Wavelet変換で周波数成分を算出して、L1Lossを出す
    - 連続Wavelet変換
      - scipy.signal.cwt
      - scipy.signal.morletをマザーwaveletとして使用する
    - 基底は36
    - 波形のL1lossと足し合わせた損失関数を定義する
  - 1000Hzのデータを扱う

### 5/17

- 位相シャッフルの実装
- 既存のサンプルベースのシャッフルの予期しない動作の修正
- PixelShuffleAutoEncoderの仮実装
- PixelShuffleAutoEncoderWithTransformerの仮実装

### 5/18

- 全体的な検証
  - 精度の悪化を確認
    - おそらくランダマイズの処理に依存していた上、ランダマイズの処理が正しくなかった
    - mutableなメソッドを用いていたため、元データが書き変わっていた

### 5/22

- ランダマイズなしでの検証
  - そんなに精度は高くない

### 5/23

- 学習率 0.0001
- バッチサイズ 64
- エポック数を増やす
- 柔軟なMatLoaderを提供する機能を追加する
- 適切なTransformerの学習環境を整える

### 5/30

- リファクタリング、neptuneへの統合の強化
  - [neptune](https://app.neptune.ai/o/typhoon/org/heart-beat-car-denoising/runs/table?viewId=9c30a6de-9bfa-42dd-9786-285c972793ef&detailsTab=images)への招待
- ノイズゲインを0.25倍した場合は、ある程度の性能が見込まれた
- バンドパスを掛けたデータで訓練、評価する

### 6/2

- 0.25, 0.50, 0.75, 1.0倍のノイズゲインに変更した場合の検証
  - WaveUNetかつ0.5の時の精度が良さそう

### 6/7

- BatchNormやPReLUの実験
  - BatchNormは振幅が極端に大きくなる
- 両端が極端に大きくなる
  - biasの影響？
- 今後の方針
  - 両端のピークを抑制する
    - output_paddingなど、decoderを確認する
  - 事前学習
    - gainが0の状態で学習したモデルを事前学習モデルとして読み込む
      - 例）0_WUN_CLを読み込んで学習する
  - プログレッシブ学習[0, 0.5]
  - epoch数を増やしてみる
  - AdamWをためしてみる
  - 警告の抑制
    - Dockerの環境を再構築など

### 6/11

- 警告の抑制
  - 最新のimageを使用した際に、警告が出るとのこと。古いimageを採用
