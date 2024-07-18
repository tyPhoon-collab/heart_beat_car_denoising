# Heart beat car denoising

## セットアップ

### ZIPから

ZIPを展開後、Dockerの作成と起動

Macで実装したため、Windowsの場合は、WSLでUbuntu 22をインストールし、そこでDockerを起動する

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
- `eval.py`
  - 杉浦先生から指示があった評価ケース(30種類)を実行する
- `.container/`
  - Dockerの定義
- `solver.py`
  - 訓練と評価の実装。CLIでラップされる

## 実行コマンド例

CLIとbashファイルを用意している。bashファイルはCLIをラップしたもの

それぞれオプションを確認して、適切な引数を設定してください。

### bashファイルを使用する場合

```bash
# オプションの確認
bash train_and_eval.sh

# 以下、使用例
# IDは任意のもの、重みの保存時のフォルダ名になる
# 末尾の数字はバッチサイズ
bash train_and_eval.sh WUET WaveUNetEnhanceTwoStageTransformer WeightedLoss 64 --gain 1 \
  --learning-rate 0.000025 --epoch-size 100 --progressive-epoch-to 50 --with-progressive-gain \
  --pretrained-weights-path output/checkpoint/path_to_weights.pth
```

### CLIを使用する場合

```bash
# オプションの確認
python cli.py train --help
python cli.py eval --help

# 以下、使用例
python cli.py train --model WaveUNetEnhanceTwoStageTransformer --loss-fn WeightedLoss --batch-size 64 --gain 1 \
  --learning-rate 0.000025 --epoch-size 100 --progressive-epoch-to 50 --with-progressive-gain \
  --pretrained-weights-path output/checkpoint/path_to_weights.pth

# python cli.py evalもあるが、eval.pyを動かすほうが正確な対照実験ができる
# 評価の実行
python eval.py
```

## .envファイルのサンプル

このディレクトリの直下に.envファイルを作成して、以下の環境変数を定義する。

Discordやneptuneなどの統合の際に必要。なくても良いですが、`STDOUT_LOGGING=1`はあると便利です。

```bash
ONLY_FIRST_BATCH=0
STDOUT_LOGGING=1
REMOTE_LOGGING=1
DISCORD_LOGGING=1
NEPTUNE_LOGGING=1
NEPTUNE_SAVE_MODEL_STATE=1
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx.../xxx..."
NEPTUNE_PROJECT_NAME="xxx/yyy"
NEPTUNE_API_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXX...=="
```
