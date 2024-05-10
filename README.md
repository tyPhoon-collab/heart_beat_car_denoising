# Heart beat car denoising

## セットアップ

1. cloneする
2. 計算機サーバーで動かす。Larkを見てセットアップする。.containerは作成済み
3. 以下のファイルを追加する

- .env
  - ロガー用の変数を置く
    - [サンプル](#サンプル)
    - LOGGING
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

## サンプル

```bash
LOGGING=1
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/xxx.../xxx..."
NEPTUNE_PROJECT_NAME="xxx/yyy"
NEPTUNE_API_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXX...=="
```
