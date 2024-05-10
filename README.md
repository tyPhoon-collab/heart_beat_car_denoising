# Heart beat car denoising

## Set up

計算機サーバーで動かす。Larkを見てセットアップする。.containerは作成済み

重要なファイルは以下の通り

- .env
  - ロガー用の変数を置く
    - LOGGING
    - DISCORD_WEBHOOK_URL
    - NEPTUNE_PROJECT_NAME
    - NEPTUNE_API_TOKEN
  - なくても動く
- dataフォルダ
  - 杉浦先生が用意したMAT形式のデータを置く
  - gitで管理していないため、必要に応じてファイルを追加する
  - Larkの心拍推定からダウンロードする
