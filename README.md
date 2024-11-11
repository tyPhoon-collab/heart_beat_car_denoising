# Car Denoising

## 実行環境

```bash
$ nvidia-smi
Sun Nov 10 23:06:57 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:01:00.0 Off |                  N/A |
|  0%   47C    P8              35W / 420W |   2715MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

## 方針

- hydraでパラメータ管理する

## 実行

基本的にmain.pyで訓練、推論、訓練＋推論のすべてが実行可能

もし動かない場合は、notebook.ipynbの各ブロックを動かすこと

```bash
python main.py
```

### パラメータ

hydraで管理している。confフォルダを参照すること

hydraの機能でCLIのタブ補完が可能

```bash
eval "$(python main.py -sc install=bash)"
```

以下でヘルプを参照可能

```bash
python main.py --help
```

| param | options                         |
| ----- | ------------------------------- |
| mode  | train, eval, train_eval         |
| data  | Raw240219, Raw240517, Raw240826 |

その他はhelpで参照可能。より詳細にコントロールしたい場合は、config.yamlを参照し、hydraの機能に従って変更する。

#### 例

- 推論モード、diffusionモデル、remoteログを有効化。

※remoteログはneptuneやdiscordなどを含む。個別に抑制可能

secret/my_secret.yamlを作成して以下を実行

```bash
python main.py mode=eval model=diffusion logging=remote secret=my_secret
```

- パラメータを調整したファイルを参照

train/tuned.yamlを編集して以下を実行

```bash
python main.py train=tuned
```

- num_encoder_layersを4に変更

モデル毎に固有なハイパラメータを指定することが可能。モデル毎にyamlファイルを作成して、そのファイル名をmodelに指定すればよい。

今回の例はwave_u_net_enhance_transformerのとき。

```bash
python main.py model.num_encoder_layers=4
```

- 推論時に音声ファイル、画像ファイル、HTMLファイルを出力

```bash
python main.py mode=eval eval=all
```

## 出力

### outputフォルダ

推論や訓練時の成果物が生成される

- output
  - audio
  - checkpoint
  - fig
  - html

### runフォルダ

- 過去の実行記録
  - hydraによって生成される
    - .hydra/config.yaml
    - .hydra/overrides.yaml
    - main.log
