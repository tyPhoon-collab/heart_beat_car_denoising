# !/bin/bash

# 6/17の会議用に必要なデータを出力するスクリプト

# 音声強調のWaveUNetモデルを試す
#   WaveUNetEnhance
#   ゲインは0, 0.25, 0.5, 0.75, 1
#   1に関しては転移学習の結果も出力する
#   プログレッシブ学習も同様に行う。緩やかにするためにepoch-sizeを40, epoch-toを20, gainを0.5に設定
# 三種類の損失関数を試す
#   L1Loss、WeightedLoss, WeightedCombinedLoss
# 評価をしてみる
#   一番もっともらしいモデルに対して、各ゲインの評価を行う

# 緑色でテキストを表示する関数
print_green() {
    tput setaf 2
    echo "$1"
    tput sgr0
}

# モデルの設定
model="WaveUNetEnhance"
model_alias="WUNE"

# ゲインと損失タイプの組み合わせ
gains=("0" "0.25" "0.5" "0.75" "1")
loss_types=("L1Loss" "WeightedLoss" "WeightedCombinedLoss")
loss_types_alias=("L1" "WL" "WCL")  # 各損失タイプに対応するエイリアス

# 共通のパラメータ
batch_size=16
epoch_size=10
learning_rate=0.0001

prog_epoch_size=40
prog_epoch_to=20
prog_max_gain=0.5

# 通常の学習と評価のループ
for gain in "${gains[@]}"; do
    for i in "${!loss_types[@]}"; do
        loss_type=${loss_types[$i]}
        alias=${loss_types_alias[$i]}
        exp_name="${gain}_${model_alias}_${alias}"  # exp_nameをエイリアスを使って生成
        print_green "Running training and evaluation for gain $gain and loss $loss_type"
        bash train_and_eval.sh $exp_name $model $loss_type $batch_size --epoch-size $epoch_size --learning-rate $learning_rate --gain $gain

        # ゲインが1の場合、転移学習も実行
        if [[ $gain == "1" ]]; then
            pretrained_path="output/checkpoint/${model}_${loss_type}/${gain}_${model_alias}_${alias}/model_weights_best.pth"
            exp_name="T_${gain}_${model_alias}_${alias}"
            print_green "Running transfer learning for gain $gain and loss $loss_type"
            bash train_and_eval.sh $exp_name $model $loss_type $batch_size --epoch-size $epoch_size --learning-rate $learning_rate --gain $gain --pretrained-weights-path $pretrained_path
        fi
    done
done

# プログレッシブ学習
for i in "${!loss_types[@]}"; do
    loss_type=${loss_types[$i]}
    alias=${loss_types_alias[$i]}
    exp_name="TP_${prog_max_gain}_${model_alias}_${alias}"
    pretrained_path="output/checkpoint/${model}_${loss_type}/0_${model_alias}_${alias}/model_weights_best.pth"
    print_green "Running progressive training for loss $loss_source"
    bash train_and_eval.sh $exp_name $model $loss_type $batch_size --learning-rate $learning_rate --with-progressive-gain --epoch-size $prog_epoch_size --progressive-epoch-to $prog_epoch_to --gain $prog_max_gain --pretrained-weights-path $pretrained_path
done