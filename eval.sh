#!/bin/bash

model="WaveUNetEnhanceTransformer"
loss_fn="L1Loss"
batch_size=64

# 使用方法を表示する関数
show_usage() {
    echo "Usage: $0 <id> <path_to_weights> <data_folder>"
    echo "Arguments:"
    echo "  <id>                A unique identifier for the evaluation."
    echo "  <path_to_weights>   Path to the weights file."
    echo "  <data_folder>       Path to the folder containing the data."
}

# 引数が -h または --help の場合、使用方法を表示して終了
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# 引数が3つ未満の場合、エラーメッセージを表示して終了
if [ $# -lt 3 ]; then
    echo "Error: Insufficient arguments provided."
    show_usage
    exit 1
fi

id=$1
path=$2
data_folder=$3

# 評価を実行する関数
run_eval() {
    local gain=$1
    local figure_filename="${id}_gain_${gain}.png"
    local audio_filename="${id}_gain_${gain}_output.wav"
    local clean_audio_filename="${id}_gain_${gain}_clean.wav"
    local noisy_audio_filename="${id}_gain_${gain}_noisy.wav"

    python cli.py eval \
        --model "$model" \
        --loss-fn "$loss_fn" \
        --batch-size "$batch_size" \
        --gain "$gain" \
        --figure-filename "$figure_filename" \
        --audio-filename "$audio_filename" \
        --clean-audio-filename "$clean_audio_filename" \
        --noisy-audio-filename "$noisy_audio_filename" \
        --weights-path "$path" \
        --data-folder "$data_folder"
}

# パスの存在を確認
if [ ! -f "$path" ]; then
    echo "Error: The specified weights file does not exist: $path"
    exit 1
fi

# 異なるgain値で評価を実行
for gain in 0 0.25 0.5 0.75 0.8 1; do
    run_eval $gain
done
