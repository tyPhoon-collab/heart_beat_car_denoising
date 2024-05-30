#!/bin/bash

# 色付きメッセージ出力のための関数定義
print_error() {
    tput setaf 1
    echo "$1"
    tput sgr0
}

print_yellow() {
    tput setaf 3
    echo "$1"
    tput sgr0
}

print_green() {
    tput setaf 2
    echo "$1"
    tput sgr0
}

print_separator() {
    echo "----------------------------------------"
}

# エラーハンドリング関数
check_status() {
    if [ $? -ne 0 ]; then
        print_error "$1"
        exit 1
    fi
}

# 引数チェック
if [ "$#" -lt 4 ]; then
    print_error "Usage: $0 <ID> <MODEL> <LOSS_FN> <BATCH_SIZE> [--gain VALUE] [--stride-samples VALUE] [--split-samples VALUE] [<ANOTHER_TRAINING_OPTIONS>]"
    exit 1
fi

# 共通パラメータの定義
ID=$1
MODEL=$2
LOSS_FN=$3
BATCH_SIZE=$4
shift 4

GAIN=""
STRIDE_SAMPLES=""
SPLIT_SAMPLES=""
ANOTHER_TRAINING_OPTIONS=""

# オプショナルなパラメータの処理
while (( "$#" )); do
    case "$1" in
        --gain)
            GAIN="--gain $2"
            shift 2
            ;;
        --stride-samples)
            STRIDE_SAMPLES="--stride-samples $2"
            shift 2
            ;;
        --split-samples)
            SPLIT_SAMPLES="--split-samples $2"
            shift 2
            ;;
        *)
            ANOTHER_TRAINING_OPTIONS="$ANOTHER_TRAINING_OPTIONS $1"
            shift
            ;;
    esac
done

FOLDER_NAME="${MODEL}_${LOSS_FN}"
CHECKPOINT_DIR="output/checkpoint/$FOLDER_NAME"
WEIGHTS_PATH="$CHECKPOINT_DIR/$ID/model_weights_epoch_5.pth"
FIGURE_FILENAME="${ID}.png"

COMMON_OPTIONS="--model $MODEL --loss-fn $LOSS_FN --batch-size $BATCH_SIZE $GAIN $STRIDE_SAMPLES $SPLIT_SAMPLES"

# トレーニングと評価
print_separator
print_yellow "Training the model"
print_separator
python cli.py train $COMMON_OPTIONS --checkpoint-dir "$CHECKPOINT_DIR" --model-id "$ID" $ANOTHER_TRAINING_OPTIONS
check_status "Training failed."

print_separator
print_yellow "Evaluating the model"
print_separator
python cli.py eval $COMMON_OPTIONS --weights-path "$WEIGHTS_PATH" --figure-filename "$FIGURE_FILENAME"
check_status "Evaluation failed."

print_separator
print_green "Training and evaluation completed successfully."
print_separator
