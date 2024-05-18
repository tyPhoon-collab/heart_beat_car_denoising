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

# 引数チェック
if [ "$#" -lt 3 ]; then
    print_error "Usage: $0 <ID> <MODEL> <LOSS_FN> [<ANOTHER_TRAINING_OPTIONS>]"
    exit 1
fi

# 引数を変数に格納
ID=$1
MODEL=$2
LOSS_FN=$3

shift 3

ANOTHER_TRAINING_OPTIONS=$*

FOLDER_NAME="${MODEL}_${LOSS_FN}"
CHECKPOINT_DIR="output/checkpoint/$FOLDER_NAME"
WEIGHTS_PATH="$CHECKPOINT_DIR/$ID/model_weights_epoch_5.pth"
FIGURE_FILENAME="${ID}.png"
# CLEAN_AUDIO_FILENAME="${ID}_clean.wav"
# NOISY_AUDIO_FILENAME="${ID}_noisy.wav"
# AUDIO_FILENAME="${ID}_output.wav"

print_separator
print_yellow "Training the model"
print_separator
python cli.py train --model "$MODEL" --loss-fn "$LOSS_FN" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --model-id "$ID" $ANOTHER_TRAINING_OPTIONS

# エラーハンドリング
if [ $? -ne 0 ]; then
    print_error "Training failed."
    exit 1
fi

print_separator
print_yellow "Evaluating the model"
print_separator
python cli.py eval --model "$MODEL" --loss-fn "$LOSS_FN" \
    --weights-path "$WEIGHTS_PATH" \
    --figure-filename "$FIGURE_FILENAME" \
    # --clean-audio-filename "$CLEAN_AUDIO_FILENAME" \
    # --noisy-audio-filename "$NOISY_AUDIO_FILENAME" \
    # --audio-filename "$AUDIO_FILENAME"

# エラーハンドリング
if [ $? -ne 0 ]; then
    print_error "Evaluation failed."
    exit 1
fi

print_separator
print_green "Training and evaluation completed successfully."
print_separator
