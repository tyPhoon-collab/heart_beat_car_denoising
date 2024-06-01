#!/bin/bash

DEFAULT_EPOCH_SIZE=10
LEARNING_RATE=0.0001

# for transformer model
STRIDE_SAMPLES=512
SPLIT_SAMPLES=$((5120 + (8 - 1) * STRIDE_SAMPLES))

train_model() {
  local model_id=$1
  local model_type=$2
  local gain=$3

  if [ -n "$gain" ]; then
    bash train_and_eval.sh "${model_id}" "${model_type}" CombinedLoss 64 --gain "${gain}" --epoch-size $DEFAULT_EPOCH_SIZE --learning-rate $LEARNING_RATE
  else
    bash train_and_eval.sh "${model_id}" "${model_type}" CombinedLoss 64 --epoch-size $DEFAULT_EPOCH_SIZE --learning-rate $LEARNING_RATE
  fi
}

train_model_with_stride_split() {
  local model_id=$1
  local model_type=$2
  local gain=$3

  if [ -n "$gain" ]; then
    bash train_and_eval.sh "${model_id}" "${model_type}" CombinedLoss 64 --gain "${gain}" --stride-samples $STRIDE_SAMPLES --split-samples $SPLIT_SAMPLES --epoch-size $DEFAULT_EPOCH_SIZE --learning-rate $LEARNING_RATE
  else
    bash train_and_eval.sh "${model_id}" "${model_type}" CombinedLoss 64 --stride-samples $STRIDE_SAMPLES --split-samples $SPLIT_SAMPLES --epoch-size $DEFAULT_EPOCH_SIZE --learning-rate $LEARNING_RATE
  fi
}

# WaveUNet models
for gain in 0.25 0.5 0.75; do
  train_model "${gain}_WUN_CL" "WaveUNet" "${gain}"
done
train_model "WUN_CL" "WaveUNet" ""

# Conv1DAutoencoder models
for gain in 0.25 0.5 0.75; do
  train_model "${gain}_AE_CL" "Conv1DAutoencoder" "${gain}"
done
train_model "AE_CL" "Conv1DAutoencoder" ""

# PixelShuffleConv1DAutoencoder models
for gain in 0.25 0.5 0.75; do
  train_model "${gain}_SAE_CL" "PixelShuffleConv1DAutoencoder" "${gain}"
done
train_model "SAE_CL" "PixelShuffleConv1DAutoencoder" ""

# PixelShuffleConv1DAutoencoderWithTransformer models
for gain in 0.25 0.5 0.75; do
  train_model_with_stride_split "${gain}_TransSAE_CL" "PixelShuffleConv1DAutoencoderWithTransformer" "${gain}"
done
train_model_with_stride_split "TransSAE_CL" "PixelShuffleConv1DAutoencoderWithTransformer" ""
