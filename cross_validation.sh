#!/bin/bash

STRIDE_SAMPLES=512
SPLIT_SAMPLES=$((5120 + (8 - 1) * STRIDE_SAMPLES))

bash train_and_eval.sh TransSAE_CL PixelShuffleConv1DAutoencoderWithTransformer CombinedLoss 64 --stride-samples $STRIDE_SAMPLES --split-samples $SPLIT_SAMPLES --epoch-size 5 --learning-rate 0.0001

