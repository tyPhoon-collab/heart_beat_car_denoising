#!/bin/bash

# bash train_and_eval.sh WUN_L1 WaveUNet L1Loss BATCH_SIZE GAIN
# bash train_and_eval.sh WUN_SL1 WaveUNet CombinedLoss BATCH_SIZE GAIN

# # bash train_and_eval.sh AE_L1_SS Conv1DAutoencoder L1Loss BATCH_SIZE GAIN --randomizer SampleShuffleRandomizer
# # bash train_and_eval.sh AE_SL1_SS Conv1DAutoencoder CombinedLoss BATCH_SIZE GAIN --randomizer SampleShuffleRandomizer

# bash train_and_eval.sh P_SAE_L1 Conv1DAutoencoder L1Loss BATCH_SIZE GAIN --with-progressive-gain
# bash train_and_eval.sh P_SAE_SL1 Conv1DAutoencoder CombinedLoss BATCH_SIZE GAIN --with-progressive-gain

# bash train_and_eval.sh SAET_L1 PixelShuffleConv1DAutoencoderWithTransformer L1Loss BATCH_SIZE GAIN --without-shuffle
# bash train_and_eval.sh SAET_SL1 PixelShuffleConv1DAutoencoderWithTransformer CombinedLoss BATCH_SIZE GAIN --without-shuffle

bash train_and_eval.sh 0.25_AE_CL Conv1DAutoencoder CombinedLoss 64 0.25 --epoch-size 5 --learning-rate 0.0001
# bash train_and_eval.sh 0.25_AE_L1 Conv1DAutoencoder L1Loss 64 0.25 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.50_AE_CL Conv1DAutoencoder CombinedLoss 64 0.5 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.75_AE_CL Conv1DAutoencoder CombinedLoss 64 0.75 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh AE_CL Conv1DAutoencoder CombinedLoss 64 1 --epoch-size 5 --learning-rate 0.0001
