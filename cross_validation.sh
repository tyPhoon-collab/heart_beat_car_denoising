#!/bin/bash

# bash train_and_eval.sh WUN_L1 WaveUNet L1Loss BATCH_SIZE GAIN
# bash train_and_eval.sh WUN_SL1 WaveUNet CombinedLoss BATCH_SIZE GAIN

# # bash train_and_eval.sh AE_L1_SS Conv1DAutoencoder L1Loss BATCH_SIZE GAIN --randomizer SampleShuffleRandomizer
# # bash train_and_eval.sh AE_SL1_SS Conv1DAutoencoder CombinedLoss BATCH_SIZE GAIN --randomizer SampleShuffleRandomizer

# bash train_and_eval.sh P_SAE_L1 Conv1DAutoencoder L1Loss BATCH_SIZE GAIN --with-progressive-gain
# bash train_and_eval.sh P_SAE_SL1 Conv1DAutoencoder CombinedLoss BATCH_SIZE GAIN --with-progressive-gain

# bash train_and_eval.sh SAET_L1 PixelShuffleConv1DAutoencoderWithTransformer L1Loss BATCH_SIZE GAIN --without-shuffle
# bash train_and_eval.sh SAET_SL1 PixelShuffleConv1DAutoencoderWithTransformer CombinedLoss BATCH_SIZE GAIN --without-shuffle

