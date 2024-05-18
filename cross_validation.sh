#!/bin/bash

bash train_and_eval.sh WUN_L1 WaveUNet L1Loss
bash train_and_eval.sh WUN_SL1 WaveUNet SmoothL1Loss

bash train_and_eval.sh AE_L1_SS Conv1DAutoencoder L1Loss --randomizer SampleShuffleRandomizer
bash train_and_eval.sh AE_SL1_SS Conv1DAutoencoder SmoothL1Loss --randomizer SampleShuffleRandomizer

bash train_and_eval.sh AE_L1 Conv1DAutoencoder L1Loss
bash train_and_eval.sh AE_SL1 Conv1DAutoencoder SmoothL1Loss

bash train_and_eval.sh P_AE_L1 Conv1DAutoencoder L1Loss --with-progressive-gain
bash train_and_eval.sh P_AE_SL1 Conv1DAutoencoder SmoothL1Loss --with-progressive-gain

bash train_and_eval.sh SAE_L1 PixelShuffleConv1DAutoencoder L1Loss
bash train_and_eval.sh SAE_SL1 PixelShuffleConv1DAutoencoder SmoothL1Loss

bash train_and_eval.sh SAET_L1 PixelShuffleConv1DAutoencoderWithTransformer L1Loss
bash train_and_eval.sh SAET_SL1 PixelShuffleConv1DAutoencoderWithTransformer SmoothL1Loss