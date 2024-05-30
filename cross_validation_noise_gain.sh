#!/bin/bash

bash train_and_eval.sh 0.25_WUN_CL WaveUNet CombinedLoss 64 0.25 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.50_WUN_CL WaveUNet CombinedLoss 64 0.5 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.75_WUN_CL WaveUNet CombinedLoss 64 0.75 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh WUN_CL WaveUNet CombinedLoss 64 1 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.25_AE_CL Conv1DAutoencoder CombinedLoss 64 0.25 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.50_AE_CL Conv1DAutoencoder CombinedLoss 64 0.5 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.75_AE_CL Conv1DAutoencoder CombinedLoss 64 0.75 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh AE_CL Conv1DAutoencoder CombinedLoss 64 1 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.25_SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 0.25 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.50_SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 0.5 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh 0.75_SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 0.75 --epoch-size 5 --learning-rate 0.0001

bash train_and_eval.sh SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 1 --epoch-size 5 --learning-rate 0.0001