bash train_and_eval.sh P_WUN_CL WaveUNet CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --with-progressive-gain
bash train_and_eval.sh P_AE_CL Conv1DAutoencoder CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --with-progressive-gain
bash train_and_eval.sh P_SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --with-progressive-gain

STRIDE_SAMPLES=512
SPLIT_SAMPLES=$((5120 + (8 - 1) * STRIDE_SAMPLES))

bash train_and_eval.sh P_TransSAE_CL PixelShuffleConv1DAutoencoderWithTransformer CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --stride-samples $STRIDE_SAMPLES --split-samples $SPLIT_SAMPLES --with-progressive-gain
