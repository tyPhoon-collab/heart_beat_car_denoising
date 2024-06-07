bash train_and_eval.sh 0_WUN_CL WaveUNet CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --gain 0
bash train_and_eval.sh 0_WUN_L1 WaveUNet L1Loss 64 --epoch-size 10 --learning-rate 0.0001 --gain 0
bash train_and_eval.sh 0_AE_CL Conv1DAutoencoder CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --gain 0
bash train_and_eval.sh 0_SAE_CL PixelShuffleConv1DAutoencoder CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --gain 0
bash train_and_eval.sh 0_TransSAE_CL PixelShuffleConv1DAutoencoderWithTransformer CombinedLoss 64 --epoch-size 10 --learning-rate 0.0001 --gain 0
