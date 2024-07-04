from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.wave_u_net_enhance_base import WaveUNetEnhanceBase


class WaveUNetEnhanceTwoStageTransformer(WaveUNetEnhanceBase):
    def __init__(
        self,
        n_layers=7,
        channels_interval=24,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=2048,
    ):
        super(WaveUNetEnhanceTwoStageTransformer, self).__init__(
            n_layers, channels_interval
        )

        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        self.transformer_layer_channel = TransformerEncoderLayer(
            d_model=self.n_layers * self.channels_interval,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )
        self.transformer_layer_time = TransformerEncoderLayer(
            d_model=40,
            nhead=20,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder_channel = TransformerEncoder(
            self.transformer_layer_channel, num_layers=self.num_encoder_layers
        )
        self.transformer_encoder_time = TransformerEncoder(
            self.transformer_layer_time, num_layers=self.num_encoder_layers
        )

    def forward_latent(self, middle):
        # middle is (batch, channel, sequence)
        o1 = middle.permute(0, 2, 1)
        # convert to (batch, sequence, channel) for Transformer
        # Transformerはbatch_first=Trueになっている
        o = self.transformer_encoder_channel(o1)
        o += o1
        o2 = o.permute(0, 2, 1)
        # reconvert to (batch, sequence, channel)
        o = self.transformer_encoder_time(o2)
        o += o2
        return o
