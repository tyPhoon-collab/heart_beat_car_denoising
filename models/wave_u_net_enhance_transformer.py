from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.wave_u_net_enhance_base import WaveUNetEnhanceBase


class WaveUNetEnhanceTransformer(WaveUNetEnhanceBase):
    def __init__(
        self,
        n_layers=10,
        channels_interval=24,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=2048,
    ):
        super(WaveUNetEnhanceTransformer, self).__init__(n_layers, channels_interval)

        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        self.transformer_layer = TransformerEncoderLayer(
            d_model=self.n_layers * self.channels_interval,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_layer, num_layers=self.num_encoder_layers
        )

    def forward_latent(self, middle):
        # Reshape for Transformer (batch, sequence, feature)
        o = middle.permute(2, 0, 1)
        o = self.transformer_encoder(o)
        o = o.permute(1, 2, 0)
        return o
