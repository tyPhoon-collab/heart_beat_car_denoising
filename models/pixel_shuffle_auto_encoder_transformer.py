import torch
import torch.nn as nn

from models.pixel_shuffle_auto_encoder import PixelShuffleConv1DAutoencoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5120):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        return x + self.encoding[:seq_len, :].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x):
        # x shape: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.transpose(1, 2)  # Back to (batch_size, d_model, seq_len)
        return x


class PixelShuffleConv1DAutoencoderWithTransformer(PixelShuffleConv1DAutoencoder):
    def __init__(self):
        super(PixelShuffleConv1DAutoencoderWithTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model=2048)
        self.transformer = TransformerBlock(d_model=2048, nhead=8)

    def forward(self, x):
        x = self.encoder(x)
        x = self.positional_encoding(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        x = self.decoder(x)
        return x
