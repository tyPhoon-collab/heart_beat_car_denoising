import torch
import torch.nn as nn

from models.modules.pixel_shuffle import PixelShuffle1d, PixelUnshuffle1d


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


class PixelShuffleConv1DAutoencoderWithTransformerNorm(nn.Module):
    def __init__(self):
        super(PixelShuffleConv1DAutoencoderWithTransformerNorm, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                1, 16, kernel_size=3, stride=1, padding=1
            ),  # output is (16, 5120)
            nn.BatchNorm1d(16),
            nn.PReLU(),
            PixelUnshuffle1d(2),  # output is (32, 2560)
            nn.Conv1d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # output is (64, 2560)
            nn.BatchNorm1d(64),
            nn.PReLU(),
            PixelUnshuffle1d(2),  # output is (128, 1280)
            nn.Conv1d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # output is (256, 1280)
            nn.BatchNorm1d(256),
            nn.PReLU(),
            PixelUnshuffle1d(2),  # output is (512, 640)
            nn.Conv1d(
                512, 1024, kernel_size=3, stride=1, padding=1
            ),  # output is (1024, 640)
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            PixelUnshuffle1d(2),  # output is (2048, 320)
        )

        self.positional_encoding = PositionalEncoding(d_model=2048)
        self.transformer = TransformerBlock(d_model=2048, nhead=8)

        self.decoder = nn.Sequential(
            nn.Conv1d(
                2048, 1024, kernel_size=3, stride=1, padding=1
            ),  # output is (1024, 320)
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            PixelShuffle1d(2),  # output is (512, 640)
            nn.Conv1d(
                512, 256, kernel_size=3, stride=1, padding=1
            ),  # output is (256, 640)
            nn.BatchNorm1d(256),
            nn.PReLU(),
            PixelShuffle1d(2),  # output is (128, 1280)
            nn.Conv1d(
                128, 64, kernel_size=3, stride=1, padding=1
            ),  # output is (64, 1280)
            nn.BatchNorm1d(64),
            nn.PReLU(),
            PixelShuffle1d(2),  # output is (32, 2560)
            nn.Conv1d(
                32, 16, kernel_size=3, stride=1, padding=1
            ),  # output is (16, 2560)
            nn.BatchNorm1d(16),
            nn.PReLU(),
            PixelShuffle1d(2),  # output is (8, 5120)
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),  # output is (1, 5120)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.positional_encoding(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        x = self.decoder(x)
        return x
