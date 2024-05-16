import torch.nn as nn


class PixelShuffle1d(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        n, c, l = x.size()
        new_c = c // self.upscale_factor
        new_l = l * self.upscale_factor
        x = x.view(n, new_c, self.upscale_factor, l)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(n, new_c, new_l)
        return x


class PixelUnshuffle1d(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1d, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        n, c, l = x.size()
        new_c = c * self.downscale_factor
        new_l = l // self.downscale_factor
        x = x.view(n, c, new_l, self.downscale_factor)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(n, new_c, new_l)
        return x
