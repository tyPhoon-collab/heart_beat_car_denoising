from random import random
from denoising_diffusion_pytorch import GaussianDiffusion1D, Unet1D
import torch
import torch.nn as nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(GaussianDiffusion1D):
    def __init__(self, criterion: nn.Module):
        unet = Unet1D(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )
        super().__init__(unet, seq_length=5120, timesteps=100, objective="pred_v")

        self.criterion = criterion

    def set_criterion(self, criterion: nn.Module):
        """
        Sets the loss function for the model. For CLI usage.
        """
        self.criterion = criterion

    def evaluate(self, noisy: torch.Tensor):
        img = noisy

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        return img

    def p_losses(self, x_start, t, noise=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.criterion(model_out, target)

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
