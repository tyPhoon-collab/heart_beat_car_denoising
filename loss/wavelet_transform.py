# GPU上でWavelet変換を行う関数
import torch
import ptwt
import pywt


def wavelet_transform(signal: torch.Tensor, scales, wavelet_name="morl"):
    wavelet = pywt.ContinuousWavelet(wavelet_name)  # type: ignore
    cwt_coeffs = ptwt.cwt(signal, scales, wavelet)
    return cwt_coeffs
