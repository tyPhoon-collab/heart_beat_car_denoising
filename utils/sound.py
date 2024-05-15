import numpy as np
from numpy.typing import ArrayLike
from scipy.io import wavfile


def save_signal_to_wav_scipy(
    data: ArrayLike,
    sample_rate: int,
    filename: str,
):
    scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(f"output/audio/{filename}", sample_rate, scaled_data)
