import os
import numpy as np
from numpy.typing import ArrayLike
from scipy.io import wavfile


def save_signal_to_wav_scipy(
    data: ArrayLike,
    sample_rate: int,
    filename: str,
    base_dir: str = "output/audio",
):
    scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)
    wavfile.write(path, sample_rate, scaled_data)
    print(f"Audio saved to {path}")
