import numpy as np
from scipy.io import wavfile


def save_signal_to_wav_scipy(data, sample_rate, filename):
    # 最大振幅を16ビット整数の範囲にスケール
    scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(f"output/audio/{filename}", sample_rate, scaled_data)
