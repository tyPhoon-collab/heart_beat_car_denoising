import numpy as np
from scipy.io import wavfile


def save_signal_to_wav_scipy(data, sample_rate, filename):
    # 最大振幅を16ビット整数の範囲にスケール
    scaled_data = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(filename, sample_rate, scaled_data)


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from dataset.loader import MatLoader
    from dataset.sampling_rate_converter import ScipySamplingRateConverter

    def convert_to_wav(filename, output_filename, *, output_sample_rate=32000):
        loader = MatLoader(
            filename,
            ["Time", "ECG", "ch1z", "ch2z", "ch3z", "ch4z", "ch5z", "ch6z"],
        )
        data = loader.load()

        single_data = data["ch1z"]
        single_data = ScipySamplingRateConverter(
            32000,
            output_sample_rate,
        ).convert(single_data)

        save_signal_to_wav_scipy(
            single_data,
            output_sample_rate,
            f"output/audio/{output_filename}.wav",
        )

    convert_to_wav("data/Stop.mat", "stop_32000")
    convert_to_wav("data/Idling.mat", "idling_32000")
    convert_to_wav("data/100km.mat", "100km_32000")

    convert_to_wav("data/Stop.mat", "stop_1024", output_sample_rate=1024)
    convert_to_wav("data/Idling.mat", "idling_1024", output_sample_rate=1024)
    convert_to_wav("data/100km.mat", "100km_1024", output_sample_rate=1024)
