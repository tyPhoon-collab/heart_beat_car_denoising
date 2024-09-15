from dataclasses import dataclass

from logger.evaluation_logger import EvaluationLogger
from numpy.typing import ArrayLike

from utils.sound import save_signal_to_wav_scipy


@dataclass
class AudioEvaluationLogger(EvaluationLogger):
    sample_rate: int
    audio_filename: str | None
    clean_audio_filename: str | None
    noisy_audio_filename: str | None

    def on_data(self, noisy: ArrayLike, clean: ArrayLike, output: ArrayLike):
        if self.clean_audio_filename is not None:
            save_signal_to_wav_scipy(clean, self.sample_rate, self.clean_audio_filename)
        if self.noisy_audio_filename is not None:
            save_signal_to_wav_scipy(noisy, self.sample_rate, self.noisy_audio_filename)
        if self.audio_filename is not None:
            save_signal_to_wav_scipy(output, self.sample_rate, self.audio_filename)

    def on_average_loss(self, loss: float):
        pass
