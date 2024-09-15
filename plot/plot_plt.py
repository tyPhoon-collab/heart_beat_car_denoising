import os
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


def _finalize_plot(filename=None):
    assert filename is None or filename.endswith(".png") or filename.endswith(".pdf")

    plt.tight_layout()
    if filename:
        save_directory = "output/fig"
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(os.path.join(save_directory, filename))
        plt.close()
        print(f"Figure saved to {os.path.join(save_directory, filename)}")
    else:
        plt.show()


def plot_signals(signals, labels):
    n = len(signals)
    plt.figure(figsize=(12, 3 * n))
    for i, (signal, label) in enumerate(zip(signals, labels), 1):
        plt.subplot(n, 1, i)
        plt.plot(signal, label=label)
        plt.title(label)
        plt.legend()


def show_signals(signals, labels, filename=None):
    plot_signals(signals, labels)
    _finalize_plot(filename)


def show_signal(signal, label, filename=None):
    show_signals([signal], [label], filename)


def show_wavelet(
    waveform: np.ndarray,
    wavelets: np.ndarray,
    max_scale: int,
    filename=None,
):
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 元の波形のプロット
    ax_top.plot(waveform)
    ax_top.set_title("Original Waveform")
    ax_top.set_xlabel("Sample")
    ax_top.set_ylabel("Amplitude")

    # CWTの結果のプロット
    ax_bottom.imshow(
        wavelets,
        extent=[0, len(waveform), 1, max_scale],
        cmap="PRGn",
        aspect="auto",
        vmax=abs(wavelets).max(),
        vmin=-abs(wavelets).max(),
    )
    ax_bottom.set_title("Continuous Wavelet Transform (CWT)")
    ax_bottom.set_xlabel("Sample")
    ax_bottom.set_ylabel("Scale")

    _finalize_plot(filename)


def show_spectrogram(
    audio_data,
    sr,
    title="Spectrogram",
    *,
    figsize=(10, 4),
    ylim=None,
    filename=None,
):
    # 短時間フーリエ変換を実行
    S = np.abs(librosa.stft(audio_data))

    # 振幅スペクトルをデシベル単位に変換
    D = librosa.amplitude_to_db(S, ref=np.max)

    # スペクトログラムを表示
    plt.figure(figsize=figsize)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)

    if ylim is not None:
        plt.ylim(ylim)

    _finalize_plot(filename)
