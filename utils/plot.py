import os
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np


def _finalize_plot(filename=None):
    """グラフの最終処理を行うヘルパー関数。
    filenameが指定されていれば保存し、指定されていなければ表示する。"""
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
