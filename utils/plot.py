import os
import librosa
from matplotlib import pyplot as plt
import numpy as np


def __finalize_plot(filename=None):
    """グラフの最終処理を行うヘルパー関数。
    filenameが指定されていれば保存し、指定されていなければ表示する。"""
    plt.tight_layout()  # サブプロット間の適切な間隔を確保
    if filename:
        save_directory = "output/fig"
        os.makedirs(save_directory, exist_ok=True)  # ディレクトリがなければ作成
        plt.savefig(os.path.join(save_directory, filename))  # ファイルに保存
        plt.close()  # フィギュアを閉じる
        print(f"Figure saved to {os.path.join(save_directory, filename)}")
    else:
        plt.show()  # グラフを表示


def plot_three_signals(
    upper, middle, lower, upper_label, middle_label, lower_label, filename=None
):
    plt.figure(figsize=(12, 6))
    plt.subplot(311)
    plt.plot(upper, label=upper_label)
    plt.title(upper_label)
    plt.legend()

    plt.subplot(312)
    plt.plot(middle, label=middle_label)
    plt.title(middle_label)
    plt.legend()

    plt.subplot(313)
    plt.plot(lower, label=lower_label)
    plt.title(lower_label)
    plt.legend()

    __finalize_plot(filename)


def plot_two_signals(upper, lower, upper_label, lower_label, filename=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(upper, label=upper_label)
    plt.title(upper_label)
    plt.legend()

    plt.subplot(212)
    plt.plot(lower, label=lower_label)
    plt.title(lower_label)
    plt.legend()

    __finalize_plot(filename)


def plot_signal(signal, label, filename=None):
    plt.figure()
    plt.plot(signal)
    plt.title(label)
    __finalize_plot(filename)


def plot_spectrogram(
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

    __finalize_plot(filename)
