from matplotlib import pyplot as plt


def __finalize_plot(filename=None):
    """グラフの最終処理を行うヘルパー関数。
    filenameが指定されていれば保存し、指定されていなければ表示する。"""
    plt.tight_layout()  # サブプロット間の適切な間隔を確保
    if filename:
        plt.savefig(f"figs/{filename}")  # ファイルに保存
        plt.close()  # フィギュアを閉じる
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
