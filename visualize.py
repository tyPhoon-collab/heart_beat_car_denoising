import matplotlib.pyplot as plt


def plot_two_signals(upper, lower, upper_label, lower_label):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(upper, label=upper_label)
    plt.title(upper_label)
    plt.legend()

    plt.subplot(212)
    plt.plot(lower, label=lower_label)
    plt.title(lower_label)
    plt.legend()
    plt.show()
