import matplotlib.pyplot as plt


def plot_three_signals(upper, middle, lower, upper_label, middle_label, lower_label):
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
    plt.show()


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


def plot_signal(signal, label):
    plt.plot(signal)
    plt.title(label)
    plt.show()
