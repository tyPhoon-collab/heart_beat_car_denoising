import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _finalize_plotly_figure(fig: go.Figure, filename: str | None):
    assert filename is None or filename.endswith(".html")
    if filename:
        save_directory = "output/html"
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, filename)
        fig.write_html(path)
        print(f"Figure saved to {path}")
    else:
        fig.show()


def plot_plotly_signals(signals, labels) -> go.Figure:
    fig = go.Figure()
    for signal, label in zip(signals, labels):
        fig.add_trace(
            go.Scattergl(
                x=list(range(len(signal))),
                y=signal,
                mode="lines",
                name=label,
            ),
        )

    return fig


def plot_plotly_subplot_signals(signals, labels) -> go.Figure:
    """
    subplotで描画する。[plot_plotly_signals]と比べて重たい。
    """
    num_signals = len(signals)

    # サブプロットの作成（行数: num_signals, 列数: 1）
    fig = make_subplots(
        rows=num_signals,
        cols=1,
        shared_xaxes=True,
        subplot_titles=labels,
        vertical_spacing=0.05,
    )

    for i, (signal, label) in enumerate(zip(signals, labels), start=1):
        fig.add_trace(
            go.Scattergl(
                x=list(range(len(signal))),
                y=signal,
                mode="lines",
                name=label,
            ),
            row=i,
            col=1,
        )

    fig.update_xaxes(title_text="Sample", row=num_signals, col=1)

    for i in range(1, num_signals + 1):
        fig.update_yaxes(title_text=labels[i - 1], row=i, col=1)

    return fig


def show_plotly_signals(signals, labels, filename=None):
    fig = plot_plotly_signals(signals, labels)
    _finalize_plotly_figure(fig, filename)


if __name__ == "__main__":
    show_plotly_signals([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ["a", "b", "c"], "test.html")
