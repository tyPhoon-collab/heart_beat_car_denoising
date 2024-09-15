import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    df = pd.DataFrame({label: signal for label, signal in zip(labels, signals)})
    fig = px.line(df)

    return fig


def show_plotly_signals(signals, labels, filename=None):
    fig = plot_plotly_signals(signals, labels)
    _finalize_plotly_figure(fig, filename)


if __name__ == "__main__":
    show_plotly_signals([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ["a", "b", "c"], "test.html")
