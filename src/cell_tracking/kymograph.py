import numpy as np


def plot_kymograph(
    line,
    ax=None,
    figsize=(10, 6),
    aspect="auto",
    cmap="PRGn",
    origin="upper",
    label="Kymograph",
    xlabel="Position along line",
    ylabel="Time (frame index)",
    title="Kymograph",
    show=True,
    save_path=None,
):
    """
    Plots a kymograph from a 2D array.

    Args:
        line (np.ndarray): 2D array representing the kymograph data.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        figsize (tuple): Size of the figure in inches (width, height).
        aspect (str): Aspect ratio of the plot. Default is 'auto'.
        cmap (str): Colormap to use for the kymograph. Default is 'PRGn'.
        origin (str): Origin of the plot. Default is 'upper'.
        label (str): Label for the colorbar.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        show (bool): Whether to display the plot immediately.

    Returns:
        None: Just displays the kymograph.
    """


def vector_kymograph(
    arr, values=["x dir"], method=np.median, combine=True, save_path=None
):
    """
    Create and optionally combine kymographs from flow data.
    """
