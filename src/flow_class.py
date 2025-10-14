import numpy as np
import tiffclass as tc


class FlowClass:
    def __init__(self, img: tc.Tiff, method: str, save_path: str):
        """
        Base class for optical flow calculation and visualization.

        Args:
            img (tc.Tiff): Input image stack.
            method (str): Optical flow calculation method.
            save_path (str): Path to save output visualizations.
        """
        self.img = img
        self.method = method
        self.save_path = save_path
        self.flow_arr = self._calculate_flow()
        self.flow_xyz = self._flow_arr_to_xyz()

    def _preprocess_img(self, **kwargs) -> np.array:
        """
        Preprocess the input image stack for optical flow calculation.

        Args:
            **kwargs: Preprocessing parameters, which may include

        Returns:
            np.array: Preprocessed image stack.
        """
        raise NotImplementedError

    def _calculate_flow(self, **kwargs) -> np.array:
        """
        Calculate optical flow using the specified method.

        Args:
            **kwargs: Preprocessing parameters, which may include:
                - arr (np.array): Stack for optical flow processing
                - pyr_scale (float): Scale factor for pyramid
                - levels (int): Number of pyramid levels
                - winsize (int): Size of the window for averaging
                - iterations (int): Number of iterations at each pyramid level
                - poly_n (int): Size of the pixel neighborhood
                - poly_sigma (float): Standard deviation of the Gaussian used for polynomial expansion
                - flags (int): Operation flags
                - Additional keys for preprocessing (see preprocess_frame)

        Returns:
            np.array: Optical flow field.
        """
        raise NotImplementedError

    def _flow_arr_to_xyz(self):
        """
        Convert the optical flow array to XYZ coordinates and save to self.save_path/flow_xyz.npy.

        Returns:
            np.array: XYZ coordinates of the optical flow.
        """
        raise NotImplementedError

    def create_heatmap(
        self, figsize: tuple[int, int] = (10, 5), dark_mode: bool = False
    ) -> None:
        """
        Create a heatmap visualization of the optical flow and save it to self.save_path/heatmap.png.

        Args:
            figsize (tuple[int, int]): Size of the figure for the heatmap.
            dark_mode (bool): Whether to use dark mode for the visualization.
        """
        raise NotImplementedError

    def generate_kymograph(
        self, figsize: tuple[int, int] = (10, 5), dark_mode: bool = False, **kwargs
    ) -> None:
        """
        Generate a kymograph visualization of the optical flow and save it to self.save_path/kymograph.png.

        Args:
            figsize (tuple[int, int]): Size of the figure for the kymograph.
            dark_mode (bool): Whether to use dark mode for the visualization.
            kwargs: Additional parameters for kymograph generation, which may include:
                - line (np.ndarray): 2D array representing the kymograph data.
                - ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
                - figsize (tuple): Size of the figure in inches (width, height).
                - aspect (str): Aspect ratio of the plot. Default is 'auto'.
                - cmap (str): Colormap to use for the kymograph. Default is 'PRGn'.
                - origin (str): Origin of the plot. Default is 'upper'.
                - label (str): Label for the colorbar.
                - xlabel (str): Label for the x-axis.
                - ylabel (str): Label for the y-axis.
                - title (str): Title of the plot.
                - show (bool): Whether to display the plot immediately.
        """
        raise NotImplementedError

    def generate_sparse_flow_video(
        self,
        tracking_points: list[tuple[float, float]],
        figsize: tuple[int, int] = (10, 5),
        dark_mode: bool = False,
    ) -> None:
        """
        Generate a sparse optical flow video visualization and save it to self.save_path/sparse_flow.mp4.

        Args:
            tracking_points (list[tuple[float, float]]): List of tracking points for sparse flow visualization.
            figsize (tuple[int, int]): Size of the figure for the video.
            dark_mode (bool): Whether to use dark mode for the visualization.
        """
        raise NotImplementedError
