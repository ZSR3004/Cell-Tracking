import numpy as np
import tifffile as tiff
import datetime
import cv2
from scipy.ndimage import gaussian_laplace
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Tiff:
    """
    This is a class that imports TIFF file to program, converts TIFF to numpy array using TIFFFILE,
    and stores the video type
    """

    def __init__(self, path: str):
        """
        Initializes a TiffStack object by loading a TIFF file and extracting its frames.

        Args:
            path (str): Path to the TIFF file.

        Attributes:
            path (str): Path to the TIFF file.
            timestamp (str): Timestamp of when the TIFF file was loaded.
            arr (np.ndarray): 4D numpy array containing the image frames, shape is (n_frames, n_channels, height, width)
            Other metadata attributes as needed.

        Returns: 
            None
        """
        self.path = path
        self.timestamp = datetime.datetime.now()
        self.arr = tiff.imread(path)

    def isolate_channel(self, channel_idx: int) -> np.ndarray:
        """
        Isolates a specific channel from the TIFF stack.

        Args:
            channel_idx (int): Index of the channel to isolate (0-indexed).

        Returns:
            np.ndarray: Isolated channel as a 3D numpy array.
        """
        assert(channel_idx >= 0)
        assert(channel_idx < len(self.arr))
        return self.arr[:,channel_idx,:,:]

    def save_original_video(self, name: str, file_path: str, im, image_stack, fig, ax, **kwargs) -> None:
        """
        Saves a video of image frames using matplotlib.

        Args:
            name (str): Name of the video file to save.
            file_path (str): The path to save the video file to.
            im: Matplotlib image display object for the original frames.
            image_stack: Image stack of shape (T, H, W) or (T, H, W, 3) for RGB.
            fig: Matplotlib figure object for the plot.
            ax: Matplotlib axes object for the plot.
            **kwargs: Additional keyword arguments that include:
                - T (int): Total number of frames in the image stack.
                - fps (int): Frames per second for the video.

        Assumptions:
            'T' is greater than or equal to image_stack.shape[0].

        Returns:
            None: Just saves the video to the specified path.
        """
        T = kwargs.get('T', image_stack.shape[0])
        fps = kwargs.get('fps', 10)

        def update(frame):
            im.set_data(image_stack[frame])
            ax.set_title(f"Frame {frame}")

        ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(file_path, writer=writer)

    def show_image(self, image: np.ndarray, title='Image', figsize=(12, 8), save_path=None) -> None:
        """
        Displays or saves an image using matplotlib.

        Args:
            image (np.ndarray): Image to display.
            title (str): Title of the window.
            figsize (tuple): Figure size in inches (width, height).
            save_path (str, optional): If provided, saves the image to this path.

        Assumptions:
            The integers in the 'figsize' tuple are greater than 0.

        Returns:
            None
        """
        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

    def preprocess_frame(self, args: tuple[np.ndarray, dict]) -> np.ndarray:
        """
        Preprocesses a single frame with optional Gaussian/median blurs, normalization,
        and type conversion.

        Args:
            args (tuple): A tuple containing the frame and a dictionary of preprocessing parameters.
                - frame (np.ndarray): Input frame to preprocess.
                - kwargs (dict): Dictionary with preprocessing parameters:
                    - gauss (dict): {'ksize': (int, int), 'sigmaX': float}
                    - median (dict): {'ksize': int}
                    - normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}
                    - contrast (dict): {'alpha': float, 'beta': int}
                    - skip (list[str]): steps to skip (e.g., ['gauss', 'median'])

        Assumptions:
            The 'ksize' value in the gauss dictionary must be a tuple of two positive odd integers.
            The 'ksize' value in the median dictionary must be a positive odd integer.

        Returns:
            np.ndarray: Preprocessed image.
        """
        frame, kwargs = args
        skip = kwargs.get("skip", [])

        if "gauss" not in skip:
            gauss_cfg = kwargs.get("gauss", {})
            ksize = gauss_cfg.get("ksize", (5, 5))
            sigmaX = gauss_cfg.get("sigmaX", 1.5)
            frame = cv2.GaussianBlur(frame, ksize, sigmaX)

        if "median" not in skip:
            median_cfg = kwargs.get("median", {})
            ksize = median_cfg.get("ksize", 5)
            frame = cv2.medianBlur(frame, ksize)

        if "minmax" not in skip:
            normalize_cfg = kwargs.get("normalize", {})
            alpha = normalize_cfg.get("alpha", 0)
            beta = normalize_cfg.get("beta", 255)
            norm_type = normalize_cfg.get("norm_type", cv2.NORM_MINMAX)
            frame = cv2.normalize(frame, None, alpha, beta, norm_type)

        if "contrast" not in skip:
            contrast_cfg = kwargs.get("contrast", {})
            alpha = contrast_cfg.get("alpha", 1.0)  # Contrast factor
            beta = contrast_cfg.get("beta", 0)      # Brightness offset
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        return frame

    def preprocess_stack(self, arr: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocesses a stack of frames with optional Gaussian/median blurs, normalization,
        and type conversion.

        Args:
            arr (np.ndarray): Input stack of frames (shape: N x H x W).
            **kwargs: Dictionary with preprocessing parameters:
                - gauss (dict): {'ksize': (int, int), 'sigmaX': float}
                - median (dict): {'ksize': int}
                - normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}
                - contrast (dict): {'alpha': float, 'beta': int}
                - skip (list[str]): steps to skip (e.g., ['gauss', 'median'])

        Assumptions:
            arr is not empty.
            The 'ksize' value in the gauss dictionary must be a tuple of two positive odd integers.
            The 'ksize' value in the median dictionary must be a positive odd integer.

        Returns:
            np.ndarray: Preprocessed stack of frames.
        """
        frames = [(arr[i], kwargs) for i in range(arr.shape[0])]
        with Pool(cpu_count()) as pool:
            preprocessed_frames = pool.map(self.preprocess_frame, frames)
        return np.stack(preprocessed_frames, axis=0)