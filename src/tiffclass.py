import numpy as np


class Tiff:
    """
    This is a class that imports TIFF file to program, converts TIFF to numpy array using TIFFFILE,
    and stores the video type
    """
    def __init__ (self, args):
        """
        Initializes a TiffStack object by loading a TIFF file and extracting its frames.
        Args:
            path (str): Path to the TIFF file.
            n_channels (int): Number of channels in the TIFF stack. Default is 3.
            dtype (np.dtype): Data type of the image frames. Default is np.uint16.
        
        Attributes:
            path (str): Path to the TIFF file.
            timestamp (str): Timestamp of when the TIFF file was loaded.
            tags (list): List of tags for each frame in the TIFF stack.
            arr (np.ndarray): 4D numpy array containing the image frames, shape is (n_frames, n_channels, height, width
            Other metadata attributes as needed.
        """
        raise NotImplementedError
    
    def import_vid (self, args):
        """
        Imports the TIFF file to the program

        Args:
        path = string that is a video path

        Returns:
        The opened file

        Preconditions:
        The path is a valid path to a TIFF video file
        """
        raise NotImplementedError
    
    def _get_name(self) -> str:
        """
        Generates a name for the TiffStack based on the file name.
        
        Returns:
            str: Name of the TiffStack.
        """
        raise NotImplementedError
    
    def isolate_channel(self, args):
        """
        Isolates a specific channel from the TIFF stack.

        Args:
            channel_idx (int): Index of the channel to isolate (0-indexed).

        Returns:
            np.ndarray: Isolated channel as a 3D numpy array.
        """
        raise NotImplementedError
    
    def convert_to_numpy (self, args):
        """
        Converts the TIFF file to a numpy array using TIFFFILE

        Args:
        The channel you want to convert to a numpy

        Returns:
        numpy array

        Preconditions:
        Will have to call import_vid in this function
        """

        raise NotImplementedError

    def save_orginal_video(self, args):
        """
        Saves a video of the original image frames from the TIFF stack.

        Args:
            idx (int): Index of the channel to visualize. Default is 0.
            figsize (tuple): Figure size in inches (width, height). Default is (12, 8).

        Returns:
            None
        """
        raise NotImplementedError
    
    def show_image(image: np.array, title='Image', figsize=(12, 8), save_path=None):
        """
        Displays or saves an image using matplotlib.

        Args:
            image (np.ndarray): Image to display.
            title (str): Title of the window.
            figsize (tuple): Figure size in inches (width, height).
            save_path (str, optional): If provided, saves the image to this path.

        Returns:
            None
        """
        raise NotImplementedError
    