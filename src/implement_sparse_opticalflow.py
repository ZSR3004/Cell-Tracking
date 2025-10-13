import numpy as np

def implement_lucas_kande(args) -> np.ndarray:
    """
    Use Lucas-Kanade to generate numpy arrays of sparse optical.
    Take numpy array of orignal video
    Take pre-selected points
    Track said points, actual implementation of Lucas-Kanade across frames
    Output numpy array of (x, y) pairs for displacement

    """
    raise NotImplementedError


def save_optflow_video(flow, idx : int = 0, step : int = 20, 
                          scale : int = 500, color : str = 'blue', fps : int = 10, 
                          figsize : int | int = (12,8),
                          title : str = None, overlay : bool = False):
     """
        Saves a video visualizing the optical flow using the lucas_kande.
     """
     raise NotImplementedError