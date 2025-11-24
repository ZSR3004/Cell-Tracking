import cv2
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors


def vector_magnitude_heatmaps(flow, normalize=True):
    """
    Computes magnitude heatmaps from a flow array of shape (frames, height, width, 2).

    Args:
        flow (np.ndarray): Array of shape (frames, height, width, 2) with (dx, dy) vectors.
        apply_colormap (bool): If True, returns heatmaps with color (BGR, 3-channel).
        normalize (bool): If True, normalizes magnitudes to 0–255 range for visualization.

    Returns:
        heatmaps (np.ndarray): Array of shape (frames, height, width) or (frames, height, width, 3)
                               depending on apply_colormap.
    """
    magnitudes = np.linalg.norm(flow, axis=-1)

    if normalize:
        heatmaps = []
        for frame in magnitudes:
            norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            norm = norm.astype(np.uint8)
            heatmaps.append(norm)
        heatmaps = np.stack(heatmaps, axis=0)
    else:
        heatmaps = magnitudes
    return heatmaps

def save_heatmap_video(flow, output_path='heatmap_video.mp4', fps=10, normalize=True):
    """
    Saves a heatmap video (MP4) from a flow array using matplotlib.

    Parameters:
        flow (np.ndarray): Array of shape (frames, height, width, 2)
        output_path (str): Path to save the MP4 video
        fps (int): Frames per second of the output video
        normalize (bool): Whether to normalize magnitudes per frame
    """
    heatmaps = vector_magnitude_heatmaps(flow, normalize=normalize)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmaps[0], cmap='jet', animated=True)
    ax.axis('off')

    def update(frame_idx):
        im.set_array(heatmaps[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(heatmaps),
        interval=1000 / fps,
        blit=True
    )
    ani.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close(fig)
    