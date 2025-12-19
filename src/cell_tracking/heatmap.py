import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FuncAnimation, FFMpegWriter


def convert_stack_to_polar(frame_stack: np.ndarray) -> np.ndarray:
    """
    Converts a channel of an array-representation of a TIFF image
    into its polar coordinate equivalent.

    Args:
        frame_stack (np.ndarray): The numpy array representation
            of the TIFF file. Shape is (f-1, h, w, 2), where 2 is (dx, dy).

    Returns:
        (np.ndarray): The same representation but in polar coordinates. Shape is (f-1, h, w, 2), where 2 is (r, theta).
    """
    x = frame_stack[..., 0]
    y = frame_stack[..., 1]

    r = np.sqrt(x**2 + y**2)
    max_f = np.max(np.abs(frame_stack))
    r_norm = r / (np.sqrt(2) * max_f)

    theta = np.arctan2(y, x)

    return np.stack([r_norm, theta], axis=-1)


def create_color_wheel(size: int = 200):
    """
    Creates a circular color wheel where center is white and edges are
    saturated colors.

    Args:
        size (int): Radius of the color wheel in pixels.

    Returns:
        (): RGB image array of the color wheel
    """
    y, x = np.ogrid[-1 : 1 : size * 1j, -1 : 1 : size * 1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    hue = (theta + np.pi) / (2 * np.pi)
    saturation = np.clip(r, 0, 1)
    value = np.ones_like(r)

    mask = r <= 1
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    rgb[~mask] = 1

    return rgb, mask


def polar_to_heatmap(polar_frame: np.ndarray) -> np.ndarray:
    """
    Converts polar coordinates to RGB heatmap.

    Args:
        polar_frame: Array of shape (height, width, 2), where last dim is (r, theta).

    Returns:
        RGB image array of shape (height, width, 3) with values in [0, 1]
        - r=0 appears white (no saturation)
        - r=1 appears as bright saturated color based on theta
    """
    r = polar_frame[:, :, 0]
    theta = polar_frame[:, :, 1]

    hue = (theta + np.pi) / (2 * np.pi)
    saturation = r
    value = np.ones_like(r)
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    return rgb


def plot_heatmap(arr: np.ndarray, title: str, output_path: str, fps: int = 20) -> None:
    """
    Creates and saves a vector-magnitude heatmap to output_path.

    Args:
        arr (np.ndarray): The array to create a heatmap out of. This
            should be shape (f-1, c, h, w, 2), where 2 is (dx, dy)
            which is in cartesian coordinates.
        title (str): Title for the heatmap.
        output_path (str): The path for the file to be saved to.
        fps (int): Fps of the output heatmap video.

    Returns:
        None.
    """
    polar_arr = convert_stack_to_polar(arr[:, 0])
    num_frames = polar_arr.shape[0]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    ax_main = fig.add_subplot(gs[0])
    ax_main.set_xlabel("Width")
    ax_main.set_ylabel("Height")

    rgb_image = polar_to_heatmap(polar_arr[0])
    im = ax_main.imshow(rgb_image, origin="lower")
    title_text = ax_main.set_title(f"{title} - Frame 0/{num_frames-1}")

    ax_wheel = fig.add_subplot(gs[1])
    color_wheel, mask = create_color_wheel(300)
    ax_wheel.imshow(color_wheel, extent=[-1, 1, -1, 1])
    ax_wheel.set_aspect("equal")

    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle_deg in angles_deg:
        angle_rad = np.radians(angle_deg)
        x = 1.15 * np.cos(angle_rad)
        y = 1.15 * np.sin(angle_rad)
        ax_wheel.text(x, y, f"{angle_deg}°", ha="center", va="center", fontsize=10)

    ax_wheel.set_xlim(-1.4, 1.4)
    ax_wheel.set_ylim(-1.4, 1.4)
    ax_wheel.axis("off")

    plt.tight_layout()

    def update(frame):
        rgb_image = polar_to_heatmap(polar_arr[frame])
        im.set_array(rgb_image)
        title_text.set_text(f"{title} - Frame {frame}/{num_frames-1}")
        return [im, title_text]

    print(f"Creating animation with {num_frames} frames...")
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    writer = FFMpegWriter(fps=fps, metadata=dict(artist="Matplotlib"), bitrate=1800)
    anim.save(output_path, writer=writer)
