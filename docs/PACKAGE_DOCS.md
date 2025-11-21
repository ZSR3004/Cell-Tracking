# Cell-Tracking as a Package

One of the ways you can use our Cell-Tracking code is as an Python library. This will allow you to create custom scripts and generate highly customizable plots beyond
the basic kymograph and heatmaps that we offer. This file serves as a general guide on how to install the library and basic programming with it.

## Table of Contents

1. [Installation](#installation) \
   1.1 [Prerequisites](#prerequisites) \
   1.2 [Main Installation Steps](#main-installation-steps)

2. [Code Documentation](#code-documentation) \
   2.1 [The Tiff Class](#the-tiff-class) \
      2.1.1 [Attributes](#attributes) \
      2.1.2 [Methods](#methods)
    - [show_image](#show_image)
    - [preprocess_stack](#preprocess_stack)

   2.2 [Optical Flow](#optical-flow) \
      2.2.1 [calculate_optical_flow (Nuclei-Labeled)](#calculate_optical_flow-nuclei-labeled) \
      2.2.2 [calcOpticalFlowRAFT (Cytoplasm-Labeled and Phase Contrast)](#calcopticalflowraft-cytoplasm-labeled-and-phase-contrast)

3. [Complete Script Example](#complete-example-script)

4. [What's Next?](#whats-next)


## Installation

### Prerequisites

You should have the following installed on your laptop already.

1. Python
2. A text editor
3. Pip

### Main Installation Steps

We highly recommend using a virtual environment. As a quick reminder, you can create and activate one by going to your terminal and typing the following.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After that, you can install our Cell-Tracking package by using the following command.

```bash
pip install git+https://github.com/ZSR3004/Cell-Tracking.git
```

That's it! You're up and running.

## Code Documentation

`cell_tracking` currently exposes three primary things. One class and two functions. Below, we've documented how to use each of these including the relevant methods 
for the class. Once you've installed our package using `pip`, import our package using the following line.

```python
import cell_tracking as ct
```

### The Tiff Class

The `Tiff Class` will be your core way of manipulating your TIFF file in Python. To create an instance, merely invoke the class with the path to your TIFF file. For
instance, if your TIFF file is on the path `~/lab/file1.tiff`, then run

```python
tiff = ct.Tiff("~/lab/file1.tiff")
```

**Attributes:**

This creates an instance off the `Tiff Class` which holds the following attributes.


| Attribute        | Type         | Description                                                                                        |
| ---------------- | ------------ | -------------------------------------------------------------------------------------------------- |
| `self.path`      | `str`        | Path of the loaded TIFF file.                                                                      |
| `self.timestamp` | `datetime`   | Time the file was loaded.                                                                          |
| `self.arr`       | `np.ndarray` | The loaded TIFF data as a NumPy array. Shape is `(n_frames, n_channels, height, width)`.       |


There are only two methods from this function that you need to use.

#### show_image

`show_image` will display or save a single frame of your Tiff file. The image will be loaded in gray scale.
The primary purpose of this function is for you to visually verify that the Tiff file was loaded in 
correctly. If you want to see a full video of the Tiff file, we recommend using another tool such as FIJI.

##### **Parameters:**

| Name        | Type              | Description                                                              |
| ----------- | ----------------- | ------------------------------------------------------------------------ |
| `image`     | `np.ndarray`      | The image to display or save. Make sure to slice it to one frame.       |
| `title`     | `str`             | Title displayed above the image. Default is `"Image"`.                   |
| `figsize`   | `tuple[int, int]` | Size of the figure in inches `(width, height)`.                          |
| `save_path` | `str or None`     | If provided, the image is saved to this path instead of being displayed. |

##### **Returns**
This function returns nothing. It either saves the image of the frame or displays it.

##### **Example:**

```python
# Display the first frame of the TIFF stack
frame = tiff.arr[0, 0]      # if your Tiff class instance is called tiff
tiff.show_image(frame, title="First Frame")

# Save the image instead of displaying
tiff.show_image(frame, save_path="output.png")
```

#### preprocess_stack

`preprocess_stack` will apply various blurs to the Tiff stack to make it easier to analyze. Every Tiff stack is different, so we recommend creating a dictionary to
store the parameters and experimenting with them that way. You can see this setup in the example below.

This function will apply the following blurs by default:
* Gaussian blur
* Median blur
* Min–max normalization
* Contrast adjustment

##### **Parameters:**

| Name      | Type          | Description                       |
|---------- | ------------  | --------------------------------- | 
| `arr`       | `np.ndarray`    | Input stack of frames (shape: N x H x W). |
| `**kwargs`  | `dict`    | Key words arguments               | 


Each of the following can be passed as keyword arguments:


| Key         | Type        | Description                           |
| ----------- | ----------- | ------------------------------------- |
| `gauss`     | `dict`      | Gaussian blur configuration.          |
| `median`    | `dict`      | Median blur configuration.            |
| `normalize` | `dict`      | Intensity normalization settings.     |
| `contrast`  | `dict`      | Contrast and brightness adjustment.   |
| `skip`      | `list[str]` | Names of preprocessing steps to skip. |

###### `gauss` dictionary

| Key      | Type              | Description                                                  |
| -------- | ----------------- | ------------------------------------------------------------ |
| `ksize`  | `tuple[int, int]` | Kernel size, must be positive odd integers (e.g., `(5, 5)`). |
| `sigmaX` | `float`           | Gaussian kernel standard deviation.                          |

###### `median` dictionary

| Key     | Type  | Description                                                  |
| ------- | ----- | ------------------------------------------------------------ |
| `ksize` | `int` | Kernel size for median blur. Must be a positive odd integer. |

###### `normalize` dictionary

| Key         | Type  | Description                                             |
| ----------- | ----- | ------------------------------------------------------- |
| `alpha`     | `int` | Minimum output intensity.                               |
| `beta`      | `int` | Maximum output intensity.                               |
| `norm_type` | `int` | OpenCV normalization type (default: `cv2.NORM_MINMAX`). |

###### `contrast` dictionary

| Key     | Type    | Description          |
| ------- | ------- | -------------------- |
| `alpha` | `float` | Contrast multiplier. |
| `beta`  | `int`   | Brightness offset.   |

###### `skip` list

Takes any of the values "gauss", "median", "normalize", or "contrast" and does not apply that blur during preprocessing.

**Return:**

| Type         | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `np.ndarray` | A new stack of preprocessed frames with the same shape as the input `(N, H, W)`. |

**Example:**

```python
from cell_tracking import Tiff

tiff_stack = Tiff("example_stack.tif")

# Define preprocessing parameters
params = {
    "gauss": {"ksize": (5, 5), "sigmaX": 1.2},
    "median": {"ksize": 3},
    "normalize": {"alpha": 0, "beta": 255, "norm_type": None},  # OpenCV NORM_MINMAX by default
    "contrast": {"alpha": 1.5, "beta": 10},
    "skip": [median]  # skipped the median blur
}

# Apply preprocessing
processed_stack = tiff_stack.preprocess_stack(tiff_stack.arr[:, 0, :, :], **params)
```

## Optical Flow 

Optical flow is the primary data analysis objective for this package. There are two types of optical flow algorithms provided. The first uses the Farneback algorithm
as a backend. The second uses the RAFT model.

### calculate_optical_flow (Nuclei-Labeled)

This calculates the optical flow of a tiff stack using the Farneback algorithm. We assume that channel 1 shows two cell sheets moving, channel 2 shows one of them moving,
and channel 2 shows only the other side moving.
It applies the algorithm to the second and third channels (the left and right), then
creates a "pseudo" first channel by adding the two channels together (cells on one channel only exist in the black space of the other channel).

The Farneback algorithm is the one used by the OpenCV project and the documentation can be found [here](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html).

#### **Parameters:**

| Name       | Type         | Description                                                                            |
| ---------- | ------------ | -------------------------------------------------------------------------------------- |
| `arr`      | `np.ndarray` | The TIFF stack array to process. Shape should be `(N, C, H, W)`.                       |
| `**kwargs` | `dict`       | Optional Farneback parameters passed to `optical_flow`:                                |
|            | `pyr_scale`  | Scale factor for the image pyramid. Default is `0.5`.                                  |
|            | `levels`     | Number of pyramid levels. Default is `3`.                                              |
|            | `winsize`    | Window size for averaging. Default is `15`.                                            |
|            | `iterations` | Number of iterations per pyramid level. Default is `3`.                                |
|            | `poly_n`     | Size of the pixel neighborhood for polynomial expansion. Default is `5`.               |
|            | `poly_sigma` | Standard deviation of the Gaussian used in the polynomial expansion. Default is `1.2`. |
|            | `flags`      | Operation flags. Default is `0`.                                                       |

#### **Return:**

| Type         | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `np.ndarray` | Combined flow vectors of shape `(n_frames - 1, n_channels, height, width, 2)` representing motion between consecutive frames. |


#### **Example:**

```python
from cell_tracking import Tiff, calculate_optical_flow

tiff_stack = Tiff("example_stack.tif")

# Compute optical flow with default Farneback parameters
flow_default = calculate_optical_flow(tiff_stack.arr)

# Compute optical flow with custom Farneback parameters
flow_custom = calculate_optical_flow(
    tiff_stack.arr,
    pyr_scale=0.7,
    levels=4,
    winsize=21,
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=0
)
```

### calcOpticalFlowRAFT (Cytoplasm-Labeled and Phase Contrast)

This calculates the optical flow of a tiff stack using the RAFT algorithm. The backend for this function is a little more complicated than the Farneback-backed one,
so to simplify the process, instead of passing an array as a parameter, you'll pass a Tiff class instance. This function will only return the optical flow for the
first, phase-contrast, channel rather than all three. This is a result of how the RAFT algorithm operates.

To learn more about the RAFT algorithm, you can visit the original paper [here](https://arxiv.org/abs/2003.12039) and the `Pytorch`  implementation (used in the 
backend for this function)
[here](https://docs.pytorch.org/vision/main/models/raft.html).

##### **Parameters:**

| Name            | Type             | Description                                                                                                           |
| --------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------- |
| `tiff_file`     | `tiff.Tiff`      | The TIFF file representing the video to be processed. Contains an array of shape `[frames, channels, height, width]`. |
| `model_size`    | `ModelSize`      | Which RAFT variant to use. Options are `SMALL` or `LARGE`. Default is `SMALL`.                                        |
| `model_weights` | `dict` or `None` | A loaded state_dict for the model, or `None` to use default pretrained weights.                                       |
| `gpu_flag`      | `bool`           | If `True`, use CUDA when available; otherwise use CPU. Default is `False`.                                            |
| `**kwargs`      | `dict`           | Additional keyword arguments. See [preprocess_stack](#preprocess_stack)                                                               |




##### **Return:**

| Type         | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| `np.ndarray` | Flow vectors of shape `(n_frames - 1, height, width, 2)` representing motion between consecutive frames. |


##### **Example:**

```python
from cell_tracking import Tiff, calcOpticalFlowRAFT, ModelSize

tiff_file = Tiff("example_stack.tif")

# Compute RAFT optical flow with default SMALL model and default preprocessing
flow_default = calcOpticalFlowRAFT(tiff_file)

# Compute RAFT optical flow with a LARGE model, GPU, and custom preprocessing
flow_custom = calcOpticalFlowRAFT(
    tiff_file,
    model_size=ModelSize.LARGE,
    gpu_flag=True,
    gauss={"ksize": (5, 5), "sigmaX": 1.2},
    median={"ksize": 3},
    normalize={"alpha": 0, "beta": 255},
    contrast={"alpha": 1.5, "beta": 10},
    skip=[]
)
```

## Complete Example Script

```python
from cell_tracking import Tiff, calculate_optical_flow, calcOpticalFlowRAFT, ModelSize

# Step 1: Load your TIFF file
tiff_file = Tiff("example_stack.tif")

# Step 2: Inspect the first frame
first_frame = tiff_file.arr[0, 0]  # first frame of first channel
tiff_file.show_image(first_frame, title="First Frame")

# Step 3: Preprocess the TIFF stack
preprocess_params = {
    "gauss": {"ksize": (5, 5), "sigmaX": 1.2},
    "median": {"ksize": 3},
    "normalize": {"alpha": 0, "beta": 255, "norm_type": None},  # default NORM_MINMAX
    "contrast": {"alpha": 1.5, "beta": 10},
    "skip": []  # list of steps to skip, e.g., ["median"]
}

# Preprocess a single channels
processed_stack1 = tiff_file.preprocess_stack(tiff_file.arr[:, 0, :, :], **preprocess_params)
processed_stack2 = tiff_file.preprocess_stack(tiff_file.arr[:, 1, :, :], **preprocess_params)
processed_stack3 = tiff_file.preprocess_stack(tiff_file.arr[:, 2, :, :], **preprocess_params)
processed_stacks = np.stack((processed_stack1, processed_stack2, processed_stack3), axis=0) # make them into one big array again

# Step 4: Compute optical flow using Farneback (nuclei-labeled example)
flow_farneback = calculate_optical_flow(
    processed_stacks,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# Step 5: Compute optical flow using RAFT (cytoplasm-labeled / phase contrast example)
flow_raft = calcOpticalFlowRAFT(
    tiff_file,
    model_size=ModelSize.SMALL,
    gpu_flag=False,
    **preprocess_params  # optional preprocessing applied before RAFT; this is actually preprocessing the image twice
)

# Step 6: Inspect results

# Show first frame of Farneback optical flow (visualization example)
import matplotlib.pyplot as plt
plt.imshow(flow_farneback[0, 0, :, :, 0], cmap='viridis')  # x-direction flow
plt.title("Farneback Optical Flow - First Frame (X)")
plt.colorbar()
plt.show()

# Show first frame of RAFT optical flow
plt.imshow(flow_raft[0, :, :, 0], cmap='viridis')  # x-direction flow
plt.title("RAFT Optical Flow - First Frame (X)")
plt.colorbar()
plt.show()
```

## What's Next?
We plan to implement features that allow you to make heatmap and kymograph visualizations of the optical flow. We have not implemented these features yet.

<!-- ## The Tiff Class -->
<!-- The Tiff Class is initialized by creating an instance of it with the desired Tiff file. You'll need to provide the full path nameFor example: -->
<!---->
<!-- ```bash -->
<!-- my_file = tiff.Tiff("/Users/jamiesloves/Downloads/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED (1).tif") -->
<!-- ``` -->
<!-- Then, there are a number of methods that can be called: -->
<!-- - isolate_channel(self, channel_idx) takes in the channel index that you want to isolate and returns it as a 3D numpy array. -->
<!--     - ```bash -->
<!--         channel_1_arr = my_file.isolate_channel(1) -->
<!--       ``` -->
<!-- - show_image(self, image, title, figsize=(12,8), save_path=None) takes in an image (i.e. a specific frame from a certain channel) and makes a plot using matplotlib that can be saved to a specific path if it is provided. -->
<!--     - ```bash -->
<!--         my_file.show_image(my_file.arr[0,2,:,:], "Oth_frame_2nd_channel") -->
<!--       ``` -->
<!---->
<!-- - preprocess_stack(arr, **kwargs) takes in a stack of frames (i.e. all the frames in a specific channel) and a dictionary containing a set of preprocessing parameters, including: gauss (dict): {'ksize': (int, int), 'sigmaX': float}, median (dict): {'ksize': int}, normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}, contrast (dict): {'alpha': float, 'beta': int}, skip (list[str]): steps to skip (e.g., ['gauss', 'median']) -->
<!--     - ```bash -->
<!--         my_processed_file = my_file.preprocess_stack(my_file.arr[:,1,:,:], pre_process_params) -->
<!--       ``` -->
<!-- ## Optical Flow -->
<!---->
<!-- ### Nuclei-Labeled Cells -->
<!-- A Tiff file containing nuclei-labeled cells will call the optical flow function that takes in that type of file. You will call calculate_optical_flow(arr : tiff.Tiff, default=False) on your Tiff file, which will output the optical flow between the first and second channels of the file. -->
<!-- - ```bash -->
<!--     my_flow_arr = calculate_optical_flow(my_file.arr) -->
<!--   ``` -->
<!-- ### Cytoplasm-Labeled Cells and Phase Contrast -->
<!-- A Tiff file containing cytoplasm-labeled cells and phase contrast will call the optical flow function that takes in that type of file. You will call calcOpticalFlowRAFT(arr: tiff.Tiff model_size: ModelSize = ModelSize.SMALL, -->
<!-- model_weights: dict | None = None, -->
<!-- gpu_flag: bool = False, -->
<!-- **kwargs) on your Tiff file, which will output the optical flow between the first and second channels of the file. Most of the time, you will only need to specify the array, model size, and the **kwargs will be available from preprocess_stack. -->
<!-- - ```bash -->
<!--     my_flow_arr = calcOpticalFlowRAFT(my_file.arr, ModelSize.SMALL, pre_processing_params) -->
<!--   ``` -->

