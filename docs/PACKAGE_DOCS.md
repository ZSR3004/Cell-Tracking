# Cell-Tracking as a Package

One of the ways you can use our Cell-Tracking code is as an Python library. This will allow you to create custom scripts and generate highly customizable plots beyond
the basic kymograph and heatmaps that we offer. This file serves as a general guide on how to install the library and basic programming with it.

## Table of Contents

## Installation

### Prerequisites

You should have the following installed on your laptop already.

1. Python
2. A text editor

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

## The Tiff Class
The Tiff Class is initialized by creating an instance of it with the desired Tiff file. You'll need to provide the full path nameFor example:

```bash
my_file = tiff.Tiff("/Users/jamiesloves/Downloads/20220929_MCF_Rab5a_WH_heterotypic_s1_SCALED (1).tif")
```
Then, there are a number of methods that can be called:
- isolate_channel(self, channel_idx) takes in the channel index that you want to isolate and returns it as a 3D numpy array.
    - ```bash
        channel_1_arr = my_file.isolate_channel(1)
      ```
- show_image(self, image, title, figsize=(12,8), save_path=None) takes in an image (i.e. a specific frame from a certain channel) and makes a plot using matplotlib that can be saved to a specific path if it is provided.
    - ```bash
        my_file.show_image(my_file.arr[0,2,:,:], "Oth_frame_2nd_channel")
      ```

- preprocess_stack(arr, **kwargs) takes in a stack of frames (i.e. all the frames in a specific channel) and a dictionary containing a set of preprocessing parameters, including: gauss (dict): {'ksize': (int, int), 'sigmaX': float}, median (dict): {'ksize': int}, normalize (dict): {'alpha': int, 'beta': int, 'norm_type': int}, contrast (dict): {'alpha': float, 'beta': int}, skip (list[str]): steps to skip (e.g., ['gauss', 'median'])
    - ```bash
        my_processed_file = my_file.preprocess_stack(my_file.arr[:,1,:,:], pre_process_params)
      ```
## Optical Flow

### Nuclei-Labeled Cells
A Tiff file containing nuclei-labeled cells will call the optical flow function that takes in that type of file. You will call calculate_optical_flow(arr : tiff.Tiff, default=False) on your Tiff file, which will output the optical flow between the first and second channels of the file.
- ```bash
    my_flow_arr = calculate_optical_flow(my_file.arr)
  ```
### Cytoplasm-Labeled Cells and Phase Contrast
A Tiff file containing cytoplasm-labeled cells and phase contrast will call the optical flow function that takes in that type of file. You will call calcOpticalFlowRAFT(arr: tiff.Tiff model_size: ModelSize = ModelSize.SMALL,
model_weights: dict | None = None,
gpu_flag: bool = False,
**kwargs) on your Tiff file, which will output the optical flow between the first and second channels of the file. Most of the time, you will only need to specify the array, model size, and the **kwargs will be available from preprocess_stack.
- ```bash
    my_flow_arr = calcOpticalFlowRAFT(my_file.arr, ModelSize.SMALL, pre_processing_params)
  ```
## What's Next?
We plan to implement features that allow you to make heatmap and kymograph visualizations of the optical flow. We have not implemented these features yet.
