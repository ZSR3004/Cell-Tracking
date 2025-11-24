import click

import file_input_cli as fic
import optical_flow_cli as opt
import visualization_cli as v
import saving_cli as s
import questionary

import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.memory import MemoryManagement

OUTPUT_OPTIONS = [
    "Optical Flow",
    "Heatmap",
    "Kymograph",
    "Raw Data"
]

def get_video():
    """
    This function gets the path to the file

    Arguments: None

    Returns: user path
    """
    path = click.prompt('Enter your file path name', type=click.Path(
                                                                        exists=True,
                                                                        file_okay=True,
                                                                        dir_okay=False,
                                                                        resolve_path=True))
    
    return path

def is_created() -> bool:
    """
    Tracks whether they've setup their folder

    Args: None

    Returns: true if they say yes, false if they say no
    """
    answer = click.prompt("Have you initialized your Cell-Tracking folder? [y/n]",
                          type=click.Choice(['y', 'n'], case_sensitive=False))

    return answer.lower() == 'y'

def desired_outputs() -> tuple:
    """
    Asks user what outputs they want

    Args: None

    Returns: tuple of their answers
    """

    outputs = questionary.checkbox(
            "Select output items to generate:",
            choices=OUTPUT_OPTIONS
        ).ask()
    
    return outputs

def get_video_type() -> str:
    """
    Asks user what type of video they input

    Args: None

    Returns: string of their answer
    """

    my_video = click.prompt("Type in the type of video  you input (n for nuclei dyed, p for phase contrast)", type=str)

    return my_video

def main():
    #Checks to see if they've created the CellTracking folder yet. Relies on the user knowing this information.
    if not is_created():
        my_path = click.prompt("Type the folder you want to save your CellTracking folder to (type ~/folder_name or folder_name)", type=str)

        parent_dir = os.path.expanduser(my_path)

        # If they typed a relative path, make it relative to home
        if not os.path.isabs(parent_dir):
            parent_dir = os.path.join(os.path.expanduser("~"), parent_dir)

        cell_tracking_path = os.path.join(parent_dir, "CellTracking")
        my_folders = MemoryManagement(cell_tracking_path)
        my_folders.create_main_dir()
        os.chdir(cell_tracking_path)
        my_folders._write_default_yaml()

    else:
        my_path = click.prompt("Type the directory where your folder is saved (type ~/folder_name or folder_name)", type=str)

        parent_dir = os.path.expanduser(my_path)

        # If they typed a relative path, make it relative to home
        if not os.path.isabs(parent_dir):
            parent_dir = os.path.join(os.path.expanduser("~"), parent_dir)

        cell_tracking_path = os.path.join(parent_dir, "CellTracking")
        my_folders = MemoryManagement(cell_tracking_path)
        my_folders.create_main_dir()
        os.chdir(cell_tracking_path)


    full_path = get_video()
    tiff_name = os.path.basename(full_path)
    my_video = fic.init_tiff_class(full_path)
    #s.save_original_video_cli(tiff_name, os.getcwd())

    if not os.path.exists(os.getcwd() + "/" + tiff_name):
        my_folders.create_tiff_dir(tiff_name)
    else:
        i=1
        while os.path.exists(os.path.join(os.getcwd(), f"{tiff_name} ({i})")):
            i += 1
        my_folders.create_tiff_dir(f"{tiff_name} ({i})")

    #Creates a Tiff class instance and preprocesses based on the yaml config file.
    my_folders.read_yaml()
    preprocess_args = my_folders.config["preprocess_args"]
    fic.preprocess_tiff(my_video, **preprocess_args)

    #Get the outputs they want via questionary
    outputs = desired_outputs()

    #Ask user what kind of video they input, do optical flow calculation
    if "Optical Flow" in outputs:
        os.chdir(parent_dir + "/CellTracking/" + tiff_name + "/optical_flows")
        vid_type = get_video_type()
        
        if vid_type == 'n':
            flow_channel_0 = opt.calculate_nuclei_optical_flow(my_video.arr, 0)
            flow_channel_1 = opt.calculate_nuclei_optical_flow(my_video.arr, 1)
            answer = click.prompt("Do you want to calculate the combined flows of channels 1 and 2? [y/n]", 
                         type=click.Choice(['y', 'n'], case_sensitive=False))
            
            if answer.lower() == 'y':
                combined_flow = opt.calculate_combined_flow(my_video.arr)
                s.save_flow_cli(combined_flow)

            s.save_flow_cli(flow_channel_0)
            s.save_flow_cli(flow_channel_1)
            #call save_vector_video here?

        elif vid_type == 'p':
            raft_flow = opt.calculate_raft_optical_flow(my_video)
            s.save_flow_cli(raft_flow)
            #call save vector video here?
    
    os.chdir(parent_dir + "/CellTracking/" + tiff_name)

    #Generates the heatmaps, saves to heatmap folder
    if "Heatmap" in outputs:
        os.chdir(parent_dir + "/CellTracking/" + tiff_name + "/heatmaps")
        #v.show_heatmaps()
        #save the heatmaps, maybe nest this in optical flow if it requires the optical flow arrays

    os.chdir(parent_dir + "/CellTracking/" + tiff_name)
    #Generates the kymographs, save to kymograph folders
    if "Kymograph" in outputs:
        os.chdir(parent_dir + "/CellTracking/" + tiff_name + "/kymographs")
        #v.show_kymograph()
        #again, nest this if it requires the optical flow arrays

    os.chdir(parent_dir + "/CellTracking/" + tiff_name)

    #Saves the raw data to the raw data folders
    if "Raw Data" in outputs:
        os.chdir(parent_dir + "/CellTracking/" + tiff_name + "/raw_data")
        #save all the raw data (xyz files, arrays, etc, here)
    
    os.chdir(parent_dir + "/CellTracking/" + tiff_name)

if __name__ == "__main__":
    main()

