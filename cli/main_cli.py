import click

import file_input_cli as fic
import optical_flow_cli as opt
import visualization_cli as v
import saving_cli as s
import questionary
import matplotlib.pyplot as plt

import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cell_tracking.memory import MemoryManager

OUTPUT_OPTIONS = [
    "Optical Flow Data",
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
        my_path = click.prompt("Type the folder you want to save your Cell-Tracking folder to (type ~/folder_name or folder_name)", type=str)

        parent_dir = os.path.expanduser(my_path)

        # If they typed a relative path, make it relative to home
        if not os.path.isabs(parent_dir):
            parent_dir = os.path.join(os.path.expanduser("~"), parent_dir)

        cell_tracking_path = os.path.join(parent_dir, "Cell-Tracking")
        my_folders = MemoryManager(cell_tracking_path)
        my_folders.create_main_dir()
        os.chdir(cell_tracking_path)
        my_folders._write_default_yaml()

    else:
        my_path = click.prompt("Type the directory where your folder is saved (type ~/folder_name or folder_name)", type=str)

        parent_dir = os.path.expanduser(my_path)

        # If they typed a relative path, make it relative to home
        if not os.path.isabs(parent_dir):
            parent_dir = os.path.join(os.path.expanduser("~"), parent_dir)

        cell_tracking_path = os.path.join(parent_dir, "Cell-Tracking")
        my_folders = MemoryManager(cell_tracking_path)
        my_folders.create_main_dir()
        os.chdir(cell_tracking_path)


    full_path = get_video()
    tiff_name = os.path.basename(full_path)
 
    my_video = fic.init_tiff_class(full_path)

    if not os.path.exists(os.getcwd() + "/" + tiff_name):
        my_folders.create_tiff_dir(tiff_name)
    else:
        i=1
        while os.path.exists(os.path.join(os.getcwd(), f"{tiff_name} ({i})")):
            i += 1
        new_name = f"{tiff_name} ({i})"
        my_folders.create_tiff_dir(new_name)
        tiff_name = new_name

    #Creates a Tiff class instance and preprocesses based on the yaml config file.
    my_folders.read_yaml()
    preprocess_args = my_folders.config["preprocess_args"]
    fic.preprocess_tiff(my_video, **preprocess_args)

    #Get the outputs they want via questionary
    outputs = desired_outputs()

    #Ask user what kind of video they input, do optical flow calculation
    vid_type = get_video_type()
        
    if vid_type == 'n':
        combined_flow = opt.calculate_combined_flow(my_video.arr)

        #Change to Tiff specific directory
        os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name)

        s.save_original_video_cli("Original_Video_Combined", full_path, 0)

        answer = click.prompt("Do you want to calculate the isolated flows of channels 1 and 2? [y/n]", 
                         type=click.Choice(['y', 'n'], case_sensitive=False))
            
        if answer.lower() == 'y':
            flow_channel_1 = opt.calculate_nuclei_optical_flow(my_video.arr, 1)
            flow_channel_2 = opt.calculate_nuclei_optical_flow(my_video.arr, 2)

            s.save_original_video_cli("Original_Video_Left", full_path, 1)
            s.save_original_video_cli("Original_Video_Right", full_path, 2)

            #all the procedures for isolated flow
            if "Optical Flow Data" in outputs:
                #Change to opt flow directory, save isolated flow data
                os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/optical_flows")
                s.save_flow_cli("Channel_1", flow_channel_1, os.getcwd())
                s.save_flow_cli("Channel_2", flow_channel_2, os.getcwd())

            if "Raw Data" in outputs:
                #Change to raw data  directory, save isolated flow data
                os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/raw_data")
                s.save_arr("tiff_array_1", my_video, os.getcwd)
                s.save_arr("tiff_array_2", my_video, os.getcwd)

            if "Heatmap" in outputs:
                #Change to heatmap directory, save isolated flow data
                os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/heatmaps")
                v.save_heatmap_video_cli(flow_channel_1, os.path.join(os.getcwd(), "/heatmap_channel_1"))
                v.save_heatmap_video_cli(flow_channel_2, os.path.join(os.getcwd(),"/heatmap_channel_2"))
                
            if "Kymograph" in outputs:
                #Change to kymograph directory, save isolated flow data
                os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/kymographs")
                v.plot_basic_kymo_cli(flow_channel_1, os.path.join(os.getcwd(), "/kymo_channel_1"))
                v.plot_basic_kymo_cli(flow_channel_2, os.path.join(os.getcwd(), "/kymo_channel_2"))


        #all the procedures for combined flow
        if "Optical Flow Data" in outputs:
            #Change to opt flow directory, save combined flow data
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/optical_flows")
            s.save_flow_cli("Combined_Flow", combined_flow, os.getcwd())

            
        if "Raw Data" in outputs:
            #Change to raw data directory, save raw Tiff array
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/raw_data")
            s.save_arr("tiff_array", my_video, os.getcwd())

        if "Heatmap" in outputs:
            #Change to heatmap directory, save combined heatmap video
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/heatmaps")
            v.save_heatmap_video_cli(combined_flow, os.path.join(os.getcwd(), "/heatmap_combined_flow"))
                
        if "Kymograph" in outputs:
            #Change to kymograph directory, save combined kymograph
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/kymographs")
            v.plot_basic_kymo_cli(combined_flow, os.path.join(os.getcwd(), "/kymo_combined_flow"))



    elif vid_type == 'p':
        raft_flow = opt.calculate_raft_optical_flow(my_video)

        os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name)
        s.save_original_video_cli("Original_Video_Combined", full_path, 0)
        s.save_original_video_cli("Original_Video_Left", full_path, 1)
        s.save_original_video_cli("Original_Video_Right", full_path, 2)

        if "Optical Flow Data" in outputs:
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/optical_flows")
            s.save_flow_cli("Flow", raft_flow, os.getcwd())
        
        if "Raw Data" in outputs:
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/raw_data")
            s.save_arr("tiff_array", my_video, os.getcwd())

        if "Heatmap" in outputs:
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/heatmaps")
            v.save_heatmap_video_cli(raft_flow, os.path.join(os.getcwd(), "/heatmap_phase_contrast"))
        
        if "Kymograph" in outputs:
            os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name + "/kymographs")
            v.plot_basic_kymo_cli(raft_flow, os.path.join(os.getcwd(), "/kymo_phase_contrast"))

    os.chdir(parent_dir + "/Cell-Tracking/" + tiff_name)



if __name__ == "__main__":
    main()

