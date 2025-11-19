import click

import file_input_cli as fic
import optical_flow_cli as opt
import visualization_cli as v

import yaml
import os
import sys

# Add the project root (Cell-Tracking) to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.memory import MemoryManagement

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

def main():
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
    if not os.path.exists(os.getcwd() + "/" + tiff_name):
        my_folders.create_tiff_dir(tiff_name)
    else:
        i=1
        while os.path.exists(os.path.join(os.getcwd(), f"{tiff_name} ({i})")):
            i += 1
        my_folders.create_tiff_dir(f"{tiff_name} ({i})")
    print("MemoryManagement path:", my_folders.path)
    print("Looking for YAML at:", my_folders.yaml_path)
    print("Exists?", os.path.exists(my_folders.yaml_path))
    my_folders.read_yaml()
    preprocess_args = my_folders.config["preprocess_args"]
    fic.preprocess_tiff(my_video, **preprocess_args)



if __name__ == "__main__":
    main()

