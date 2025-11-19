import click

import file_input_cli as fic
import optical_flow_cli as opt
import visualization_cli as v

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


def main():
    my_args = MemoryManagement()
    path = get_video()
    my_video = fic.init_tiff_class(path)
    #preprocess = get_preprocessing_params()
    #fic.preprocess_tiff(my_video, **preprocess)



if __name__ == "__main__":
    main()

