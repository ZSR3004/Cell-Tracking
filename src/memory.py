import os, sys
import yaml

class MemoryManagement:
    def __init__(self, path: str) -> None:
        """
        Args: 
            path (str): The path where the directory will be created. In particular, 
                "path/cell-tracking/"
        """
        self.path = path

    def create_main_dir(self) -> None:
        """
        Creates the main cell tracking directory where all files are stored. Does NOT 
        overwrite the directory if it exists already.
        """
        raise NotImplementedError

    def read_yaml(self) -> None:
        """
        Reads the YAML file from the main path.
        """
        raise NotImplementedError

    def create_tiff_dir(self) -> None:
        """
        Handles creations of subdirectories for each Tiff file.
        """
        raise NotImplementedError

