import os
import yaml
from defaults import default_yaml_config

class MemoryManagement:
    def __init__(self, path: str) -> None:
        """
        Args: 
            path (str): The path where the directory will be created. In particular, 
                "path/cell-tracking/"

        Attributes:
            path (str): See above.
            yaml_path (str): The path of the YAML configuration file.
            config (dict): The dictionary representation of the YAML file.
        """
        self.path = path
        self.yaml_path = os.path.join(self.path, "config.yaml")
        self.config = None

    def _write_default_yaml(self) -> None:
        """
        Sets the default configuration YAML settings.
        """
        with open(self.yaml_path, "w") as y:
            yaml.dump(default_yaml_config, y)

    def create_main_dir(self) -> None:
        """
        Creates the main cell tracking directory where all files are stored. Does NOT 
        overwrite the directory if it exists already.
        """
        os.makedirs(self.path, exist_ok=True)
        try: 
                self._write_default_yaml()
        except FileExistsError:
            pass
        
    def read_yaml(self) -> None:
        """
        Reads the YAML file from the main path.
        """
        try:
            with open(self.yaml_path, "r") as y:
                self.config = yaml.safe_load(y)
        except:
            print("Failed to read configuration YAML file.")

    def create_tiff_dir(self, name: str) -> None:
        """
        Handles creations of subdirectories for each Tiff file.
        """
        tiff_dir_path = os.path.join(self.path, name)
        os.makedirs(tiff_dir_path)

        sub_dirs = [
                    "raw_data",
                    "optical_flows",
                    "heatmaps",
                    "kymographs"
                ]

        for path in sub_dirs:
            os.makedirs(os.path.join(tiff_dir_path, path))

