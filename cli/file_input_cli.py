import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import tiffclass as tiff
import numpy as np


def init_tiff_class(path: str) -> tiff.Tiff:
    raise NotImplementedError


def preprocess_tiff(tiff: tiff.Tiff) -> np.ndarray:
    raise NotImplementedError
