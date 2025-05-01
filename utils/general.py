import math
import torch
import os
from pathlib import Path
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv3 root directory


def make_divisible(x, divisor):
    """Adjusts `x` to be nearest and greater than or equal to value divisible by `divisor`."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def check_yaml(file, suffix=(".yaml", ".yml")):
    """Searches/downloads a YAML file and returns its path, ensuring it has a .yaml or .yml suffix."""
    return check_file(file, suffix)


def check_file(file, suffix=""):
    files = []
    for d in "data", "models", "utils":  # search directories
        files.extend(glob.glob(str(ROOT / d / "**" / file), recursive=True))  # find file
    assert len(files), f"File not found: {file}"  # assert file was found
    assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
    return files[0]  # return file
    