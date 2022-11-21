from os import getcwd, listdir
from os.path import join, isfile, isdir, exists, abspath, dirname, basename
from typing import Generator

import cv2
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from sys import path

path.append("~/Masters/CS/Big-Data-Research-Project/src/object_detection/yolov7_with_object_tracking")

from ..object_detection.yolov7_with_object_tracking.models.experimental import attempt_load


# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, \
#                 check_imshow, non_max_suppression, apply_classifier, \
#                 scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
#                 increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# from sort import *

def get_frame() -> Generator[str, None, None]: # https://stackoverflow.com/questions/42531143/how-to-type-hint-a-generator-in-python-3
    """Generator of all the frames in the current directory.
    """
    for file in listdir(getcwd()):
        if isfile(file) and file.endswith('.png'):
            yield file

def main() -> None:
    print("Starting the program")



if __name__ == "__main__":
    main()