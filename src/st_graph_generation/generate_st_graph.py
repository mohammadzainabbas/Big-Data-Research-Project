from os import getcwd, listdir
from os.path import join, isfile, isdir, exists, abspath, dirname, basename
from typing import Generator, Union, List, Tuple, Dict, Any

import cv2
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from sys import path

from stg_utils import print_log, print_error

path.append("/Users/mohammadzainabbas/Masters/CS/Big-Data-Research-Project/src/object_detection/yolov7_with_object_tracking")

from models.experimental import attempt_load, Ensemble
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *

def load_model(model_path: str) -> Ensemble:
    if not exists(model_path): raise FileNotFoundError("Model file not found")
    device = select_device()
    return attempt_load(model_path, map_location=device)  # load FP32 model

def get_frame(video_path: Union[Path, str]) -> Generator[str, None, None]: # https://stackoverflow.com/questions/42531143/how-to-type-hint-a-generator-in-python-3
    if not exists(video_path): raise FileNotFoundError("Video file not found")

    dataset = LoadImages(video_path, img_size=640)



def main() -> None:
    print_log("Starting the program")

    print_log("Loading the model")
    model_path = join("/Users/mohammadzainabbas/Masters/CS/Big-Data-Research-Project/src/object_detection/yolov7_with_object_tracking/yolov7.pt")
    model = load_model(model_path)







if __name__ == "__main__":
    main()