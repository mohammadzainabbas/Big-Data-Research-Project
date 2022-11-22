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
from dataclasses import dataclass, field
from stg_utils import *

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
    return attempt_load(model_path, map_location=device), device  # load FP32 model

def get_frames(video_path: Union[Path, str]) -> Generator[str, None, None]: # https://stackoverflow.com/questions/42531143/how-to-type-hint-a-generator-in-python-3
    if not exists(video_path): raise FileNotFoundError("Video file not found")
    return LoadImages(video_path, img_size=640)

def main() -> None:
    print_log("Starting the program")

    print_log("Loading the model")
    model_path = join("/Users/mohammadzainabbas/Masters/CS/Big-Data-Research-Project/src/object_detection/yolov7_with_object_tracking/yolov7.pt")
    imgsz = 640
    delta_time = (1, 5) # take 1 frame every 5 frames

    model, device = load_model(model_path)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    print_log("Model loaded")

    data_dir = join("/Users/mohammadzainabbas/Masters/CS/Big-Data-Research-Project/data")
    video_path = join(data_dir, "street.mp4")

    dataset = get_frames(video_path)
    params = get_video_params(video_path)
    print_log(f"{len(dataset) = }")
    print_log(f"{dataset.nframes = }")
    print_log(f"{params = }")

    # for path, img, im0s, vid_cap in dataset:
    #     print_log(f"Processing image: {path}")
    #     img = torch.from_numpy(img).to(device)
    #     img = img.float()

    print_log(f"{type(dataset) = }")






if __name__ == "__main__":
    main()
