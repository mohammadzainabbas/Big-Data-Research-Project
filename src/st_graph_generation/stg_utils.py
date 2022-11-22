from os.path import join, exists
from typing import Union
from pathlib import Path
import cv2

def print_log(text: str) -> None: print(f"[ log ] {text}")
def print_error(text: str) -> None: print(f"[ error ] {text}")

def get_video_params(video_path: Union[Path, str]) -> int:
    if not exists(video_path): raise FileNotFoundError("Video file not found")
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)
    return cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)

def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params

def get_video_params(fname):
    cap = cv2.VideoCapture(fname)
    params = get_video_params_cap(cap)
    cap.release()
    return params