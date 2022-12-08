from os.path import join, exists
from typing import Union
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Any, Union
from argparse import Namespace

def get_video_params(video_path: Union[Path, str]) -> dict:
    if not exists(video_path): raise FileNotFoundError("Video file not found")
    cap = cv2.VideoCapture(video_path)
    params = dict({
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "nframes": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    })
    cap.release()
    return params

def dict_with_attributes(d: Dict[str, Any]) -> Namespace:
    """
    Convert a dictionary to a class with attributes
    """
    return Namespace(**d)
