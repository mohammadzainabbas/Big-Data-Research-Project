from os.path import join, exists
from typing import Union
from pathlib import Path
import cv2

def print_log(text: str) -> None: print(f"[ log ] {text}")
def print_error(text: str) -> None: print(f"[ error ] {text}")

def get_video_params(video_path: Union[Path, str]) -> int:
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
