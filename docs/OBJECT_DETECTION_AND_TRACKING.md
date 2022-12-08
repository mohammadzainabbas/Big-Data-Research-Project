## Object detection and tracking 👨🏻‍💻

### Table of contents

- [Object detection](#object-detection)
  * [Yolov7](#yolov7)
- [Object tracking](#object-tracking)
  * [Kalman Filter](#kalman-filter)

#

<a id="object-detection" />

### 1. Object detection

We are using [`Yolov7`](https://github.com/WongKinYiu/yolov7) for object detection. The original paper can be found [here](https://arxiv.org/abs/2207.02696).

There are [several models](https://github.com/WongKinYiu/yolov7#performance) which are availble for free; but for purpose, we are using the baseline [`YOLOv7`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) model.

<a id="yolov7" />

#### 1.1. Yolov7

`YOLOv7` surpasses all known object detectors in both speed and accuracy in the range from `5` FPS to `160` FPS and has the highest accuracy `56.8%` AP among all known real-time object detectors with `30` FPS or higher on GPU V100.

Checkout details about `Yolov7`'s architecture and some benchmarks [here](https://arxiv.org/pdf/2207.02696.pdf).

Currently, 

<a id="object-tracking" />

### 2. Object tracking



<a id="kalman-filter" />

#### 2.1. Kalman Filter

