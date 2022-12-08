## Object detection and tracking 👨🏻‍💻

### Table of contents

- [Object detection](#object-detection)
  * [Yolov7](#yolov7)
- [Output of object detection](#detection-output)
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

Currently, the baseline model that we are using is trained to detect `80` different classes. i.e:

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

<details>
<summary>Click to copy the classes</summary>

```python
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
```

</details>

<a id="detection-output" />

### 2. Output of object detection

Yolov7 outputs the following things:

- [x] Bounding box coordinates _(x1, y1, x2, y2)_
- [x] Class name and it's confidence score

Note that the coordinates are similar to OpenCV, i.e:

```txt
(0,0) --- X --->
 |  
 |  (x1, y1)--------
 Y      |           |
 |      |           |
 |       -----------(x2, y2)
 v      
```
<a id="object-tracking" />

### 3. Object tracking




<a id="kalman-filter" />

#### 2.1. Kalman Filter

