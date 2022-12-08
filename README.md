## Big Data Research Project 👨🏻‍💻

![GitHub contributors](https://img.shields.io/github/contributors/mohammadzainabbas/Big-Data-Research-Project)
![GitHub activity](https://img.shields.io/github/commit-activity/w/mohammadzainabbas/Big-Data-Research-Project?logoColor=brightgreen)

### Table of contents

- [Introduction](#introduction)
- [Setup](#setup)

#

<a id="introduction" />

### 1. Introduction

__`Data drives the world.`__ Nowadays, most of the data (_structured_ or _unstructured_) can be analysed as a graph. Today, many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs (_in some cases billions of vertices, trillions of edges_) poses challenges to their efficient processing.

#

<a id="setup" />

### 2. Setup

In order to start, we'd recommend you to clone with all the submodule(s), simply run:

```bash
git clone --recurse-submodules -j8 https://github.com/mohammadzainabbas/Big-Data-Research-Project.git
```

and then run the following to setup env and installing the package locally:

```bash
sh scripts/setup.sh
```

```bash
pip install -e .
```
#

```bash
python src/st_graph_generation/detect_with_networkx.py --no-trace --view-img --source test/street.mp4 --show-fps --seed 2 --track --show-track --project data --name live_graph
```

```bash
python src/st_graph_generation/generate_st_graph.py --no-trace --view-img --source test/street.mp4 --show-fps --seed 2 --track --show-track --project data --name live_graph
```

#
