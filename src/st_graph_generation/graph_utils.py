from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
@dataclass(unsafe_hash=True)
class Node:
    id: int = field(default=0)
    x1: int = field(default=0)
    y1: int = field(default=0)
    x2: int = field(default=0)
    y2: int = field(default=0)
    conf: float = field(default=float(0))
    detclass: int = field(default=0)
    class_name: str = field(default="")
    centroid: tuple = field(default=(0, 0))

@dataclass(unsafe_hash=True)
class Edge:
    weight: Union[float, int] = field(default=0)

