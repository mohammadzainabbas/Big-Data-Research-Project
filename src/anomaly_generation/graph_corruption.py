'''
POSSIBLE IDEAS FOR CORRUPTION (to develop, update, etc.)

- change weight of the edges 
- permutate edges (like Anograph) (if the graph is fully connected, it could be useful to change the weight of the nodes)
- add nodes (which nodes?)
- change boundary boxes
- randomly change categories

removing nodes should not be interesting for generating anomalies

according to the approach used for the generation of edges, generating or removing edges could be other ways to generate anomalies
'''

import networkx as nx
from dataclasses import dataclass, field
from scipy.spatial import distance
import random
from typing import TypeVar
import uuid

#TODO import this class
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
    # def __post_init__(self):
    #     self.centroid = ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def boundary_box(self) -> str:
        return f"tl: ({self.x1}, {self.y1}) - br: ({self.x2}, {self.y2})"

CATEGORIES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",\
    "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"
]

def random_id():
    #TODO make it unique?
    return int(str(uuid.uuid4().fields[-1]))

def dummy_graph(): #TODO remove this function as soon as you have the real code
    graph = nx.Graph()
    graph = add_random_nodes_in(graph, 5)
    
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            if n1!=n2:
                graph.add_edge(n1, n2, weight=distance.euclidean(n1.centroid, n2.centroid))
                #print(graph.edges[n1, n2]['weight'])

    return graph

def corrupt_weights_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        k = len(graph.edges)
    edges_to_corrupt = random.sample(list(graph.edges(data=True)), k=k)

    for n1, n2, d in graph.edges(data=True):
        if (n1, n2, d) in edges_to_corrupt:
            d['weight']+=7 #TODO change update
    return g

def permute_weights_in(graph: nx.Graph) -> nx.Graph:
    # this function is meaningful only if the graph is fully connected (since only the weights are permuted)
    # otherwise, TODO add a function to permute also the involved nodes
    weights = [d['weight'] for _, _, d in graph.edges(data=True)]
    random.shuffle(weights)

    index = 0
    for index, edge in enumerate(graph.edges(data=True)):
        n1, n2, d = edge
        d['weight'] = weights[index]
    return g
    
    

def corrupt_boundary_boxes_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        k = len(graph.nodes)
    nodes_to_corrupt = random.sample(list(graph.nodes), k=k)

    for node in nodes_to_corrupt:
        bigger = bool(random.getrandbits(1))
        if bigger:
            node.x1 -= int(random.random()*node.x1)
            node.y1 -= int(random.random()*node.y1)
            node.x2 += int(random.random()*node.x2)
            node.y2 += int(random.random()*node.y2)
        else:
            node.x1 += int(random.random()*node.x1)
            node.y1 += int(random.random()*node.y1)
            node.x2 -= int(random.random()*node.x2)
            node.y2 -= int(random.random()*node.y2)
    
    return graph

def add_random_nodes_in(graph: nx.Graph, k: int) -> nx.Graph:
    # TODO decide how to connect these nodes
    for _ in range(k):
        id = random_id()
        x1=int(random.random()*100)
        y1=int(random.random()*100)
        x2=int(random.random()*100)
        y2=int(random.random()*100)
        category = random.choice(CATEGORIES)
        graph.add_node(Node(id, x1, y1, x2, y2, conf=0, detclass="", class_name=category, centroid=((x1 + x2) // 2, (y1 + y2) // 2)))

    return graph

def corrupt_category_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        k = len(graph.nodes)
    nodes_to_corrupt = random.sample(list(graph.nodes), k=k)
    corruputed_categories = random.choices(CATEGORIES, k=k)

    for node, category in zip(nodes_to_corrupt, corruputed_categories):
        node.class_name = category
    
    return graph




if __name__=='__main__':
    g = dummy_graph()
    print([n.class_name for n in g.nodes])
    print([d['weight'] for _, _, d in g.edges(data=True)])
    print([n.boundary_box() for n in g.nodes])
    print()

    g = corrupt_category_in(g, 3)
    print([n.class_name for n in g.nodes])

    g = corrupt_weights_in(g, 3)
    print([d['weight'] for _, _, d in g.edges(data=True)])

    g = permute_weights_in(g)
    print([d['weight'] for _, _, d in g.edges(data=True)])

    g = corrupt_boundary_boxes_in(g, 3)
    print([n.boundary_box() for n in g.nodes])
