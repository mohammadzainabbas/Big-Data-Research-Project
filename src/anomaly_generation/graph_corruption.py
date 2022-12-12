'''
POSSIBLE IDEAS FOR CORRUPTION (to develop, update, etc.)

- [only if the graph is NOT fully connected] permutate edges (like Anograph) [only if the graph is not fully connected]
- [only if the graph is NOT fully connected] generating or removing edges

- change weight of the edges
    - for every frame, increase the speed/reduce the distance wrt to the previous frame
- add nodes
    - at a random time, generate a still object for the following frames #TODO
    - at a random time, generate an object and make it move every frame #TODO
- change boundary boxes
    - randomly change (increase) dimension of some randomly chosen objects in random frames #TODO
- randomly corrupt categories
    (- in a sequence, from the first frame where there are objects randomly choose some and corrupt their categories from that moment on
    (- in a sequence, from a random frame where there are objects, randomly choose some and corrupt their categories from that moment on #TODO
    (- during the sequence, whenever a new object comes out decide randomly whether to corrupt its category from that moment on
    summary of all these options: for each frame, if the object is new decide whether to corrupt its category from that moment on
        (optionally) if the object is not new, you can also revaluate with a smaller probability whether to corrupt its cat from that moment on

'''

import networkx as nx
from dataclasses import dataclass, field
from scipy.spatial import distance
import random
import uuid

CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',\
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',\
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',\
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',\
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',\
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',\
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class Corruptor:
    def __init__(self, frame_height: int, frame_width: int, is_stg: bool) -> None:
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.is_stg = is_stg # if stg, the edge has only one attribute (distance). otherwise, it has two (distance and speed)
        self.seen_nodes = dict() #TODO when a node is not there anymore, delete it from here (the id could be reassigned)
    
    def corrupt_graph(graph: nx.Graph):
        copy = graph.copy()
        #TODO corrupt copy
        #TODO if nodes are added, provide the list of added nodes (how to know the object tracker id from Zain's code?)
        return copy, list(["added nodes"])


''''
PSEUDOCODE FOR ZAIN
for clip in video:
    corr = Corruptor(..., ..)
    list1 = list()
    corr_list = list()

    for frame in clip:
        graph = nx.Graph()
        corrupted_graph, added_nodes = corr.corrupt_graph(graph)
        list1.append(graph)
        corr_list.append(corrupted_graph)
'''

#TODO make it unique?
def random_id():
    return int(str(uuid.uuid4().fields[-1]))

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

#TODO remove this function as soon as you have the real code
def dummy_graph():
    graph = nx.Graph()
    add_random_nodes_in(graph, 5)
    
    for n1 in graph.nodes:
        for n2 in graph.nodes:
            if n1!=n2:
                graph.add_edge(n1, n2, weight=distance.euclidean(n1.centroid, n2.centroid))
                #print(graph.edges[n1, n2]['weight'])

    return graph

#TODO see if you can implement this or not
# for making this function consistent in time, the set of generated nodes should be kept for the following frames
# however, this would make the objects still
def add_random_nodes_in(graph: nx.Graph, k: int) -> nx.Graph: #TODO missing arguments: img_width: int, img_height: int
    # TODO decide how to connect these nodes
    for _ in range(k):
        id = random_id()
        #TODO according to the meaning of x1,x2,y1,y2 you need to multiply the random percentage to the width/height of the image
        # to get the first point, and then multiply the percentage only to the difference (the available height/width) for the other
        x1=int(random.random()*100)
        y1=int(random.random()*100)
        x2=int(random.random()*100)
        y2=int(random.random()*100)
        category = random.choice(CATEGORIES)
        graph.add_node(Node(id, x1, y1, x2, y2, conf=0, detclass="", class_name=category, centroid=((x1 + x2) // 2, (y1 + y2) // 2)))





def corrupt_boundary_boxes_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        k = len(graph.nodes)
    nodes_to_corrupt = random.sample(list(graph.nodes), k=k)

    for node in nodes_to_corrupt:
        bigger = bool(random.getrandbits(1)) #this could be kept constant for a node in a sequence of frames
        #TODO sanity checks on boundaries
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
    
        node.centroid = ((node.x1 + node.x2) // 2, (node.y1 + node.y2) // 2)





# TODO delete because it works only for a single frame
def corrupt_category_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        k = len(graph.nodes)
    nodes_to_corrupt = random.sample(list(graph.nodes), k=k)
    corruputed_categories = random.choices(CATEGORIES, k=k)

    for node, category in zip(nodes_to_corrupt, corruputed_categories):
        node.class_name = category
    






        

''' FUNCTIONS THAT ARE CURRENTLY SETTLED FOR SEQUENCE (CORRUPTING CONSISTENTLY WITH TIME)'''

# this function is probably unaffected by the temporal sequence (a frame can have completely different weights than before)
def corrupt_weights_in(graph: nx.Graph, k: int = None) -> nx.Graph:
    if k is None:
        edges_to_corrupt = list(graph.edges(data=True))
    else:
        edges_to_corrupt = random.sample(list(graph.edges(data=True)), k=k)

    for n1, n2, d in edges_to_corrupt:
        d['weight']+=7 #TODO change update (if the weight represents the distance, for anomalies it should be increased)

def corrupt_weights_in_sequence(graphs: list[nx.Graph]) -> list[nx.Graph]:
    for graph in graphs:
        corrupt_weights_in(graph)
    

# this function is probably unaffected by the temporal sequence (a frame can have completely different weights than before)
def permute_weights_in(graph: nx.Graph) -> nx.Graph:
    # this function is meaningful only if the graph is fully connected (since only the weights are permuted)
    # otherwise, TODO add a function to permute also the involved nodes
    weights = [d['weight'] for _, _, d in graph.edges(data=True)]
    random.shuffle(weights)

    index = 0
    for index, edge in enumerate(graph.edges(data=True)):
        n1, n2, d = edge
        d['weight'] = weights[index]

def permute_weights_in_sequence(graphs: list[nx.Graph]) -> list[nx.Graph]:
    for graph in graphs:
        permute_weights_in(graph)


def corrupt_category_in_sequence(graphs: list[nx.Graph]) -> list[nx.Graph]:
    nodes_to_corr_category = dict()
    for graph in graphs:
        # choose nodes to corrupt once in a sequence and apply this corruption to the following nodes
        if not nodes_to_corr_category:
            # skip frames without objects
            if not len(graph.nodes): continue
            # randomly choose some objects in the frame
            k = random.randint(1, len(graph.nodes))
            nodes_to_corrupt = random.sample(list(graph.nodes), k=k)
            corrupted_categories = random.choices(CATEGORIES, k=k)
            nodes_to_corr_category = {node : corr_category for node, corr_category in zip(nodes_to_corrupt, corrupted_categories)}
        # corrupt nodes in the current frame
        for node, corr_category in nodes_to_corr_category.items():
            if node in graph.nodes: #TODO check if this if works if the graphs are different but with node with the same id
                node.class_name = corr_category

#TODO alternative version
def corrupt_category_in_sequence(graphs: list[nx.Graph], corruption_prob: float = 0.4) -> list[nx.Graph]:
    nodes_to_corr_category = dict() #TODO decide whether to put the id as key
    for graph in graphs:
        for node in graph.nodes:
            # if the object is new
            if node not in nodes_to_corr_category.keys():
                if random.random() < corruption_prob:
                    nodes_to_corr_category[node] = random.choice(CATEGORIES)
                else:
                    nodes_to_corr_category[node] = node.class_name
            node.class_name = nodes_to_corr_category[node]
                


#TODO remove these methods
def test_weights(seq):
    print([[d for _, _, d in g.edges(data=True)] for g in seq])
    corrupt_weights_in_sequence(seq)
    print([[d for _, _, d in g.edges(data=True)] for g in seq])

def test_categories(seq):
    print([[n.class_name for n in g.nodes] for g in seq])
    corrupt_category_in_sequence(seq)
    print([[n.class_name for n in g.nodes] for g in seq])



if __name__=='__main__':
    '''
    CORRUPTION OF A SINGLE GRAPH
    g = dummy_graph()
    print([n.class_name for n in g.nodes])
    print([d['weight'] for _, _, d in g.edges(data=True)])
    print([n.boundary_box() for n in g.nodes])
    print()

    corrupt_category_in(g, 3)
    print([n.class_name for n in g.nodes])

    corrupt_weights_in(g, 3)
    print([d['weight'] for _, _, d in g.edges(data=True)])

    permute_weights_in(g)
    print([d['weight'] for _, _, d in g.edges(data=True)])

    corrupt_boundary_boxes_in(g, 3)
    print([n.boundary_box() for n in g.nodes])
    '''


    '''
    CORRUPTION OF A SEQUENCE OF GRAPHS
    '''
    g = dummy_graph()
    c = g.copy()
    c.remove_node(list(g.nodes)[0])
    l = [g, c]
    
    test_weights(l)
