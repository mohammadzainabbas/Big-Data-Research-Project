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

#TODO make these corruptive functions consistent with time (i.e. different frames of the same video should be corrupted consistently)
# it is also possible to corrupt every frame differently AS AN ADDITIONAL GENERATION METHOD (not the main one)

#TODO I miss height and width of an image

#TODO the input for this script must be a list of graphs, but the graphs should be a copy (because the methods change them)

import networkx as nx
from dataclasses import dataclass, field
from scipy.spatial import distance
import random
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

#TODO make it unique?
def random_id():
    return int(str(uuid.uuid4().fields[-1]))

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
        k = len(graph.edges)
    edges_to_corrupt = random.sample(list(graph.edges(data=True)), k=k)

    for n1, n2, d in graph.edges(data=True):
        if (n1, n2, d) in edges_to_corrupt:
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
    #TODO another idea
    # - during the sequence, whenever a new object comes out decide randomly whether to corrupt its category from that moment on
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
    print([[n.class_name for n in g.nodes] for g in l])

    corrupt_category_in_sequence(l)

    print([[n.class_name for n in g.nodes] for g in l])

