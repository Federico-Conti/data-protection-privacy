
class Node:
    def __init__(self, node_id, label):
        self.node_id = node_id
        self.label = label
        self.Visited = False
        self.Anonymized = False
        self.edges = []  # List to store connected node IDs
        
    def addEdge(self, neighbor_id):
        if neighbor_id not in self.edges:  # ANoid duplicate edges
            self.edges.append(neighbor_id)
            
    def getEdgesInComponent(self, component):
        edgesInComponent = []
        for e in self.edges:
            if e in [n.node_id for n in component]:
                edgesInComponent.append(e)
        return edgesInComponent
            
    def isNeighbor(self, neighbor_id):
        return neighbor_id in self.edges

    def __repr__(self):
        return f"Node(id={self.node_id}, label={self.label}, edges={self.edges}, count_edges={len(self.edges)})"
    
class Neighborhood:
    def __init__(self, components, NCC):
        self.components = components
        self.NCC = NCC  # List of nodes in the neighborhood
    
    def getNumberOfEdges(self,component):
        edges = 0
        for node in component:
            edges += len(node.getEdgesInComponent(component))
        return edges//2
   

class Graph:
    def __init__(self):
        self.N = []  # List of Node (Node objects (vertex and edges))
        self.neighborhoods = {}  # Dictionary to store neighborhoods: key is Node, value is list of NCC
        
    def addVertex(self, node: Node):
        self.N.append(node)

    def getNode(self, node_id):
        if node_id is not None:
            for node in self.N:
                if node.node_id == node_id:
                    return node
        return None

        
