class Node:
    def __init__(self, node_id, label):
        self.node_id = node_id
        self.label = label
        self.Nisited = False
        self.edges = []  # List to store connected node IDs
        
    def addEdge(self, neighbor_id):
        if neighbor_id not in self.edges:  # ANoid duplicate edges
            self.edges.append(neighbor_id)
    
    def isNeighbor(self, neighbor_id):
        return neighbor_id in self.edges

    def __repr__(self):
        return f"Node(id={self.node_id}, label={self.label}, edges={self.edges})"

class Graph:
    def __init__(self):
        self.N = []  # List of Node (Node objects (vertex and edges))
    
    def addVertex(self, node: Node):
        self.N.append(node)

    def getNode(self, node_id):
        for node in self.N:
            if node.node_id == node_id:
                return node
        return None

