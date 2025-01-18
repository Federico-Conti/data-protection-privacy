class Node:
    def __init__(self, node_id, label):
        self.node_id = node_id
        self.label = label
        self.Visited = False
        self.edges = []  # List to store connected node IDs
        
    def addEdge(self, neighbor_id):
        if neighbor_id not in self.edges:  # ANoid duplicate edges
            self.edges.append(neighbor_id)
            
    def getEdgesInComponent(self, component):
        for e in self.edges:
            if e in [n.node_id for n in component]:
                yield e
            
    def induced_subgraph_size(self,graph):
        neighbors = set(self.edges)  # Nodes directly connected to 'node'
        edges_in_neighborhood = 0
        for neighbor_id in neighbors:
            neighbor = graph.getNode(neighbor_id)
            if neighbor:
                # Count edges among neighbors
                edges_in_neighborhood += sum(1 for edge in neighbor.edges if edge in neighbors)
        edges_in_neighborhood //= 2  # Avoid double-counting edges
        return len(neighbors), edges_in_neighborhood
    
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
        if node_id is not None:
            for node in self.N:
                if node.node_id == node_id:
                    return node
        return None

