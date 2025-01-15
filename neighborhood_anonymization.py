import csv

class Node:
    def __init__(self, node_id, label):
        self.node_id = node_id
        self.label = label
        self.visited = False
        self.edges = []  # List to store connected node IDs
        
    def addEdge(self, neighbor_id):
        if neighbor_id not in self.edges:  # Avoid duplicate edges
            self.edges.append(neighbor_id)
    
    def isNeighbor(self, neighbor_id):
        return neighbor_id in self.edges

    def __repr__(self):
        return f"Node(id={self.node_id}, label={self.label}, edges={self.edges})"

class Graph:
    def __init__(self):
        self.V = []  # List of vertices (Node objects)
    
    def addVertex(self, node: Node):
        self.V.append(node)

    def getNode(self, node_id):
        for node in self.V:
            if node.node_id == node_id:
                return node
        return None

def main():
    # Load the CSV file
    file_path = 'test.csv'
    graph = Graph()

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            id_1, id_2, label = int(row[0]), int(row[1]), row[2]
            
            # Handle the first node
            node1 = graph.getNode(id_1)
            if node1 is None:
                node1 = Node(id_1, label)
                graph.addVertex(node1)
            
            # Handle the second node
            node2 = graph.getNode(id_2)
            if node2 is None:
                node2 = Node(id_2, label)
                graph.addVertex(node2)
            
            # Add edges
            node1.addEdge(id_2)
            node2.addEdge(id_1)

    # Output the nodes and their connections
    for node in graph.V:
        print(node)

if __name__ == "__main__":
    main()
