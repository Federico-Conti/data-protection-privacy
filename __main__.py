import csv
from approach import Anonymization
from graph import Graph, Node

def main():
    # Load the CSV file
    file_path = 'test.csv'
    graph = Graph()

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  
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
            
            node1.addEdge(id_2)
            node2.addEdge(id_1)

    # Output the nodes and their connections
    for node in graph.N:
        print(node)
    print("\n\n\n")
    anon = Anonymization(graph)
    n =  anon.extract_neighborhood(1)

    for node in n.N:
        print(node)
 
if __name__ == "__main__":
    main()
