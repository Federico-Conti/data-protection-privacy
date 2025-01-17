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
                node1 = Node(id_1)
                graph.addVertex(node1)
            elif node1.label is None:
                node1.label = label
                
            # Handle the second node
            node2 = graph.getNode(id_2)
            if node2 is None:
                node2 = Node(id_2)
                graph.addVertex(node2)
            
            node1.addEdge(id_2)
            node2.addEdge(id_1)

    # Output the nodes and their connections
    for node in graph.N:
        print(node)
    print("\n\n\n")

    # Anonymize the graph
    anon = Anonymization(graph)
    anon.extract_neighborhoods()
    
    for key, value in anon._neighborhoods.items():
        print(f"Node: {key.node_id}, Neighborhood: {value}\n")
    print("\n\n\n")
    VertexList = anon.nodes_sorted_by_neighborhood_size() #lista di nodi ordinati rispetto alle loro NCC
    
    for v in VertexList:
        print(v)
       
    
 
if __name__ == "__main__":
    main()
