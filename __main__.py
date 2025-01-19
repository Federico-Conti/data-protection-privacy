import csv
from approach import Anonymization
from graph import Graph, Node
import argparse

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', type=int, default=3, help='Example k value')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for alpha in cost function')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for beta in cost function')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for gamma in cost function')
    parser.add_argument('--file_path', type=str, default='test.csv', help='Path to the CSV file')

    args = parser.parse_args()

    k = args.k
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    file_path = args.file_path
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
            elif node1.label is None:
                node1.label = label
                
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
    print("\n\n")

    # PREPARE PHASE: Extract neighborhoods and sort vertices by induced subgraph size
    anon = Anonymization(graph)
    anon.extract_neighborhoods()
    
    for key, value in anon._neighborhoods.items():
        print(f"Node: {key.node_id}, Neighborhood: {value}\n")
    print("\n\n\n")
    VertexList = anon.G_prime.N
    VertexList.sort(key=lambda node: node.induced_subgraph_size(anon.G_prime), reverse=True)  # Descending order
    # for v in VertexList:
    #     print(v)

    VertexListCopy = VertexList[:]

    # ANONYMIZATION PHASE: Anonymize the neighborhoods of all vertices in the graph
    while VertexListCopy:
        # Select seed vertex
        SeedVertex = VertexListCopy.pop(0)

        # Calculate costs for all remaining vertices
        costs = [(anon.cost(SeedVertex, v, alpha, beta, gamma), v) for v in VertexListCopy]
        costs.sort(key=lambda x: x[0])  # Sort by cost

        # Create candidate set
        if len(VertexListCopy) >= 2 * k - 1:
            CandidateSet = [v for _, v in costs[:k - 1]]
        else:
            CandidateSet = VertexListCopy
        
        print(f"Seed Vertex: {SeedVertex.node_id}, Candidate Set: {[v.node_id for v in CandidateSet]}")
        # Anonymize the neighborhoods
        anon.anonymize_neighborhoods([SeedVertex] + CandidateSet)

        for node in anon.G_prime.N:
            print(f"Node {node.node_id} anonymized: {node.Anonymized}")


        # Update VertexList
        VertexListCopy = [v for v in VertexListCopy if v not in CandidateSet]
    
    print("\n\n")
    # OUTPUT PHASE: Output the anonymized graph
    for node in anon.G_prime.N:
        print(node)
        
    # WRITE PHASE: Save the anonymized graph to a CSV file
    with open('result.csv', mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['id_1', 'id_2', 'label'])
        
        for node in anon.G_prime.N:
            for neighbor_id in node.edges:
                csv_writer.writerow([node.node_id, neighbor_id, node.label])
 
if __name__ == "__main__":
    main()