import csv
from approach import Anonymization
from graph import Graph, Node
import argparse
import os

from dotenv import load_dotenv
load_dotenv()

EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")
RESULT_EDGES_PATH = os.getenv("RESULT_EDGES_PATH")
RESULT_NODES_PATH = os.getenv("RESULT_NODES_PATH")

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', type=int, default=2, help='Example k value')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for alpha in cost function')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for beta in cost function')
    parser.add_argument('--gamma', type=float, default=1.0, help='Weight for gamma in cost function')
    parser.add_argument('--nodes_file_path', type=str, default=NODES_PATH, help='Path to the CSV file')
    parser.add_argument('--edges_file_path', type=str, default=EDGES_PATH, help='Path to the CSV file')

    args = parser.parse_args()

    k = args.k
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    nodes_file_path = args.nodes_file_path
    edges_file_path = args.edges_file_path
    graph = Graph()
# \                             ** PREPARE PHASE **
    with open(nodes_file_path, mode='r') as nodes_file:
        csv_reader = csv.reader(nodes_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            id_1, label = int(row[0]), row[1]
            
            # Handle the node
            node1 = graph.getNode(id_1)
            if node1 is None:
                node1 = Node(id_1, label)
                graph.addVertex(node1)
            elif node1.label is None:
                node1.label = label

    # Step 2: Read and process edges CSV
    with open(edges_file_path, mode='r') as edges_file:
        csv_reader = csv.reader(edges_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            id_1, id_2 = int(row[0]), int(row[1])
            
            node1 = graph.getNode(id_1)
            if node1 is None:
                node1 = Node(id_1, None)
                graph.addVertex(node1)
            
            node2 = graph.getNode(id_2)
            if node2 is None:
                node2 = Node(id_2, None)
                graph.addVertex(node2)
            
            node1.addEdge(id_2)
            node2.addEdge(id_1)
    
# \                             ** NEIGHBORHOODS EXTRACTION AND CODING **
    anon = Anonymization(graph,k,alpha, beta, gamma)
    anon.extract_neighborhoods()
    
    VertexList = anon.G_prime.N
    assert len(VertexList) > 2 * k - 1, f"Number of vertices in the graph must be at least 2k-1 = {2 * k - 1}"
    VertexList.sort(key=lambda node: (len(node.edges), sum(len(anon.G_prime.getNode(edge).edges) if anon.G_prime.getNode(edge) else 0 for edge in node.edges)), reverse=True)
    print("\n\nSorted Vertices:")
    for v in VertexList:
        print(v)


# \                                              ** ANONYMIZATION **
    VertexListCopy = VertexList[:]

    while VertexListCopy:
        # Select seed vertex
        SeedVertex = VertexListCopy.pop(0)

        # Calculate costs for all remaining vertices
        costs = [(anon.cost(anon.G_prime.neighborhoods[SeedVertex], anon.G_prime.neighborhoods[v], alpha, beta, gamma), v) for v in VertexListCopy]
        costs.sort(key=lambda x: x[0])  # Sort by cost

        # Create candidate set
        if len(VertexListCopy) >= 2 * k - 1:
            CandidateSet = [v for _, v in costs[:k - 1]]
        else:
            CandidateSet = VertexListCopy
        
        # Anonymize the neighborhoods
        # Anonymize Neighbor(SeedVertex) and Neighbor(u1)
        anon.anonymize_neighborhoods([SeedVertex] + [CandidateSet[0]])
        if len(CandidateSet) == 1:
            for node in [SeedVertex] + CandidateSet:
                node.Anonymized = True
        
        # Anonymize Neighbor(uj) and {Neighbor(SeedVertex), Neighbor(u1), ..., Neighbor(uj-1)}
        for j in range(1, len(CandidateSet)):
            
            ncc_values = tuple()
            
            while len(set(ncc_values)) != 1:
                candidate_vertices = [SeedVertex] + CandidateSet[:j]
                
                for node in candidate_vertices:  
                    anon.anonymize_neighborhoods([CandidateSet[j]]+[node])
                    
                # Check if all NCCs are equal
                ncc_values = [tuple(map(tuple, anon.G_prime.neighborhoods[node].NCC)) for node in candidate_vertices + [CandidateSet[j]]]
        
            for node in candidate_vertices+[CandidateSet[j]]:
                node.Anonymized = True
                    # Mark all candidate vertices as anonymized
        anon.anonymized_groups.append([SeedVertex] + CandidateSet)
        
        for node in anon.G_prime.N:
            print(f"Node {node.node_id} anonymized: {node.Anonymized}")
        
        # Update VertexList
        VertexListCopy = [v for v in anon.G_prime.N if v.Anonymized == False]
    
    print("\n\n")
    
    # Print the NCC (Normalized Clustering Coefficient) of each node in the anonymized graph
    # OUTPUT PHASE: Output the anonymized graph
    print("\n\n")
    for group in anon.anonymized_groups:
        print("Anonymized Group:")
        for node in group:
            print(node)
        print("\n")
        
    # # WRITE PHASE: Save the anonymized graph to a CSV file
    with open(RESULT_NODES_PATH, mode='w', newline='') as nodes_out:
        csv_writer = csv.writer(nodes_out)
        csv_writer.writerow(["id", "label"])  # Header
        for node in anon.G_prime.N:
            csv_writer.writerow([node.node_id, node.label])


    # Write anonymized edges to CSV
    with open(RESULT_EDGES_PATH, mode='w', newline='') as edges_out:
        csv_writer = csv.writer(edges_out)
        csv_writer.writerow(["id_1", "id_2"])  # Header
        processed_edges = set()
        
        for node in anon.G_prime.N:
            for neighbor in node.edges:
                edge = tuple(sorted([node.node_id, neighbor]))  # Ensure unique edges
                if edge not in processed_edges:
                    csv_writer.writerow(edge)
                    processed_edges.add(edge)

if __name__ == "__main__":
    main()