import csv
import argparse
import os
import copy
import time

from dotenv import load_dotenv
load_dotenv()

# Import your anonymization and graph classes
from approach import Anonymization
from graph import Graph, Node

# Environment variables for file paths
EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")
RESULT_EDGES_PATH = os.getenv("RESULT_EDGES_PATH")
RESULT_NODES_PATH = os.getenv("RESULT_NODES_PATH")


def run_anonymization(anon, k):
    """
    Runs the anonymization procedure (the while loop) and returns the anonymized graph along with metrics.
    """
    start = time.time()

    VertexList = anon.G_prime.N
    VertexList.sort(key=lambda node: (len(node.edges), 
                                        sum(len(anon.G_prime.getNode(edge).edges) if anon.G_prime.getNode(edge) else 0 
                                            for edge in node.edges)),
                    reverse=True)
    VertexListCopy = VertexList[:]  

    while VertexListCopy:
        SeedVertex = VertexListCopy.pop(0)
        if not VertexListCopy:
            break 

        # Calculate costs for remaining vertices (using your cost function)
        costs = [
            (anon.cost(anon.G_prime.neighborhoods[SeedVertex],
                       anon.G_prime.neighborhoods[v],
                       anon.alpha, anon.beta, anon.gamma), v)
            for v in VertexListCopy
        ]
        costs.sort(key=lambda x: x[0]) 

        # Create candidate set: either k-1 candidates or all that remain
        if len(VertexListCopy) >= 2 * k - 1:
            CandidateSet = [v for _, v in costs[:k - 1]]
        else:
            CandidateSet = VertexListCopy

        if CandidateSet:
            # Anonymize the neighborhood of SeedVertex with the first candidate
            anon.anonymize_neighborhoods([SeedVertex, CandidateSet[0]])
            # For additional candidates, anonymize pairwise with already anonymized vertices
            for j in range(1, len(CandidateSet)):
                candidate_vertices = [SeedVertex] + CandidateSet[:j]
                for node in candidate_vertices:
                    anon.anonymize_neighborhoods([CandidateSet[j], node])
            # Check if all neighborhoods now match (using a hash on NCC values)
            ncc_values = [anon.G_prime.neighborhoods[v].NCC for v in [SeedVertex] + CandidateSet]
            hashed_ncc_values = {tuple(map(tuple, ncc)) for ncc in ncc_values}
            if len(hashed_ncc_values) != 1:
                for v in candidate_vertices:
                    v.Anonymized = False
            if all(v.Anonymized for v in [SeedVertex] + CandidateSet):
                anon.anonymized_groups.append([SeedVertex] + CandidateSet)

        # Update the list of vertices that still need anonymization
        VertexListCopy = [v for v in anon.G_prime.N if not v.Anonymized]

    runtime = time.time() - start

    # Compute anonymization metrics:
    original_edges = {
        tuple(sorted([node.node_id, neighbor]))
        for node in anon.G.N for neighbor in node.edges
    }
    anonymized_edges = {
        tuple(sorted([node.node_id, neighbor]))
        for node in anon.G_prime.N for neighbor in node.edges
    }
    edges_added = len(anonymized_edges) - len(original_edges)

    original_labels = {node.node_id: node.label for node in anon.G.N}
    anonymized_labels = {node.node_id: node.label for node in anon.G_prime.N}
    labels_anonymized = sum(1 for node_id in anonymized_labels
                            if anonymized_labels[node_id] != original_labels.get(node_id))

    metrics = {
        "edges_added": edges_added,
        "labels_anonymized": labels_anonymized,
        "runtime": runtime
    }
    return anon.G_prime, metrics

###############################################
#            ANALYSIS FUNCTIONS               #
###############################################

def test_anonymization_parameters(original_graph, k, param_combinations):
    """
    Tests the anonymization procedure using different combinations of alpha, beta, and gamma.
    For each combination, it deep-copies the original graph, runs the anonymization,
    and prints metrics such as the number of edges added, labels changed, and runtime.
    """
    results = []
    for (alpha, beta, gamma) in param_combinations:
        print(f"\nTesting with parameters: α={alpha}, β={beta}, γ={gamma}")
        graph_copy = copy.deepcopy(original_graph)
        anon = Anonymization(graph_copy, k, alpha, beta, gamma)
        anon.extract_neighborhoods()
        _, metrics = run_anonymization(anon, k)
        print(f"  Edges added: {metrics['edges_added']}")
        print(f"  Labels anonymized: {metrics['labels_anonymized']}")
        print(f"  Runtime: {metrics['runtime']:.4f} seconds")

def calculate_utility_loss(original_graph, anonymized_graph):
    """
    Computes the utility loss in terms of label distortion and edge additions.
    """
    # Define label hierarchy
    label_hierarchy = {
        "*": 3,           
        "Professional": 2,
        "Student": 1
    }
    max_level = label_hierarchy["*"]

    def get_level(label):
        """Return the generalization level for a given label."""
        if label is None:
            return 0
        return label_hierarchy.get(label, 0)

    def label_distance(orig_label, anon_label):
        """
        Compute the normalized difference between two labels.
        If one label is missing, we treat that as maximum difference.
        """
        if orig_label is None and anon_label is None:
            return 0
        if orig_label is None or anon_label is None:
            return 1
        return abs(get_level(orig_label) - get_level(anon_label)) / max_level

    # Compute average label loss over all nodes
    anon_labels = {node.node_id: node.label for node in anonymized_graph.N}
    label_losses = []
    for node in original_graph.N:
        orig_label = node.label
        anon_label = anon_labels.get(node.node_id)
        loss = label_distance(orig_label, anon_label)
        label_losses.append(loss)
    avg_label_loss = sum(label_losses) / len(label_losses) if label_losses else 0

    # Compute edge loss as the proportion of added edges.
    def extract_edges(graph):
        edges = set()
        for node in graph.N:
            for neighbor in node.edges:
                edge = tuple(sorted([node.node_id, neighbor]))
                edges.add(edge)
        return edges

    orig_edges = extract_edges(original_graph)
    anon_edges = extract_edges(anonymized_graph)
    added_edges = len(anon_edges - orig_edges)
    edge_loss = (added_edges / len(orig_edges)) if orig_edges else 0

    print("\nUtility Loss Analysis:")
    print(f"  Average Label Loss: {avg_label_loss:.2f} (0 = no distortion, 1 = maximum distortion)")
    print(f"  Edge Loss (Proportion of Added Edges): {edge_loss:2f}")


###############################################
#                MAIN FUNCTION                #
###############################################

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', type=int, default=2, help='Example k value')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for alpha in cost function')
    parser.add_argument('--beta', type=float, default=2.0, help='Weight for beta in cost function')
    parser.add_argument('--gamma', type=float, default=3.0, help='Weight for gamma in cost function')
    parser.add_argument('--nodes_file_path', type=str, default=NODES_PATH, help='Path to the nodes CSV file')
    parser.add_argument('--edges_file_path', type=str, default=EDGES_PATH, help='Path to the edges CSV file')
    args = parser.parse_args()

    k = args.k
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    nodes_file_path = args.nodes_file_path
    edges_file_path = args.edges_file_path

    # ------------------ PREPARE PHASE: Build the Graph ------------------
    graph = Graph()
    with open(nodes_file_path, mode='r') as nodes_file:
        csv_reader = csv.reader(nodes_file)
        next(csv_reader)  # skip header
        for row in csv_reader:
            node_id, label = int(row[0]), row[1]
            node = graph.getNode(node_id)
            if node is None:
                node = Node(node_id, label)
                graph.addVertex(node)
            elif node.label is None:
                node.label = label

    # Load edges
    with open(edges_file_path, mode='r') as edges_file:
        csv_reader = csv.reader(edges_file)
        next(csv_reader)  # skip header
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

    # ------------------ ANONYMIZATION PHASE ------------------
    original_graph = copy.deepcopy(graph)
    anon = Anonymization(graph, k, alpha, beta, gamma)
    anon.extract_neighborhoods()
    anonymized_graph, metrics = run_anonymization(anon, k)

    print("\nFinal Anonymization Metrics:")
    print(f"  Edges added: {metrics['edges_added']}")
    print(f"  Labels anonymized: {metrics['labels_anonymized']}")
    print(f"  Runtime: {metrics['runtime']:.4f} seconds")

    # ------------------ OUTPUT PHASE: Write the Anonymized Graph ------------------
    # with open(RESULT_NODES_PATH, mode='w', newline='') as nodes_out:
    #     csv_writer = csv.writer(nodes_out)
    #     csv_writer.writerow(["id", "label"])  # header
    #     for node in anonymized_graph.N:
    #         csv_writer.writerow([node.node_id, node.label])
    # with open(RESULT_EDGES_PATH, mode='w', newline='') as edges_out:
    #     csv_writer = csv.writer(edges_out)
    #     csv_writer.writerow(["id_1", "id_2"])  # header
    #     processed_edges = set()
    #     for node in anonymized_graph.N:
    #         for neighbor in node.edges:
    #             edge = tuple(sorted([node.node_id, neighbor]))
    #             if edge not in processed_edges:
    #                 csv_writer.writerow(edge)
    #                 processed_edges.add(edge)

    # ------------------ ANALYSIS FUNCTIONS ------------------
    param_combinations = [
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3)
    ]
    print("\n=== Testing Different Parameter Combinations ===")
    test_anonymization_parameters(original_graph, k, param_combinations)

    print("\n=== Calculating Utility Loss ===")
    calculate_utility_loss(original_graph, anonymized_graph)

if __name__ == "__main__":
    main()
