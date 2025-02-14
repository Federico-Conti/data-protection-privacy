import pandas as pd
import random
from faker import Faker
import os
from dotenv import load_dotenv
load_dotenv()

EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")

# Initialize Faker and set random seed for reproducibility
fake = Faker()
random.seed(42)

# Helper function to generate hierarchical names with probability
def generate_name():
    role = random.choices(
        ["*", "Student", "Professional", fake.first_name()],  # Choices
        weights=[0.1, 0.1, 0.1, 0.7],                           # Probabilities
        k=1
    )[0]
    return role

# Generate nodes and labels
def generate_node_data(num_vertices):
    node_data = []
    for node_id in range(1, num_vertices + 1):
        label = generate_name()
        node_data.append({"id": node_id, "label": label})
    return pd.DataFrame(node_data)

# Generate edges
def generate_edge_data(num_vertices):
    edge_data = []
    for node_id in range(1, num_vertices + 1):
        num_links = random.randint(1, 3)  # Each vertex connects to 1-3 other vertices
        for _ in range(num_links):
            target_id = random.randint(1, num_vertices)
            if node_id != target_id:  # Avoid self-loops
                edge_data.append({"id_1": node_id, "id_2": target_id})
    return pd.DataFrame(edge_data)

# Generate and save the datasets
def generate_graph_csv(num_vertices, node_file, edge_file):
    # Generate nodes and edges
    nodes = generate_node_data(num_vertices)
    edges = generate_edge_data(num_vertices)

    # Save to CSV files
    nodes.to_csv(node_file, index=False)
    edges.to_csv(edge_file, index=False)

    print(f"Node data saved to: {node_file}")
    print(f"Edge data saved to: {edge_file}")

# Parameters
num_vertices = 10

# Generate the graph CSV files
generate_graph_csv(num_vertices, NODES_PATH, EDGES_PATH)
