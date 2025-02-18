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

# Generate edges ensuring each node gets more than five connections
# and no duplicate edges (i.e. if 1,2 exists, 2,1 will not be generated).
def generate_edge_data(num_vertices):
    edge_data = []
    edges_set = set()
    for node_id in range(1, num_vertices + 1):
        # Each node will attempt to create between 6 and 8 edges (more than five).
        desired_links = random.randint(5, 10)
        links_created = 0
        while links_created < desired_links:
            target_id = random.randint(1, num_vertices)
            if node_id == target_id:
                continue  # Avoid self-loops
            # Order the nodes to avoid duplicates (e.g., (1,2) and (2,1))
            edge = (min(node_id, target_id), max(node_id, target_id))
            if edge in edges_set:
                continue  # Skip if this edge already exists
            edges_set.add(edge)
            edge_data.append({"id_1": edge[0], "id_2": edge[1]})
            links_created += 1
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
num_vertices = 50

# Generate the graph CSV files
generate_graph_csv(num_vertices, NODES_PATH, EDGES_PATH)
