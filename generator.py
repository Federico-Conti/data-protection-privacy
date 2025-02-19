import pandas as pd
import random
from faker import Faker
import os
from dotenv import load_dotenv
load_dotenv()

EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")

fake = Faker()
random.seed(42)

def generate_name():
    role = random.choices(
        ["*", "Student", "Professional", fake.first_name()],  
        weights=[0.1, 0.1, 0.1, 0.7],                           
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

def generate_edge_data(num_vertices):
    edge_data = []
    edges_set = set()
    for node_id in range(1, num_vertices + 1):
        desired_links = random.randint(5, 10)
        links_created = 0
        while links_created < desired_links:
            target_id = random.randint(1, num_vertices)
            if node_id == target_id:
                continue  
            edge = (min(node_id, target_id), max(node_id, target_id))
            if edge in edges_set:
                continue 
            edges_set.add(edge)
            edge_data.append({"id_1": edge[0], "id_2": edge[1]})
            links_created += 1
    return pd.DataFrame(edge_data)


def generate_graph_csv(num_vertices, node_file, edge_file):
    nodes = generate_node_data(num_vertices)
    edges = generate_edge_data(num_vertices)
    nodes.to_csv(node_file, index=False)
    edges.to_csv(edge_file, index=False)

    print(f"Node data saved to: {node_file}")
    print(f"Edge data saved to: {edge_file}")

num_vertices = 50

generate_graph_csv(num_vertices, NODES_PATH, EDGES_PATH)
