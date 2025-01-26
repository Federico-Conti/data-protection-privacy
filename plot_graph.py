import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")
BEFORE_PLOT_PATH = os.getenv("BEFORE_PLOT_PATH")

# Read nodes and edges into DataFrames
nodes_df = pd.read_csv(NODES_PATH)
edges_df = pd.read_csv(EDGES_PATH)

# Create a graph
G = nx.Graph()

# Add nodes with labels
for _, row in nodes_df.iterrows():
    G.add_node(row['id'], label=row['label'])

# Add edges
for _, row in edges_df.iterrows():
    G.add_edge(row['id_1'], row['id_2'])

# Draw the graph
plt.figure(figsize=(10, 10))  # Adjust figure size for better visualization
pos = nx.spring_layout(G, seed=42)  # Position nodes using the spring layout

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)

# Prepare labels for nodes
labels = {node: f"{node}: {data['label']}" for node, data in G.nodes(data=True)}

# Draw labels
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="black")

# Add title and clean up plot
plt.title("Graph Visualization", fontsize=16)
plt.axis("off")  # Turn off the axes

# Show or save the graph
plt.savefig(BEFORE_PLOT_PATH)  # Save as an image
plt.show()
