import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

EDGES_PATH = os.getenv("EDGES_PATH")
NODES_PATH = os.getenv("NODES_PATH")
BEFORE_PLOT_PATH = os.getenv("BEFORE_PLOT_PATH")

# EDGES_PATH = os.getenv("RESULT_EDGES_PATH")
# NODES_PATH = os.getenv("RESULT_NODES_PATH")
# BEFORE_PLOT_PATH = os.getenv("AFTER_PLOT_PATH")

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

# Increase figure size for better readability
plt.figure(figsize=(12, 12))  

# Use a layout with better node separation
pos = nx.spring_layout(G, seed=42, k=1.5)  # Increase 'k' for better spacing

# Draw graph elements
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, width=1.2, edge_color="gray", alpha=0.6)

# Prepare labels with reduced font size
labels = {node: f"{node}: {data['label']}" for node, data in G.nodes(data=True)}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black")

# Title and cleanup
plt.title("Graph Visualization", fontsize=16)
plt.axis("off")

# Save and show the plot
plt.savefig(BEFORE_PLOT_PATH, dpi=300, bbox_inches="tight")
plt.show()