import networkx as nx 
import matplotlib.pyplot as plt
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Define paths for both versions
paths = {
    "before": {
        "edges": os.getenv("EDGES_PATH"),
        "nodes": os.getenv("NODES_PATH"),
        "plot": os.getenv("BEFORE_PLOT_PATH"),
    },
    "after": {
        "edges": os.getenv("RESULT_EDGES_PATH"),
        "nodes": os.getenv("RESULT_NODES_PATH"),
        "plot": os.getenv("AFTER_PLOT_PATH"),
    },
}

def generate_plot(edges_path, nodes_path, plot_path, title):
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(row['id'], label=row['label'])

    for _, row in edges_df.iterrows():
        G.add_edge(row['id_1'], row['id_2'])

    plt.figure(figsize=(12, 12))  

    pos = nx.spring_layout(G, seed=42, k=1.5)  

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=1.2, edge_color="gray", alpha=0.6)

    labels = {node: f"{node}: {data['label']}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black")

    plt.title(title, fontsize=16)
    plt.axis("off")

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

generate_plot(paths["before"]["edges"], paths["before"]["nodes"], paths["before"]["plot"], "Graph Visualization - Before")
generate_plot(paths["after"]["edges"], paths["after"]["nodes"], paths["after"]["plot"], "Graph Visualization - After")
