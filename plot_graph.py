import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Define the path to your CSV file
csv_file_path = 'result.csv'

data = {"id_1": [], "id_2": []}

# Open the CSV file
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)2,2,Quinn
    
    # Skip the header
    next(csv_reader)
    
    # Iterate over the rows in the CSV file
    for row in csv_reader:
        data["id_1"].append(row[0])
        data["id_2"].append(row[1])

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Create a graph
G = nx.from_pandas_edgelist(df, "id_1", "id_2")

# Draw the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=42)  # Spring layout for better visualization

# Draw nodes and edges with improved visualization
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

# Set title and save the graph
plt.title("Undirected Graph Visualization with Node IDs", fontsize=16)
plt.axis("off")  # Hide the axes for a cleaner look
plt.savefig("result.png")
plt.show()
