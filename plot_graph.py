import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Define the path to your CSV file
csv_file_path = 'test.csv'

data = {"id_1": [], "id_2": [], "label": []}

# Open the CSV file
with open(csv_file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Skip the header
    next(csv_reader)
    
    # Iterate over the rows in the CSV file
    for row in csv_reader:
        data["id_1"].append(row[0])
        data["id_2"].append(row[1])
        data["label"].append(row[2])

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
# Combine node id and label for display
combined_labels = {row["id_1"]: f'{row["id_1"]}: {row["label"]}' for _, row in df.iterrows()}

# Draw the combined labels
nx.draw_networkx_labels(G, pos, labels=combined_labels, font_size=10, font_color="black")

# Set title and save the graph
plt.title("Undirected Graph Visualization", fontsize=16)
plt.axis("off")  # Hide the axes for a cleaner look
plt.savefig("before.png")
plt.show()
