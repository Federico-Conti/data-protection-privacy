import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

# Define the path to your CSV file
csv_file_path = 'test.csv'

data = {"id_1": [], "id_2": []}

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


# Create a DataFrame from the data
df = pd.DataFrame(data)

# Create a graph
G = nx.from_pandas_edgelist(df, "id_1", "id_2")

# Draw the graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=42)  # Spring layout for better visualization
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=40,
    edge_color="gray",
    node_color="blue"
)
plt.title("Undirected Graph Visualization")
plt.savefig("out.png")
