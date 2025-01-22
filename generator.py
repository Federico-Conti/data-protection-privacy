import pandas as pd
import random
from faker import Faker

# Initialize Faker and set random seed for reproducibility
fake = Faker()
random.seed(42)

# Helper function to generate hierarchical names with probability
def generate_name(role_level):
    # Assign probabilities to "Student" and "Professional"
    role = random.choices(
        ["*","Student", "Professional",fake.first_name()],  # Choices
        weights=[0.1,0.1,0.1,0.7],          # Probabilities
        k=1                          # Number of selections
    )[0]
    return f"{role}"

# Generate graph data
def generate_graph_data(num_vertices):
    data = []
    ids = list(range(num_vertices))
    assigned_names = {}

    for id_1 in ids:
        # Assign a name to each vertex based on a hierarchy
        if id_1 not in assigned_names:
            assigned_names[id_1] = generate_name(random.randint(0, 3))

        # Generate random edges
        num_links = random.randint(1, 3)  # Each vertex connects to 1-3 other vertices
        for _ in range(num_links):
            id_2 = random.choice(ids)
            if id_1 != id_2:  # Avoid self-loops
                data.append({"id_1": id_1, "id_2": id_2, "name": assigned_names[id_1]})
                
    return pd.DataFrame(data)

# Generate and display the dataset
graph_data = generate_graph_data(num_vertices=30)
file_path = "./real.csv"
graph_data.to_csv(file_path, index=False)

print("Graph data generated and saved to:", file_path)
