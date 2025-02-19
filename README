# Preserving Privacy in Social Networks Against Neighborhood Attacks

This research paper addresses the crucial problem of preserving privacy in social network data when releasing it publicly. The authors focus on a specific type of privacy attack called neighborhood attacks, where an adversary, knowing a target individual's immediate connections and their relationships, can re-identify the individual even if their name is removed. The paper introduces a novel anonymization method that aims to counteract neighborhood attacks by grouping individuals with similar neighborhood structures and generalizing their attributes or adding edges to ensure k-anonymity. The effectiveness of this method is demonstrated through experiments with real and synthetic social network data, showing that it preserves privacy while allowing for accurate aggregate network queries. The paper also discusses challenges for future work, including protecting more extended neighborhoods (d>1) and addressing limitations of k-anonymity in terms of protecting sensitive information.

## Steps

The method is in two steps:

1. Extracting the neighborhoods of all vertices in the network and representing them using a neighborhood component coding technique to facilitate comparisons.
2. Greedily organizing vertices into groups and anonymizing the neighborhoods of vertices in the same group, starting with those vertices of high degrees

# Neighborhood Anonyization Simulation

## Dataset Generator

- **Run**: Execute the generator using `generator.py`.
- **Description**: Generate a fake undirected graph dataset give a number of vertex.
- **Output**: Results are saved in the `dataset` folder as `edges-x.csv` and `nodes-x.csv`.
  
## Plot Graph

- **Run**: Execute the plot using `plot_graph.py`.
- **Description**: Take an existing graph dataset from the `dataset` folder and plot it.
- **Output**: Results are saved in the `plots` folder as `before-x.png` or `after-x.png`.

## Neighborhood Anonymization

- **Run**: Execute the simulation using `__main__.py`.
- **Description**: Take an undirected graph dataset from the `dataset` folder and anonymize it using k-anonymity.
- **Scripts**: `approach.py` and `graph.py`.
- **Output**: Results are saved in the `dataset` folder as `edges-result-x.csv` and `nodes-result-x.csv`.