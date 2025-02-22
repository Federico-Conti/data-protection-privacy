# Preserving Privacy in Social Networks Against Neighborhood Attacks Simulation

## Abstract
This research paper addresses the crucial problem of preserving privacy in social network data when releasing it publicly. The authors focus on a specific type of privacy attack called neighborhood attacks, where an adversary, knowing a target individual's immediate connections and their relationships, can re-identify the individual even if their name is removed. The paper introduces a novel anonymization method that aims to counteract neighborhood attacks by grouping individuals with similar neighborhood structures and generalizing their attributes or adding edges to ensure k-anonymity. The effectiveness of this method is demonstrated through experiments with real and synthetic social network data, showing that it preserves privacy while allowing for accurate aggregate network queries.

## Introduction
Our approach is implemented in Python, where we handle graph construction, neighborhood extraction, DFS code generation, and the entire anonymization process.

## The Problem
Many assume that by stripping away identifying labels from nodes and edges,privacy is automatically preserved. However, this research shows that this is not the case. Even when names and direct identifiers are removed, an attacker armed with knowledge about a target’s neighbors and the interrelationships among those neighbors can re-identify the target. In other words, the structure of the network itself carries information. By studying the connectivity patterns and the relative position of a node within its local network, an adversary can infer the identity of that node despite the absence of explicit labels. This phenomenon is what we refer to as a “neighborhood attack.”  

## Neighborhood Attacks
To elaborate, a neighborhood attack leverages the fact that each user in a social network has a unique set of connections. Even if the labels on nodes or edges are generalized or removed, the pattern of connections the neighborhood can serve as a fingerprint. By analyzing the induced subgraph that represents a user’s neighborhood, an attacker can match this fingerprint with external information to re-identify the user.

## k-anonymity
So, how do we protect against this form of attack? This approach is based on the principle of k-anonymity. The goal here is to ensure that every individual in the network is indistinguishable from at least k–1 other individuals, with confidence larger than 1/k. By grouping vertices with similar neighborhood structures, we limit an adversary’s confidence when attempting to pinpoint any single target.
 
## Neighborhood Extraction
Let’s now discuss how we begin the process. The first step is neighborhood extraction. For every vertex in the network, we extract its “neighborhood,” which is the induced subgraph comprising all its immediate neighbors. This extraction is crucial because it lays the foundation for the rest of the anonymization process.

Once we have each vertex’s neighborhood, we move to decompose it further. We perform what we call Neighborhood Component Decomposition—dividing the neighborhood into its maximal connected subgraphs or “components.” Think of these components as clusters within a user’s circle, each representing a group of interconnected friends. [see `def extract_neighborhoods()`]
 
## DFS Code
After decomposing the neighborhood into components, the next step is to generate a representative code for each. We use Depth-First Search, or DFS, to traverse each component. During this traversal, we record each edge in a specific format—capturing not only the connection between vertices but also the labels of these vertices. 
The DFS code is ordered carefully. Forward edges, which represent the primary tree structure in the DFS traversal, are prioritized over backward edges. This ordering is critical to ensure that when we compare two components, we have a consistent, lexicographically minimal representation. [see `def getBestComponentDFS()`]
Once we’ve generated the DFS code for each component, we combine the minimum codes of all components to create what we call the Neighborhood Component Code, or NCC, for the vertex. This NCC acts as a fingerprint for the neighborhood, allowing us to compare and eventually match neighborhoods across different vertices.

## Matching and Uniqueness
Delving a little deeper into the DFS representation, the purpose is twofold. First, it provides a standardized way to capture the structure of a neighborhood. Second, it resolves the uniqueness problem that can arise during anonymization. By having a lexicographically best DFS code, we can reliably determine whether two neighborhood components are identical—or, in technical terms, isomorphic.
 
## Neighborhoods Anonymization Process
The goal is to modify the network such that each vertex’s neighborhood becomes indistinguishable from that of at least k–1 other vertices. We process vertices in descending order of their neighborhood size—starting with those that have the most connections. For each “seed” vertex, we identify a group of candidate vertices that exhibit the most similar neighborhood structures based on cost function. [see `def cost()`]
The function uses three parameters: α, β, and γ. Each parameter weighs a different aspect of the modification:  

- α is associated with the normalized certainty penalty, which measures the loss from label generalization.
- β corresponds to the cost of adding edges.  
- γ accounts for the number of vertices that need to be linked in the anonymization process.
 
## Neighborhoods Isomorphism
Achieving k-anonymity isn’t just about grouping similar vertices—it’s also about ensuring that their neighborhoods are truly identical. This is where neighborhood isomorphism comes in. Once candidate vertices are grouped, we must transform their neighborhood structures so that they become isomorphic, meaning they have identical topologies. [see `def make_isomorphic()`]
There are several key steps involved in this transformation:

1. Vertex Addition: Sometimes, to balance the two neighborhoods, it’s necessary to add extra vertices. This ensures that both neighborhoods have the same number of vertices, which is a prerequisite for isomorphism.   [see `def addVertexToComponent()`]
2. Label Generalization: Even if the structure is similar, labels might differ. In such cases, we apply label generalization techniques to assign a common, more generic label to mismatched nodes.  [see `def get_best_generalization_label()`]
3. Edge Alignment: Finally, we compare the adjacency matrices of the two neighborhoods. If one 
neighborhood has an edge where the other does not, we add that edge to achieve alignment. This process can be iterative until both neighborhoods become identical. [NOTE: In our implementation, unlike the paper, we use adjacency matrices instead of BFS.]
 
## Evaluation
In our case, the algorithm we developed has an exponential computational complexity. While this makes it infeasible to test on large real-world datasets, it provides important insights into the underlying mechanics of the approach.
In our experimental evaluation, we measured several key metrics:

- Edges Added: How many new connections were introduced to achieve isomorphism.  
- Labels Anonymized: The extent of label generalization needed.
- Runtime: How long the anonymization process took.

# Getting Started

## Syntetic data generator

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