from graph import Graph

class Anonymization(Graph):
    
    def __init__(self, G: Graph):
        self.G = G
        self.G_prime = self.G


    def extract_neighborhood(self, node_id):
        """
        Extracts the neighborhood of the given node_id.
        """
        node = self.G.getNode(node_id)
        if not node:
            return None

        neighborhood = Graph()
        neighbors = node.edges

        for neighbor_id in neighbors:
            neighbor_node = self.G.getNode(neighbor_id)
            if neighbor_node:
                neighborhood.addVertex(Node(neighbor_id, neighbor_node.label)) # ha senso ricrearsi una classe Node?

        for neighbor_id in neighbors:
            neighbor_node = self.graph.getNode(neighbor_id)
            if neighbor_node:
                for edge in neighbor_node.edges:
                    if edge in neighbors:  # Ensure the edge is within the neighborhood
                        neighborhood.getNode(neighbor_id).addEdge(edge)

        return neighborhood

    
    
    
    
    # def extractNeighborhood(self, node_id):
    #     """Extract the neighborhood (induced subgraph) of a node."""
    #     node = self.G.getNode(node_id)
    #     if not node:
    #         return []

    #     neighbors = node.edges
    #     subgraph_nodes = [self.G.getNode(nid) for nid in neighbors]

    #     return [n for n in subgraph_nodes if n is not None]

    # def decomposeNeighborhood(self, neighborhood):
    #      """Decompose a neighborhood into connected components."""
    #      components = []
    #      visited = set()

    #      def dfs(node, component):
    #         if node.node_id in visited:
    #             return
    #         visited.add(node.node_id)
    #         component.append(node.node_id)
    #         for neighbor_id in node.edges:
    #             neighbor = self.G.getNode(neighbor_id)  
    #             if neighbor and neighbor.node_id not in visited:
    #                 dfs(neighbor, component)

    #      for node in neighborhood:
    #         if node.node_id not in visited:
    #             component = []
    #             dfs(node, component)
    #             components.append(component)

    #      return components
