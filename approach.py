from graph import Graph, Node

class Anonymization(Graph):

    
    def __init__(self, G: Graph):
        self.G = G
        self.G_prime = self.G
        self._neighborhoods = {}

    def extract_components(self,node):
        neighbors = node.edges
        components = {}
        for neighbor_id in neighbors:
            n_node = self.G_prime.getNode(neighbor_id)
            n_neighbors = n_node.edges

    def get_dfs_code(self, components):
        return []

    def extract_neighborhoods(self):
        for node in self.G_prime.N:
            components = self.extract_components(node)
            dfs_code = self.get_dfs_code(components)
            self._neighborhoods[node] = dfs_code
            
    
    def nodes_sorted_by_neighborhood_size(self):
        return sorted(self._neighborhoods, key=lambda x: len(self._neighborhoods[x]), reverse=True)

    
    
    
    
 