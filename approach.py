from graph import Graph, Node

class Anonymization(Graph):

    
    def __init__(self, G: Graph):
        self.G = G
        self.G_prime = self.G
        self._neighborhoods = {}

    def extract_components(self,node):
        neighbors = [self.G.getNode(neighbor_id) for neighbor_id in node.edges]
        components = []
        def dfs(current, component):
            if current.Visited:
                return
            current.Visited = True
            component.append(current)
            for neighbor_id in current.edges:
                neighbor = self.G.getNode(neighbor_id)
                if neighbor in neighbors and not neighbor.Visited:
                    dfs(neighbor, component)

        for n in self.G.N:
            n.Visited = False

        for neighbor in neighbors:
            if not neighbor.Visited:
                component = []
                dfs(neighbor, component)
                components.append(component)

        return components
        

    def get_dfs_code(self, component):
        component = sorted(component, key=lambda x: x.node_id)
        edges = []
        for node in component:
            for neighbor_id in node.edges:
                if neighbor_id in {neighbor.node_id for neighbor in component}:
                    edge = (node.node_id, neighbor_id, node.label, self.G.getNode(neighbor_id).label)
                    reverse_edge = (neighbor_id, node.node_id, self.G.getNode(neighbor_id).label, node.label)
                    if edge not in edges and reverse_edge not in edges:
                        edges.append(edge)

        edges = sorted(edges)  # Sort edges lexically
        dfs_code = []
        for edge in edges:
            dfs_code.append(f"({edge[0]},{edge[1]},{edge[2]},{edge[3]})")
        
        return ''.join(dfs_code)

    def extract_neighborhoods(self):
        for node in self.G_prime.N:
            components = self.extract_components(node)
            
            # Get the DFS code for each component and sort them
            coded_components = [self.get_dfs_code(component) for component in components]
            coded_components.sort(key=lambda x: (len(x), x))  # Sort by length and lexically
            
            # Store the Neighborhood Component Code (NCC)
            self._neighborhoods[node] = tuple(coded_components)
            
    
    def nodes_sorted_by_neighborhood_size(self):
        return sorted(self._neighborhoods, key=lambda x: len(self._neighborhoods[x]))

    
    
    
    
 