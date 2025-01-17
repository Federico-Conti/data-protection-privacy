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
        
    def DFS(self,start_node,component):
        """
        Perform Depth First Search (DFS) starting from a given node.

        Args:
            start_node (Node): The starting node for the DFS.

        Returns:
            List[Tuple[int, int, string, string]]: A list of tuples representing the DFS traversal 
            in the format (id1, id2, l1, l2).
        """
        stack = [(start_node, None)]  # (current_node, parent_node)
        visited = set()
        dfs_result = []

        while stack:
            current_node, parent_node = stack.pop()

            if current_node.node_id in visited:
                continue

            visited.add(current_node.node_id)

            # If there's a parent, record the edge information
            if parent_node is not None:
                dfs_result.append(
                    (parent_node.node_id, current_node.node_id, parent_node.label, current_node.label)
                )

            # Add neighbors to the stack for further exploration
            for neighbor_id in current_node.getEdgesInComponent(component):
                neighbor = self.G.getNode(neighbor_id)
                if neighbor.node_id not in visited:
                    stack.append((neighbor, current_node))

        return dfs_result

    
    def getBestComponentDFS(self, component):
        """per ogni nodo della dela componente facciamo una dfs e controlliamo quella lessicamente migliore"""
        # Initialize visited to False for each component node
        for node in component:
            node.visited = False

        # Initialize R as a list of list of sets of Result between FW and BW
        R = []
        
        for node in component:
            # BW = [] # this tuple are swapped respect FW tuple (id2,id1,l2,l1)

            # Perform DFS on the node 
            """
            DFS--> return a list of tuple (id1,id2,l1,l2)
            
            """
            FW = self.DFS(node,component)
            R_aux = FW
  
           
            # BW step
            """
            
                FW tuple {0,3 ; 3,9 ; 9,6; 6,2} 
                C1 of Vertex1:  Node(id=0, edges=[1, 2, 3])  
                                Node(id=2, edges=[0, 1, 9, 6, 5, 7, 8])  
                                Node(id=9, edges=[2, 5, 3, 1, 4, 6])  
                                Node(id=3, edges=[1, 5, 0, 4, 7, 9])  
                                Node(id=6, edges=[2, 4, 5, 1, 9])]
            
            """
            for node in component:
                for e in node.getEdgesInComponent(component):
                    if not any((node.node_id == tuple[0] and e == tuple[1]) or (node.node_id == tuple[1] and e == tuple[0]) for tuple in R_aux):
                        R_aux.append((e, node.node_id, self.G.getNode(e).label, node.label))

            R_aux.sort(key=lambda x: (x[0], x[1])) # <=
            R.append(R_aux)

        R.sort(key=lambda x: (len(x), [(edge[2], edge[3], edge[0], edge[1]) for edge in x])) #restituiamo la best DFS per ogni C 
 
        return R[0] # the head is the first lessically correct
        
        
    def extract_neighborhoods(self):
       for node in self.G_prime.N:
            components = self.extract_components(node)
            
            # components_str = "- ".join([f"C{i+1}=[{' ; '.join(str(n) for n in comp)}]\n" 
            #                             for i, comp in enumerate(components)])
            # print(f"Node {node.node_id}\n- {components_str}\n")
                

            NCC = []
            # Get the DFS code for each component of the node and sort them, after append the best in NCC
            for c in components:
                
                dfs_code = self.getBestComponentDFS(c)
                NCC.append(dfs_code) #Rimepo la NCC con la DFS migliore per ogni C dello stesso vertice 
            
            NCC.sort(key=lambda x: (len(x), x))  # Sort by length and lexically
            
            # Store the Neighborhood Component Code (NCC)
            self._neighborhoods[node] = NCC
        
    
    def nodes_sorted_by_neighborhood_size(self):
        """
        Sort nodes based on the number of couples (pairs of nodes) in each neighborhood.
        """
        return sorted(
            self._neighborhoods.keys(), 
            key=lambda node_id: sum(len(c) for c in self._neighborhoods[node_id]),
            reverse=True
        )

    
    def cost(self,u,v,alpha,beta,gamma):
        NotImplemented
        
        
    
        
    
  
    
