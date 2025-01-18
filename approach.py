from graph import Graph, Node

class Anonymization(Graph):

    
    def __init__(self, G: Graph):
        self.G = G
        self.G_prime = self.G
        self._neighborhoods = {}

    def extract_components(self,node):
        neighbors = [self.G_prime.getNode(neighbor_id) for neighbor_id in node.edges]
        components = []
        def dfs(current, component):
            if current.Visited: 
                return
              
            current.Visited = True
            component.append(current)
            for neighbor_id in current.edges:
                neighbor = self.G_prime.getNode(neighbor_id)
                if neighbor in neighbors and not neighbor.Visited:
                    dfs(neighbor, component)

        for n in self.G_prime.N:
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
                neighbor = self.G_prime.getNode(neighbor_id)
                if neighbor.node_id not in visited:
                    stack.append((neighbor, current_node))

        return dfs_result

    
    def getBestComponentDFS(self, component):
        """per ogni nodo della dela componente facciamo una dfs e controlliamo quella lessicamente migliore"""
        # Initialize visited to False for each component node
        if len(component) == 1:
            node = component[0]
            return [(node.node_id, None, node.label, None)]

        for node in component:
            node.visited = False

        # Initialize R as a list of list of sets of Result between FW and BW
        R = []
        
        for node in component:
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
                        R_aux.append((e, node.node_id, self.G_prime.getNode(e).label, node.label))

            R_aux.sort(key=lambda x: (x[0], x[1])) # <=
            R.append(R_aux)

        R.sort(key=lambda x: (len(x), [(edge[2], edge[3], edge[0], edge[1]) for edge in x])) #restituiamo la best DFS per ogni C 
 
        return R[0] # the head is the first lessically correct
        

    def get_generalization_level(self,label):
        """
        Retrieve the generalization level of a label from the hierarchy.

        Args:
            label (str): The label whose generalization level is required.

        Returns:
            int: The generalization level of the label (higher is more general).
        """
        label_hierarchy = {
            "*": 3,  # Root (most general)
            "Professional": 2,
            "Student": 1,
        }
        return label_hierarchy.get(label, 0)  # Default to 0 if the label is missing

        
    def extract_neighborhoods(self):

        def generalized_sort_key(edge):
            """
            Sorting key for edges based on label generalization and edge attributes.

            Args:
                edge (tuple): An edge represented as (id1, id2, l1, l2).

            Returns:
                tuple: Sorting key (generalized l1, generalized l2, id1, id2).
            """
            # Use the hierarchy to generalize labels if applicable
            label1, label2 = edge[2], edge[3]
            return (self.get_generalization_level(label1),  # Generalized level of label1
                    self.get_generalization_level(label2),  # Generalized level of label2
                    edge[0], edge[1])  # Edge IDs for tie-breaking

        for node in self.G_prime.N:
            components = self.extract_components(node)

            NCC = []
            # Get the DFS code for each component of the node and sort them
            for component in components:
                dfs_code = self.getBestComponentDFS(component)
                
                # Sort the DFS code edges based on generalization and attributes
                dfs_code.sort(key=generalized_sort_key)
                
                NCC.append(dfs_code)

            # Sort canonical labels by the defined neighborhood component order
            NCC.sort(key=lambda x: (len(x), sum(len(edge) for edge in x), x))
            
            # Store the sorted Neighborhood Component Code (NCC)
            self._neighborhoods[node] = NCC
        
    
    def ncp(self, label1, label2):
        """
        Compute the Normalized Certainty Penalty (NCP) for label generalization.
        Use the existing get_generalization_level method to determine levels.

        Args:
            label1 (str): The first label.
            label2 (str): The second label.

        Returns:
            float: The NCP value, normalized by the max hierarchy level.
        """
        # Retrieve the generalization levels using the existing method
        level1 = self.get_generalization_level(label1)
        level2 = self.get_generalization_level(label2)

        # If the labels are identical, no penalty
        if level1 == level2:
            return 0

        # Normalize the difference by the maximum level
        max_level = 3  # The max hierarchy level (assumes hierarchy levels are static)
        return abs(level1 - level2) / max_level

    def cost(self, u, v, alpha, beta, gamma):
        """
        Compute the anonymization cost between two neighborhoods using greedy matching.

        Args:
            u (Node): First node.
            v (Node): Second node.
            alpha (float): Weight for label generalization.
            beta (float): Weight for edge addition.
            gamma (float): Weight for vertex addition.

        Returns:
            float: The calculated anonymization cost.
        """
        # Get the NCCs for both nodes
        ncc_u = self._neighborhoods[u]
        ncc_v = self._neighborhoods[v]

        # Initialize costs
        label_cost = 0
        edge_cost = 0
        vertex_addition_cost = 0

        # Mark components as matched
        matched_u = set()
        matched_v = set()

        # Greedy matching of components
        for i, comp_u in enumerate(ncc_u):
            best_match_cost = float('inf')
            best_match_idx = None

            for j, comp_v in enumerate(ncc_v):
                if j in matched_v:  # Skip already matched components
                    continue

                # Compute the cost of matching comp_u and comp_v
                match_cost = self.match_components_cost(comp_u, comp_v)

                # Track the best match
                if match_cost < best_match_cost:
                    best_match_cost = match_cost
                    best_match_idx = j

            # If a match is found, update costs and mark components as matched
            if best_match_idx is not None:
                matched_u.add(i)
                matched_v.add(best_match_idx)
                label_cost += best_match_cost  # Add the label cost of this match

        # Handle unmatched components in u
        for i, comp_u in enumerate(ncc_u):
            if i not in matched_u:
                label_cost += self.generalization_cost(comp_u)
                edge_cost += len(comp_u)  # Edges needed to link this unmatched component

        # Handle unmatched components in v
        for j, comp_v in enumerate(ncc_v):
            if j not in matched_v:
                label_cost += self.generalization_cost(comp_v)
                edge_cost += len(comp_v)  # Edges needed to link this unmatched component

        # Compute vertex addition cost
        size_u = sum(len(comp) for comp in ncc_u)
        size_v = sum(len(comp) for comp in ncc_v)
        vertex_addition_cost = abs(size_u - size_v)

        # Total cost calculation
        total_cost = alpha * label_cost + beta * edge_cost + gamma * vertex_addition_cost
        return total_cost
    
    def match_components_cost(self, comp_u, comp_v):
        """
        Compute the cost of matching two components.

        Args:
            comp_u (list): First component (list of DFS codes).
            comp_v (list): Second component (list of DFS codes).

        Returns:
            float: The cost of matching the two components.
        """
        cost = 0

        # Greedy matching of vertices within the components
        matched_v = set()
        for edge_u in comp_u:
            best_match_cost = float('inf')
            for edge_v in comp_v:
                if edge_v in matched_v:  # Skip already matched edges
                    continue

                # Skip calculation if nodes are missing
                if edge_u[1] is None or edge_v[1] is None:
                    continue

                # Compute the cost of matching edge_u and edge_v
                label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                degree_cost = abs(len(self.G_prime.getNode(edge_u[1]).edges) - len(self.G_prime.getNode(edge_v[1]).edges))
                match_cost = label_cost + degree_cost

                # Track the best match
                if match_cost < best_match_cost:
                    best_match_cost = match_cost

            # Add the best match cost to the total cost
            cost += best_match_cost

        return cost

    def generalization_cost(self, comp):
        """
        Compute the generalization cost for a single component.

        Args:
            comp (list): Component (list of DFS codes).

        Returns:
            float: The generalization cost for the component.
        """
        cost = 0
        for edge in comp:
            cost += self.ncp(edge[2], '*') + self.ncp(edge[3], '*')  # Generalize to the most abstract label
        return cost
    
    def anonymize_neighborhoods(self, candidate_vertices, k):
        """
        Anonymize the neighborhoods of the given candidate vertices.

        Args:
            candidate_vertices (list[Node]): List of nodes including SeedVertex and its CandidateSet.
            k (int): Anonymization requirement parameter.
        """
        # Step 1: Extract neighborhoods of the candidate vertices
        neighborhoods = {v: self._neighborhoods[v] for v in candidate_vertices}

        # Step 2: Greedy matching for each pair of neighborhoods
        while neighborhoods:
            # Pop a neighborhood to anonymize
            v, ncc_v = neighborhoods.popitem()

            for u, ncc_u in list(neighborhoods.items()):
                # Match components in NeighborG(v) and NeighborG(u)
                matched_v = set()
                matched_u = set()

                # Step 2.1: Match perfectly matched components
                for i, comp_v in enumerate(ncc_v):
                    for j, comp_u in enumerate(ncc_u):
                        if j in matched_u:  # Skip already matched components
                            continue
                        if comp_v == comp_u:  # Perfect match based on minimum DFS code
                            matched_v.add(i)
                            matched_u.add(j)
                            break

                # Step 2.2: Match remaining components using greedy matching
                for i, comp_v in enumerate(ncc_v):
                    if i in matched_v:
                        continue
                    best_match_cost = float('inf')
                    best_match_idx = None

                    for j, comp_u in enumerate(ncc_u):
                        if j in matched_u:
                            continue

                        # Compute cost of matching comp_v and comp_u
                        match_cost = self.match_components_cost(comp_v, comp_u)

                        if match_cost < best_match_cost:
                            best_match_cost = match_cost
                            best_match_idx = j

                    # Perform the matching
                    if best_match_idx is not None:
                        matched_v.add(i)
                        matched_u.add(best_match_idx)
                        self.make_isomorphic(comp_v, comp_u, k)

                # Step 2.3: Handle unmatched components
                for i, comp_v in enumerate(ncc_v):
                    if i in matched_v:
                        continue

                    # Add new vertices or edges to anonymize
                    new_vertex = self.select_new_vertex()
                    self.add_vertex_to_component(new_vertex, comp_v)

                for j, comp_u in enumerate(ncc_u):
                    if j in matched_u:
                        continue

                    # Add new vertices or edges to anonymize
                    new_vertex = self.select_new_vertex()
                    self.add_vertex_to_component(new_vertex, comp_u)

            # Mark the vertices in the neighborhoods as anonymized
            for v in candidate_vertices:
                v.Visited = True

        # Step 3: Update the graph and neighborhoods
        self.extract_neighborhoods()

    def make_isomorphic(self, comp_v, comp_u, k):
        """
        Modify components to make them isomorphic by adding vertices or edges.

        Args:
            comp_v (list): Component of NeighborG(v).
            comp_u (list): Component of NeighborG(u).
            k (int): Anonymization requirement parameter.
        """
        size_v = len(comp_v)
        size_u = len(comp_u)

        # Balance the number of vertices
        if size_v < size_u:
            for _ in range(size_u - size_v):
                new_vertex = self.select_new_vertex()
                self.add_vertex_to_component(new_vertex, comp_v)
        elif size_u < size_v:
            for _ in range(size_v - size_u):
                new_vertex = self.select_new_vertex()
                self.add_vertex_to_component(new_vertex, comp_u)

        # Balance edges (if needed, implement a method to match edges)


    def select_new_vertex(self):
        """
        Select a new vertex to be added to a component.

        Returns:
            Node: The selected vertex.
        """
        # Prioritize unanonymized vertices with the smallest degree
        unanonymized = [node for node in self.G_prime.N if not node.Visited]
        if unanonymized:
            return min(unanonymized, key=lambda x: (len(x.edges), self.get_generalization_level(x.label)))

        # If all vertices are anonymized, reuse an anonymized vertex
        anonymized = [node for node in self.G_prime.N if node.Visited]
        return min(anonymized, key=lambda x: len(x.edges))

    def add_vertex_to_component(self, vertex, component):
        """
        Add a vertex to a component by connecting it to other nodes.

        Args:
            vertex (Node): The vertex to be added.
            component (list): The component to modify.
        """
        for edge in component:
            neighbor_id = edge[1]  # Get the neighbor ID from the edge
            neighbor = self.G_prime.getNode(neighbor_id)
            if neighbor:
                neighbor.addEdge(vertex.node_id)
                vertex.addEdge(neighbor_id)
        component.append((vertex.node_id, None, vertex.label, None))
