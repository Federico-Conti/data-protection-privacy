from graph import Graph, Node

class Anonymization(Graph):

    
    def __init__(self, G: Graph):
        self.G = G
        self.G_prime = self.G
        self.anonymized_groups = []
        
    
    def insert_anonymized_group(self, group):
        self.anonymized_groups.append(group)
        
    

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
            '''
            N{          
             0:   [(1, 2, 'Student', Anna), (2, 3, 'Student', Anna), (3, 1, 'Student', Anna)]
             1:   [(2, 3, 'Student', None),(3, 4, 'Student', None),(4, 5, 'Student', None)]
            }
            '''
            # Sort canonical labels by the defined neighborhood component order
            NCC.sort(key=lambda comp: (len({edge[0] for edge in comp}.union({edge[1] for edge in comp if edge[1] is not None})),
                                        1 if len(comp) == 1 and any(edge[1] is None for edge in comp) else 0, 
                                        comp))
            
            # Store the sorted Neighborhood Component Code (NCC)
            self.G_prime.neighborhoods[node] = NCC
            self.G_prime.components_vertexes[node] = components
        
    
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
        ncc_u = self.G_prime.neighborhoods[u]
        ncc_v = self.G_prime.neighborhoods[v]

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
                if edge_u[1] is None and edge_v[1] is None:
                    # Compute cost based on label generalization and vertex addition
                    label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                    vertex_addition_cost = 0  # Penalty for standalone nodes
                    match_cost = label_cost + vertex_addition_cost
                elif edge_u[1] is None or edge_v[1] is None:
                    # Compute cost based on label generalization and vertex addition
                    label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                    vertex_addition_cost = 1  # Penalty for standalone nodes
                    match_cost = label_cost + vertex_addition_cost
                else:
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
    
    def anonymize_neighborhoods(self, candidate_vertices):
        """
        Anonymize the neighborhoods of the given candidate vertices.

        Args:
            candidate_vertices (list[Node]): List of nodes including SeedVertex and its CandidateSet.
        """
        # Step 1: Extract neighborhoods of the candidate vertices
        neighborhoods = {v : self.G_prime.neighborhoods[v] for v in candidate_vertices}

        # Seed Vertex ncc
        _, ncc_v = next(iter(neighborhoods.items()))

        # Candidate Set ncc 
        for _ , ncc_u in list(neighborhoods.items())[1:]:
            # Match components in NeighborG(v) and NeighborG(u)
            matched_v = set()
            matched_u = set()

            # Step 2.1: Match perfectly matched components
            for i, comp_v in enumerate(ncc_v):
                for j, comp_u in enumerate(ncc_u):
                    if j in matched_u:
                        continue
                    if len(comp_v) == len(comp_u) and all(edge_v[2:] == edge_u[2:] for edge_v, edge_u in zip(comp_v, comp_u)):
                        matched_v.add(i)
                        matched_u.add(j)
                        break

            # Step 2.2: Handle unmatched components
            unmatched_v = [comp_v for i, comp_v in enumerate(ncc_v) if i not in matched_v]
            unmatched_u = [comp_u for j, comp_u in enumerate(ncc_u) if j not in matched_u]
            
            
            for comp_v in unmatched_v:  
                best_match_cost = float('inf')
                best_match = None
                for comp_u in unmatched_u:
                    match_cost = self.match_components_cost(comp_v, comp_u)
                    if match_cost < best_match_cost:
                        best_match_cost = match_cost
                        best_match = comp_u

                if best_match:
                    self.make_isomorphic(comp_v, best_match)
                    unmatched_u.remove(best_match)
        
                    
        # Mark the vertices in the neighborhoods as anonymized
        for node in candidate_vertices:
            node.Anonymized = True
        self.extract_neighborhoods() #restract NCC

   
    def make_isomorphic(self, comp_v, comp_u):
        """
        Make two components isomorphic

        Args:
            comp_v (list): First component.
            comp_u (list): Second component.
        """
        # Update the edges in comp_u to match comp_v
        """
        Example:    
            comp_u = [(7, None, 'Brian', None)]
            comp_v = [(6, 4, 'Eva', 'Linda')]
        """
        def add_node_to_component(node_v, node_u, comp_v, comp_u):
            
            # Step 1: Identify candidates (unanonymized vertices in the graph)
            candidates = [node for node in self.G_prime.N if not node.Anonymized and node.node_id != vertex_comp_v and node.node_id != vertex_comp_u  and node.node_id not in node_v and node.node_id not in node_u]
            
            # Step 2: Prioritize by smallest degree
            #NODES_V OR NODES_U IS A SET OF NODES.ID NOT A SINGLE NODE_ID. FIX THE LOGIC.
            for node_v_id in nodes_v:
                candidates.sort(key=lambda n: (len(n.edges), self.ncp(self.G_prime.getNode(node_v_id).label, n.label)))
            
            # Step 3: If no unanonymized vertices are found, fallback to anonymized vertices
            if not candidates:
                candidates = [node for node in self.G_prime.N if node.Anonymized]
                candidates.sort(key=lambda n: (len(n.edges), self.ncp(node_v.label, n.label)))
                
                if candidates:
                    selected = candidates[0]
                    # Mark selected vertex and its group as unanonymized
                    anonymized_group = None
                    for idx, group in enumerate(self.anonymized_groups):
                        if selected in group:
                            anonymized_group = self.anonymized_groups.pop(idx)
                            break
                    if anonymized_group:
                        for member in anonymized_group:
                            member.Anonymized = False
                else:
                    raise ValueError("No suitable candidate found to add to the component.")
            else: 
                # Step 4: Add the selected vertex to the target component
                selected = candidates[0]
                
            for edge in comp_v:
                if edge[0] == node_v.node_id:
                    selected.addEdge(edge[1])
                elif edge[1] == node_v.node_id:
                    selected.addEdge(edge[0])
            comp_u.append((selected.node_id, None, selected.label, None))
            
        
        def add_edge_to_component():
            NotImplemented
            
        def generalize_labels():
            NotImplemented
        
        """
            1. Add missing nodes	
            2. Add edges to make the structure the same
            3. Generalize labels to make them the same if they were not already
        """	
        
        nodes_v = {edge[0] for edge in comp_v}.union({edge[1] for edge in comp_v if edge[1] is not None})
        nodes_u = {edge[0] for edge in comp_u}.union({edge[1] for edge in comp_u if edge[1] is not None})
        
        # Add missing nodes to comp_u to match comp_v
        if len(nodes_v) > len(nodes_u):
            while len(nodes_v) > len(nodes_u):
                add_node_to_component(nodes_v,nodes_u,comp_v,comp_u)
        else:
            while len(nodes_u) > len(nodes_v):
                add_node_to_component()
                
        # Add missing edges to comp_u to match comp_v
        
        

       


    






 