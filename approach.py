from graph import Graph, Node, Neighborhood

class Anonymization(Graph):

    
    def __init__(self, G: Graph, k: int):
        self.G = G
        self.G_prime = self.G
        self.k = k
        self.anonymized_groups = []
        self.label_hierarchy = {
            "*": 3,  # Root (most general)
            "Professional": 2,
            "Student": 1,
        }

    def printAllNodes(self):
      for v in self.G_prime.N:
        print(v)
      print("\n")
        
    def printAllNcc(self):
      for vertex, ncc in self.G_prime.neighborhoods.items():
         print(f"  \nVertex {vertex.node_id}:")
         for i, comp in enumerate(ncc.NCC):
            print(f"  C{i + 1}:")
            for edges in comp:
                print(f"    {edges}")
                
    
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
        
        return self.label_hierarchy.get(label, 0)  # Default to 0 if the label is missing
    
    def get_best_generalization_label(self,label1, label2):
        
        """
        Retrieve the best generalization label between two labels.

        Args:
            label1 (str): The first label.
            label2 (str): The second label.

        Returns:
            str: The best generalization label between the two input labels.
        """

        level1 = self.get_generalization_level(label1)
        level2 = self.get_generalization_level(label2)

        if level1 > level2:
            return label1
        elif level2 > level1:
            return label2
        else:
            if level1 == level2:
                return next(key for key, value in self.label_hierarchy.items() if value == 1)
            return label1 if len(label1) < len(label2) else label2

        
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

            # Create a list of tuples (component, DFS code)
            component_to_dfs = []
            for component in components:
                dfs_code = self.getBestComponentDFS(component)
                # Sort the DFS code edges based on generalization and attributes
                dfs_code.sort(key=generalized_sort_key)
                component_to_dfs.append((component, dfs_code))

            # Sort the list of (component, DFS code) by neighborhood component order
            component_to_dfs.sort(key=lambda item: (
                len({edge[0] for edge in item[1]}.union({edge[1] for edge in item[1] if edge[1] is not None})),  # Size of the component
                1 if len(item[1]) == 1 and any(edge[1] is None for edge in item[1]) else 0,  # Single-node component handling
                [(edge[2] if edge[2] is not None else '', edge[3] if edge[3] is not None else '', edge[0], edge[1]) for edge in item[1]]  # Lexicographical order of DFS code
            ))

            # Separate the sorted components and their DFS codes
            sorted_components = [item[0] for item in component_to_dfs]
            sorted_NCC = [item[1] for item in component_to_dfs]

            # Store the neighborhood using the updated mapping
            neighborhood = Neighborhood(sorted_components, sorted_NCC)
            self.G_prime.neighborhoods[node] = neighborhood
        
        print("** END NEIGHBORHOODS EXTRACTION AND CODING **")

          
            
    
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
        # Retrieve Neighborhood objects for nodes u and v
        neighborhood_u = self.G_prime.neighborhoods[u]
        neighborhood_v = self.G_prime.neighborhoods[v]

        # Initialize costs
        label_cost = 0
        edge_cost = 0
        vertex_addition_cost = 0

        # Track matched components by their indices
        matched_u = set()
        matched_v = set()

        # Greedy matching of components using DFS codes
        for i, dfs_u in enumerate(neighborhood_u.NCC):
            best_match_cost = float('inf')
            best_match_idx = None

            for j, dfs_v in enumerate(neighborhood_v.NCC):
                if j in matched_v:  # Skip already matched components
                    continue

                # Compute the cost of matching the two components
                match_cost = self.match_components_cost(dfs_u, dfs_v)

                # Track the best match
                if match_cost < best_match_cost:
                    best_match_cost = match_cost
                    best_match_idx = j

            # If a match is found, update costs and mark components as matched
            if best_match_idx is not None:
                matched_u.add(i)
                matched_v.add(best_match_idx)
                label_cost += best_match_cost

        # Handle unmatched components in u
        for i, dfs_u in enumerate(neighborhood_u.NCC):
            if i not in matched_u:
                label_cost += self.generalization_cost(dfs_u)
                edge_cost += len(dfs_u)  # Edges needed to link this unmatched component

        # Handle unmatched components in v
        for j, dfs_v in enumerate(neighborhood_v.NCC):
            if j not in matched_v:
                label_cost += self.generalization_cost(dfs_v)
                edge_cost += len(dfs_v)  # Edges needed to link this unmatched component

        # Compute vertex addition cost
        size_u = sum(len(component) for component in neighborhood_u.components)
        size_v = sum(len(component) for component in neighborhood_v.components)
        vertex_addition_cost = abs(size_u - size_v)

        # Total cost calculation
        total_cost = alpha * label_cost + beta * edge_cost + gamma * vertex_addition_cost
        return total_cost


    def match_components_cost(self, dfs_u, dfs_v):
        """
        Compute the cost of matching two components using their DFS codes.

        Args:
            dfs_u (list): First component's DFS code.
            dfs_v (list): Second component's DFS code.

        Returns:
            float: The cost of matching the two components.
        """
        cost = 0
        matched_v = set()  # Track matched edges in dfs_v

        # Greedy matching of edges within the components
        for edge_u in dfs_u:
            best_match_cost = float('inf')
            for edge_v in dfs_v:
                if edge_v in matched_v:  # Skip already matched edges
                    continue

                # Compute label cost and degree cost for matching edges
                if edge_u[1] is None and edge_v[1] is None:
                    label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                    vertex_addition_cost = 0
                    match_cost = label_cost + vertex_addition_cost
                elif edge_u[1] is None or edge_v[1] is None:
                    label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                    vertex_addition_cost = 1
                    match_cost = label_cost + vertex_addition_cost
                else:
                    label_cost = self.ncp(edge_u[2], edge_v[2]) + self.ncp(edge_u[3], edge_v[3])
                    degree_cost = abs(len(self.G_prime.getNode(edge_u[1]).edges) - len(self.G_prime.getNode(edge_v[1]).edges))
                    match_cost = label_cost + degree_cost

                # Track the best match
                if match_cost < best_match_cost:
                    best_match_cost = match_cost

            # Add the best match cost to the total cost
            cost += best_match_cost

        return cost


    def generalization_cost(self, dfs_code):
        """
        Compute the generalization cost for a single component's DFS code.

        Args:
            dfs_code (list): DFS code of the component.

        Returns:
            float: The generalization cost for the component.
        """
        cost = 0
        for edge in dfs_code:
            cost += self.ncp(edge[2], '*') + self.ncp(edge[3], '*')  # Generalize to the most abstract label
        return cost
    
    def anonymize_neighborhoods(self, candidate_vertices):
        """
        Anonymize the neighborhoods of the given candidate vertices.

        Args:
            candidate_vertices (list[Node]): List of nodes including SeedVertex and its CandidateSet.
        """
        # Step 1: Extract neighborhoods for the candidate vertices
        neighborhoods = {v: self.G_prime.neighborhoods[v] for v in candidate_vertices}

        # Seed Vertex's neighborhood
        seed_vertex, seed_neighborhood = next(iter(neighborhoods.items()))
       

        # Iterate over Candidate Set's neighborhoods
        for candidate_vertex, candidate_neighborhood in list(neighborhoods.items())[1:]:
            print(f"\n\n** Start Anonymous {seed_vertex.node_id} and {candidate_vertex.node_id} **")
            matched_seed = set()
            matched_candidate = set()

            # Step 2.1: Perfectly match components
            for i, (seed_comp, seed_dfs) in enumerate(zip(seed_neighborhood.components, seed_neighborhood.NCC)):
                for j, (candidate_comp, candidate_dfs) in enumerate(zip(candidate_neighborhood.components, candidate_neighborhood.NCC)):
                    if j in matched_candidate:
                        continue
                    if len(seed_dfs) == len(candidate_dfs) and all(
                        edge_s[2:] == edge_c[2:] for edge_s, edge_c in zip(seed_dfs, candidate_dfs)
                    ):
                        matched_seed.add(i)
                        matched_candidate.add(j)
                        break

            # Step 2.2: Handle unmatched components
            unmatched_seed = [
                (seed_neighborhood.components[i], seed_neighborhood.NCC[i])
                for i in range(len(seed_neighborhood.NCC)) if i not in matched_seed
            ]
            unmatched_candidate = [
                (candidate_neighborhood.components[j], candidate_neighborhood.NCC[j])
                for j in range(len(candidate_neighborhood.NCC)) if j not in matched_candidate
            ]
        
            for seed_comp, seed_dfs in unmatched_seed:
                best_match_cost = float('inf')
                best_match = None

                for candidate_comp, candidate_dfs in unmatched_candidate:
                    match_cost = self.match_components_cost(seed_dfs, candidate_dfs)
                    if match_cost < best_match_cost:
                        best_match_cost = match_cost
                        best_match = (candidate_comp, candidate_dfs)

                if best_match:
                    candidate_comp, candidate_dfs = best_match
                    self.make_isomorphic(seed_comp, candidate_comp, seed_vertex, candidate_vertex)
                    self.printAllNodes()
                    unmatched_candidate.remove(best_match)
                    unmatched_seed.remove((seed_comp, seed_dfs))
                    
            # Step 2.3: Handle remaining unmatched components
            if unmatched_seed:
                for seed_comp, seed_dfs in unmatched_seed:
                    self.make_isomorphic(seed_comp, [], seed_vertex, candidate_vertex)
                
            if unmatched_candidate:
                for candidate_comp, candidate_dfs in unmatched_candidate:
                    self.make_isomorphic([], candidate_comp, seed_vertex, candidate_vertex)

         # Print all NCCs in a pretty way
        for vertex in candidate_vertices:
            neighborhood = self.G_prime.neighborhoods[vertex]
            print(f"\nNeighborhood for vertex {vertex.node_id}:")
            for i, ncc in enumerate(neighborhood.NCC):
                print(f"  Component {i + 1}:")
                for edge in ncc:
                    print(f"    {edge}")

        self.extract_neighborhoods()
        
        # Print all NCCs in a pretty way
        for vertex in candidate_vertices:
            neighborhood = self.G_prime.neighborhoods[vertex]
            print(f"\nNeighborhood for vertex {vertex.node_id}:")
            for i, ncc in enumerate(neighborhood.NCC):
                print(f"  Component {i + 1}:")
                for edge in ncc:
                    print(f"    {edge}")
        # Check if all NCCs are equal in candidate vertices
        first_ncc = None
        for vertex in candidate_vertices:
            neighborhood = self.G_prime.neighborhoods[vertex]
            if first_ncc is None:
                first_ncc = neighborhood.NCC
            else:
                for comp1, comp2 in zip(first_ncc, neighborhood.NCC):
                    if len(comp1) != len(comp2):
                        raise ValueError("Not all NCCs are equal in candidate vertices.")
                    for edge1, edge2 in zip(comp1, comp2):
                        if edge1[2:] != edge2[2:]:
                            raise ValueError("Not all NCCs are equal in candidate vertices.")


   
    def make_isomorphic(self, comp_v, comp_u, seed_vertex, candidate_vertex):
        """
        Make two components isomorphic by modifying their structure directly.

        Args:
            comp_v (list[Node]): Component from neighborhood `v` (list of Node objects).
            comp_u (list[Node]): Component from neighborhood `u` (list of Node objects).
        """

        def find_starting_nodes():
            """
            Find a pair of starting nodes to initiate BFS.
            Tries to match nodes with the same degree and label.
            """
            candidates = []
            for node_v in comp_v:
                for node_u in comp_u:
                    
                    if node_v == node_u:
                        return node_v, node_u
                    
                    if node_v.label == node_u.label and len(node_v.getEdgesInComponent(comp_v)) == len(node_u.getEdgesInComponent(comp_u)):
                        candidates.append((node_v, node_u))

            # If multiple candidates, choose the pair with the highest degree
            if candidates:
                return max(candidates, key=lambda pair: len(pair[0].edges))

            # If no exact match, relax the matching criteria
            best_pair = None
            best_cost = float('inf')
            
            max_degree_v = max((len(node.getEdgesInComponent(comp_v)) for node in comp_v), default=0)
            max_degree_u = max((len(node.getEdgesInComponent(comp_u)) for node in comp_u), default=0)
            max_degree = max_degree_v + max_degree_u
            
            for node_v in comp_v:
                for node_u in comp_u:
                    # Normalize degree difference
                    degree_diff = abs(len(node_v.getEdgesInComponent(comp_v)) - len(node_u.getEdgesInComponent(comp_u)))
                    normalized_degree_diff = degree_diff / max_degree if max_degree > 0 else 0

                    # Normalize NCP cost (if needed)
                    ncp_cost = self.ncp(node_v.label, node_u.label)
                    # Assume max_possible_ncp is known or calculate it dynamically
                    normalized_ncp_cost = ncp_cost  # Add normalization logic if needed

                    # Calculate total cost
                    total_cost = normalized_degree_diff + normalized_ncp_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_pair = (node_v, node_u)

            return best_pair

        def bfs_and_match(node_v, node_u, comp_v, comp_u):
            """
            Perform BFS on both components to make them structurally similar.
            Args:
                node_v (Node): Starting node in component `comp_v`.
                node_u (Node): Starting node in component `comp_u`.
                comp_v (list[Node]): Component from neighborhood `v`.
                comp_u (list[Node]): Component from neighborhood `u`.
            """
            queue_u = [node_u]
            queue_v = [node_v]
            visited_u = set()
            visited_v = set()

            while queue_u and queue_v:
                current_u = queue_u.pop(0)
                current_v = queue_v.pop(0)

                # Mark nodes as visited
                visited_u.add(current_u.node_id)
                visited_v.add(current_v.node_id)

                # Match labels
                current_u.label = current_v.label = self.get_best_generalization_label(current_u, current_v)

                # Get neighbors in the component
                neighbors_u = set(current_u.getEdgesInComponent(comp_u)) - visited_u
                neighbors_v = set(current_v.getEdgesInComponent(comp_v)) - visited_v

                # Balance the number of neighbors
                while len(neighbors_u) < len(neighbors_v):
                    new_neighbor = addVertexToComponent(current_u, current_v, comp_u, candidate_vertex)
                    if new_neighbor:
                        neighbors_u.add(new_neighbor)
                    else:
                        break

                while len(neighbors_v) < len(neighbors_u):
                    new_neighbor = addVertexToComponent(current_v, current_u, comp_v, seed_vertex)
                    if new_neighbor:
                        neighbors_v.add(new_neighbor)
                    else:
                        break

                # Add neighbors to the queue for further traversal
                for neighbor_id in neighbors_u:
                    neighbor_node = self.G_prime.getNode(neighbor_id)
                    if neighbor_node and neighbor_id not in visited_u:
                        queue_u.append(neighbor_node)

                for neighbor_id in neighbors_v:
                    neighbor_node = self.G_prime.getNode(neighbor_id)
                    if neighbor_node and neighbor_id not in visited_v:
                        queue_v.append(neighbor_node)
                    
                
                
        def addVertexToComponent(cur_component_vertex, node_to_be_matched ,component, owning_vertex):
            candidates = [
                node for node in self.G_prime.N 
                if not node.Anonymized 
                if node.node_id != candidate_vertex.node_id
                if node.node_id != seed_vertex.node_id
                and node.node_id not in comp_u
                and node.node_id not in comp_v
            ]
            
            if candidates:
                candidates.sort(key=lambda n: (len(n.edges), self.ncp(node_to_be_matched.label, n.label))) 
                selected = candidates[0]
            else:
                candidates = [
                    node for node in self.G_prime.N 
                    if node.node_id != candidate_vertex.node_id
                    if node.node_id != seed_vertex.node_id
                    and node.node_id not in comp_u
                    and node.node_id not in comp_v
                ]
                if candidates:
                    candidates.sort(key=lambda n: (len(n.edges), self.ncp(node_to_be_matched.label, n.label)))  
                    selected = candidates[0]
                    anonymized_group = next((group for group in self.anonymized_groups if selected in group), None)
                    if anonymized_group:
                        for member in anonymized_group:
                            member.Anonymized = False
                        self.anonymized_groups.remove(anonymized_group)
                else:
                    raise ValueError("No more candidates available.")
            component.append(selected)
            selected.addEdge(cur_component_vertex.node_id)
            cur_component_vertex.addEdge(selected.node_id)
            selected.addEdge(owning_vertex.node_id)
            owning_vertex.addEdge(selected.node_id)
            
            return selected.node_id
        
        def addVertexToEmptyComponent(component, owning_vertex):
            candidates = [
                node for node in self.G_prime.N 
                if not node.Anonymized 
                if node.node_id != candidate_vertex.node_id
                if node.node_id != seed_vertex.node_id
            ]
            if candidates:
                candidates.sort(key=lambda n: (len(n.edges))) 
                selected = candidates[0]
            else:
                candidates = [
                    node for node in self.G_prime.N 
                    if node.node_id != candidate_vertex.node_id
                    if node.node_id != seed_vertex.node_id
                    and node.node_id not in comp_u
                    and node.node_id not in comp_v
                ]
                if candidates:
                    candidates.sort(key=lambda n: (len(n.edges)))  
                    selected = candidates[0]
                    anonymized_group = next((group for group in self.anonymized_groups if selected in group), None)
                    if anonymized_group:
                        for member in anonymized_group:
                            member.Anonymized = False
                        self.anonymized_groups.remove(anonymized_group)
                else:
                    raise ValueError("No more candidates available.")
            component.append(selected)
            selected.addEdge(owning_vertex.node_id)
            owning_vertex.addEdge(selected.node_id)
            
        if not comp_v:
            addVertexToEmptyComponent(comp_v, seed_vertex)
        if not comp_u:
            addVertexToEmptyComponent(comp_u, candidate_vertex)   
            
        # Step 1: Find starting nodes for BFS
        start_v, start_u = find_starting_nodes()
        
        # Mark all nodes in both components as not visited
        for node in comp_v:
            node.visited = False
        for node in comp_u:
            node.visited = False

        # Step 2: Perform BFS and modify components
        bfs_and_match(start_v, start_u, comp_v, comp_u) 

    


        
        

       


    






 