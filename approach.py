from graph import Graph, Node, Neighborhood
from functools import cmp_to_key
import random

class Anonymization(Graph):

    
    def __init__(self, G: Graph, k: int, alpha: float , beta: float , gamma: float ):
        self.G = G
        self.G_prime = self.G
        self.k = k
        self.anonymized_groups = []
        self.label_hierarchy = {
            "*": 3,  # Root (most general)
            "Professional": 2,
            "Student": 1,
        }
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
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
        
        mapping = {}
        counter = 0

        while stack:
            current_node, parent_node = stack.pop()
            
            if current_node.node_id in visited:
                continue
            
            mapping[current_node.node_id] = counter
            counter += 1

            visited.add(current_node.node_id)

            # If there's a parent, record the edge information
            if parent_node is not None:
                dfs_result.append(
                    (mapping[parent_node.node_id], mapping[current_node.node_id], parent_node.label, current_node.label)
                )
                
            neighbors = []
            for neighbor_id in current_node.getEdgesInComponent(component):
                neighbor = self.G_prime.getNode(neighbor_id)
                if neighbor_id not in visited:  
                    neighbors.append(neighbor) 
                else:
                    if not any((mapping[current_node.node_id] == tuple[0] and mapping[neighbor_id] == tuple[1]) or (mapping[neighbor_id] == tuple[0] and mapping[current_node.node_id] == tuple[1]) for tuple in dfs_result):
                        dfs_result.append(
                            (mapping[current_node.node_id], mapping[neighbor_id], current_node.label, neighbor.label)                  
                        )            
            neighbors.sort(key=lambda n: (len(n.getEdgesInComponent(component)), n.label), reverse=True)
            for neighbor in neighbors:
                stack.append((neighbor, current_node))
        return dfs_result

    
    def getBestComponentDFS(self, component):
        """per ogni nodo della dela componente facciamo una dfs e controlliamo quella lessicamente migliore"""
        # Initialize visited to False for each component node
        
        def dfs_edge_comparator(edge1, edge2):
            u1, v1, label_u1, label_v1 = edge1
            u2, v2, label_u2, label_v2 = edge2
            # Rule (1): Both are forward edges
            if u1 < v1 and u2 < v2:
                if v1 != v2:
                    return -1 if v1 < v2 else 1
                return -1 if u1 > u2 else 1 if u1 < u2 else 0

            # Rule (2): Both are backward edges
            if u1 > v1 and u2 > v2:
                if u1 != u2:
                    return -1 if u1 < u2 else 1
                return -1 if v1 < v2 else 1 if v1 > v2 else 0

            # Rule (3): edge1 is forward and edge2 is backward
            if u1 < v1 and u2 > v2:
                return -1 if v1 <= u2 else 1

            # Rule (4): edge1 is backward and edge2 is forward
            if u1 > v1 and u2 < v2:
                return -1 if u1 < v2 else 1

            return 0  # Equal edges
        
        if len(component) == 1:
            node = component[0]
            return [(0, None, node.label, None)]

        for node in component:
            node.visited = False

        # Initialize R as a list of list of sets of Result between FW and BW
        R = []
        
        
        for node in component:
            """
            DFS--> return a list of tuple (id1,id2,l1,l2)
            
            """
            R_aux = self.DFS(node,component)
           
            # BW step
            """
            
                FW tuple {0,3 ; 3,9 ; 9,6; 6,2} 
                C1 of Vertex1:  Node(id=0, edges=[1, 2, 3])  
                                Node(id=2, edges=[0, 1, 9, 6, 5, 7, 8])  
                                Node(id=9, edges=[2, 5, 3, 1, 4, 6])  
                                Node(id=3, edges=[1, 5, 0, 4, 7, 9])  
                                Node(id=6, edges=[2, 4, 5, 1, 9])]
            
            """
           

            R_aux.sort(key=cmp_to_key(dfs_edge_comparator))
            R.append(R_aux)
            

        R.sort(key=lambda x: (len(x), [(edge[2], edge[3],edge[0], edge[1]) for edge in x])) #restituiamo la best DFS per ogni C 
 
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
            if level1 == 0 and level2 == 0:
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

          
            
    
    def compare_ncp(self, label1, label2):
        """
        Compute the Normalized Certainty Penalty (NCP) for label generalization.
        Use the existing get_generalization_level method to determine levels.

        Args:
            label1 (str): The first label.
            label2 (str): The second label.

        Returns:
            float: The NCP value, normalized by the max hierarchy level.
        """
        if not label1:
            return self.ncp(label1)
        if not label2:
            return self.ncp(label2)
        
        level1 = self.get_generalization_level(label1)
        level2 = self.get_generalization_level(label2)

        if level1 == level2:
            return 0
        max_level = self.get_generalization_level("*")
        return abs(level1 - level2) / max_level

        
        
    def ncp(self, label):
        """
        Compute the Normalized Certainty Penalty (NCP) for label generalization.

        Args:
            label (str): The label to compute the NCP for.

        Returns:
            float: The NCP value for the label.
        """
        level = self.get_generalization_level(label)
        max_level = self.get_generalization_level("*")
        return level / max_level
        
    def cost_aux(self, neighborhood_u, comp_u, dfs_u, neighborhood_v, comp_v, dfs_v, alpha, beta, gamma):
        
        comp_edge_cost = abs(neighborhood_u.getNumberOfEdges(comp_u) - neighborhood_v.getNumberOfEdges(comp_v))
        
        comp_vertex_addition_cost = abs(len(comp_u) - len(comp_v))
        
        comp_label_cost = sum(self.compare_ncp(edge_u[2], edge_v[2]) + self.compare_ncp(edge_u[3], edge_v[3]) for edge_u, edge_v in zip(dfs_u, dfs_v))
        
        match_cost = alpha * comp_label_cost + beta * comp_edge_cost + gamma * comp_vertex_addition_cost

        return match_cost
        
    def cost(self, neighborhood_v, neighborhood_u, alpha, beta, gamma):
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
        total_match_cost = 0

        # Track matched components by their indices
        matched_u = []
        matched_v = []

        # Greedy matching of components using DFS codes
        best_match_cost = float('inf')
        best_match_idx_i = None
        best_match_idx_j = None
        
        while len(matched_u) < len(neighborhood_u.components) and len(matched_v) < len(neighborhood_v.components):
            for i, (comp_u, dfs_u) in enumerate(zip(neighborhood_u.components, neighborhood_u.NCC)):

                for j, (comp_v, dfs_v) in enumerate(zip(neighborhood_v.components, neighborhood_v.NCC)):
                    
                    match_cost = self.cost_aux(neighborhood_u, comp_u,dfs_u, neighborhood_v, comp_v,dfs_v, alpha, beta, gamma)
                    
                    # Track the best match
                    if match_cost < best_match_cost:
                        best_match_cost = match_cost
                        best_match_idx_j = j
                        best_match_idx_i = i

            # If a match is found, update costs and mark components as matched
            if best_match_idx_j is not None :
                matched_u.append(best_match_idx_i)
                matched_v.append(best_match_idx_j)
                total_match_cost += total_match_cost

        # Handle unmatched components
        for i, dfs_u in enumerate(neighborhood_u.NCC):
            if i not in matched_u:
                total_match_cost += self.generalization_cost(dfs_u)*alpha
        
       # Handle unmatched components
        for i, dfs_v in enumerate(neighborhood_v.NCC):
            if i not in matched_v:
                total_match_cost += self.generalization_cost(dfs_v)*alpha

        return total_match_cost

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
            cost += self.compare_ncp(edge[2], '*') + self.compare_ncp(edge[3], '*')  # Generalize to the most abstract label
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
        seed_vertex, seed_neighborhood = list(neighborhoods.items())[0]

        # Iterate over Candidate Set's neighborhoods
        candidate_vertex, candidate_neighborhood = list(neighborhoods.items())[1]
      
      
      
        best_label = self.get_best_generalization_label(seed_vertex.label, candidate_vertex.label)
        seed_vertex.label = best_label
        candidate_vertex.label = best_label
        
        
        unmatched_candidate = True
        unmatched_seed = True
        
        while unmatched_seed or unmatched_candidate:
        
            matched_seed = set()
            matched_candidate = set()

            # Step 2.1: Perfectly match components
            for i, (seed_comp, seed_dfs) in enumerate(zip(seed_neighborhood.components, seed_neighborhood.NCC)):
                for j, (candidate_comp, candidate_dfs) in enumerate(zip(candidate_neighborhood.components, candidate_neighborhood.NCC)):
                    if j in matched_candidate:
                        continue
                    if len(seed_dfs) == len(candidate_dfs) and len(seed_comp) == len(candidate_comp) and all(
                        edge_s == edge_c for edge_s, edge_c in zip(seed_dfs, candidate_dfs)
                    ):
                        matched_seed.add(i)
                        matched_candidate.add(j)
                        print(f"Matched component {i + 1} in seed with component {j + 1} in candidate.")
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
            
            if not unmatched_seed and not unmatched_candidate:
                break

            if not unmatched_seed:
                for component in unmatched_candidate:
                    unmatched_seed.append(([],[]))
        
            if not unmatched_candidate:
                for component in unmatched_seed:
                    unmatched_candidate.append(([],[]))
            
            best_candidate_match = None
            best_seed_match = None        
            
            for seed_comp, seed_dfs in unmatched_seed:
                best_match_cost = float('inf')

                for candidate_comp, candidate_dfs in unmatched_candidate:
                    match_cost = self.cost_aux(seed_neighborhood, seed_comp, seed_dfs, candidate_neighborhood, candidate_comp, candidate_dfs, self.alpha, self.beta, self.gamma)
                    if match_cost < best_match_cost:
                        best_match_cost = match_cost
                        best_candidate_match = (candidate_comp, candidate_dfs)
                        best_seed_match = (seed_comp, seed_dfs)

            if best_seed_match and best_candidate_match:	
                seed_comp, seed_dfs = best_seed_match
                candidate_comp, candidate_dfs = best_candidate_match                       
                self.make_isomorphic(seed_comp, candidate_comp, seed_vertex, candidate_vertex)
                self.extract_neighborhoods()
                
                neighborhoods = {v: self.G_prime.neighborhoods[v] for v in candidate_vertices}
                seed_neighborhood = neighborhoods[seed_vertex]
                candidate_neighborhood = neighborhoods[candidate_vertex]
                
                
        # Print all NCCs in a pretty way
        for vertex in candidate_vertices:
            neighborhood = self.G_prime.neighborhoods[vertex]
            print(f"\nNeighborhood for vertex {vertex.node_id}:")
            for i, ncc in enumerate(neighborhood.NCC):
                print(f"  Component {i + 1}:")
                for edge in ncc:
                    print(f"    {edge}")


   
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

                    # Normalize NCP cost (if needed)
                    ncp_cost = self.compare_ncp(node_v.label, node_u.label)


                    # Calculate total cost
                    total_cost = degree_diff + ncp_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_pair = (node_v, node_u)

            return best_pair

        def bfs_and_match(comp_v, comp_u):
            """"
           Use adjacency matrices to enforce structural isomorphism between two components.
            This function assumes:
            - comp_v and comp_u are lists of Node objects representing two components.
            - Each Node object supports:
                    - getEdgesInComponent(component): returns a list (or set) of neighbor node IDs within the given component.
                    - addEdge(other_node_id): adds an edge from this node to the node with ID other_node_id.
                    - a property 'node_id' that uniquely identifies the node.
                    - a property 'label' that may be generalized using self.get_best_generalization_label(node1, node2)
            - Helper functions addVertexToComponent (and/or addVertexToComponent) are available
                to add new vertices when the components differ in size.
            
            The method works as follows:
            1. If the two components do not have the same number of nodes, add vertices to the smaller one.
            2. Sort each component’s nodes (using a key based on label and degree) so that we have a fixed pairing.
            3. For each corresponding pair, enforce that the labels match.
            4. Build the adjacency matrix for each component (using the same ordering).
            5. For each pair of nodes (i,j), if one component has an edge that the other is missing, add the missing edge.
            6. Finally, verify that the two matrices (and hence the structures) are identical.
            """
            # === STEP 0: Balance the components (if needed) so they have the same number of nodes.
            # (Assumes addVertexToComponent(component, component_to_match, owning_vertex)
            #  is available in the enclosing scope.)
            while len(comp_v) < len(comp_u):
                addVertexToComponent(comp_v, comp_u, seed_vertex)  # seed_vertex belongs to comp_v's group
            while len(comp_u) < len(comp_v):
                addVertexToComponent(comp_u, comp_v, candidate_vertex)  # candidate_vertex for comp_u

            n = len(comp_v)
            if n != len(comp_u):
                raise ValueError("Failed to balance components to the same size.")

            # === STEP 1: Sort nodes to define a fixed correspondence.
            # Here we sort by (label, degree_in_component) to provide a natural matching.
            def sort_key_v(node):
                return (node.label, len(node.getEdgesInComponent(comp_v)))
            def sort_key_u(node):
                return (node.label, len(node.getEdgesInComponent(comp_u)))
            
            sorted_v = sorted(comp_v, key=sort_key_v)
            sorted_u = sorted(comp_u, key=sort_key_u)
            
            # === STEP 2: Enforce label matching for corresponding nodes.
            for i in range(n):
                if sorted_v[i].label != sorted_u[i].label:
                    common_label = self.get_best_generalization_label(sorted_v[i].label, sorted_u[i].label)
                    sorted_v[i].label = common_label
                    sorted_u[i].label = common_label

            # === STEP 3: Build index maps and the adjacency matrices.
            # Create maps: node_id -> index (according to the sorted order)
            idx_map_v = {sorted_v[i].node_id: i for i in range(n)}
            idx_map_u = {sorted_u[i].node_id: i for i in range(n)}
            
            # Initialize n x n matrices with zeros.
            A = [[0] * n for _ in range(n)]  # for comp_v
            B = [[0] * n for _ in range(n)]  # for comp_u
            
            # Fill matrix A: for each node in sorted_v, mark an edge if the neighbor is in comp_v.
            for i in range(n):
                node = sorted_v[i]
                for neighbor_id in node.getEdgesInComponent(comp_v):
                    if neighbor_id in idx_map_v:  # neighbor belongs to comp_v
                        j = idx_map_v[neighbor_id]
                        A[i][j] = 1
                        A[j][i] = 1  # assume undirected edges
            # Similarly for matrix B:
            for i in range(n):
                node = sorted_u[i]
                for neighbor_id in node.getEdgesInComponent(comp_u):
                    if neighbor_id in idx_map_u:
                        j = idx_map_u[neighbor_id]
                        B[i][j] = 1
                        B[j][i] = 1

            # === STEP 4: Compare the matrices and add missing edges.
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i][j] != B[i][j]:
                        # If comp_v has an edge that comp_u is missing, add it to comp_u.
                        if A[i][j] == 1 and B[i][j] == 0:
                            node1 = sorted_u[i]
                            node2 = sorted_u[j]
                            if node2.node_id not in node1.edges:
                                node1.addEdge(node2.node_id)
                                node2.addEdge(node1.node_id)
                            B[i][j] = 1
                            B[j][i] = 1
                        # If comp_u has an edge that comp_v is missing, add it to comp_v.
                        elif B[i][j] == 1 and A[i][j] == 0:
                            node1 = sorted_v[i]
                            node2 = sorted_v[j]
                            if node2.node_id not in node1.edges:
                                node1.addEdge(node2.node_id)
                                node2.addEdge(node1.node_id)
                            A[i][j] = 1
                            A[j][i] = 1

            # === STEP 5: Final check: the two matrices should now be identical.
            if A != B:
                raise ValueError("Components are not isomorphic after applying adjacency matrix matching.")

       
        def addVertexToComponent(component, component_to_be_matched ,owning_vertex):
            candidates = [
                node for node in self.G_prime.N 
                if not node.Anonymized 
                and node.node_id != seed_vertex.node_id
                and node.node_id != candidate_vertex.node_id
                and node.node_id not in [node for node in owning_vertex.edges]
            ]
            if candidates:
                candidates.sort(key=lambda n: (len(n.edges))) 
                selected = candidates[0]
            else:
                candidates = [
                    node for node in self.G_prime.N 
                    if node.node_id != seed_vertex.node_id
                    and node.node_id != candidate_vertex.node_id
                    and node.node_id not in [node for node in owning_vertex.edges]
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
                    raise ValueError("No more candidates available for anonymization.")
            component.append(selected)
            selected.addEdge(owning_vertex.node_id)
            owning_vertex.addEdge(selected.node_id)
        
        # Step 2: Perform BFS and modify components
        bfs_and_match(comp_v, comp_u) #

    


        
        

       


    






 