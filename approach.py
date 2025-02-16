import numpy as np
from scipy.optimize import linear_sum_assignment
from graph import Graph, Node, Neighborhood
from functools import cmp_to_key
import random

class Anonymization(Graph):

    def __init__(self, G: Graph, k: int, alpha: float, beta: float, gamma: float):
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
                    
    def extract_components(self, node):
        # Only consider the neighbors of the given node
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
        # Reset visited flag only for nodes in the neighbors list
        for neighbor in neighbors:
            neighbor.Visited = False
        for neighbor in neighbors:
            if not neighbor.Visited:
                component = []
                dfs(neighbor, component)
                components.append(component)
        return components
        
    def DFS(self, start_node, component):
        """
        Perform Depth First Search (DFS) starting from a given node.
        Returns a list of tuples representing the DFS traversal in the format (id1, id2, l1, l2).
        """
        stack = [(start_node, None)]  # (current_node, parent_node)
        visited = set()
        dfs_result = []
        mapping = {}
        counter = 0
        seen_edges = set()  # Use a set to prevent duplicate edge entries
        
        while stack:
            current_node, parent_node = stack.pop()
            if current_node.node_id in visited:
                continue
            
            mapping[current_node.node_id] = counter
            counter += 1
            visited.add(current_node.node_id)
            
            if parent_node is not None:
                edge_key = (min(mapping[parent_node.node_id], mapping[current_node.node_id]),
                            max(mapping[parent_node.node_id], mapping[current_node.node_id]))
                if edge_key not in seen_edges:
                    dfs_result.append((mapping[parent_node.node_id], mapping[current_node.node_id],
                                       parent_node.label, current_node.label))
                    seen_edges.add(edge_key)
                    
            neighbors = []
            for neighbor_id in current_node.getEdgesInComponent(component):
                if neighbor_id not in visited:
                    neighbors.append(self.G_prime.getNode(neighbor_id))
                else:
                    edge_key = (min(mapping[current_node.node_id], mapping[neighbor_id]),
                                max(mapping[current_node.node_id], mapping[neighbor_id]))
                    if edge_key not in seen_edges:
                        dfs_result.append((mapping[current_node.node_id], mapping[neighbor_id],
                                           current_node.label, self.G_prime.getNode(neighbor_id).label))
                        seen_edges.add(edge_key)
            # Sort neighbors by degree and label (largest first)
            neighbors.sort(key=lambda n: (len(n.getEdgesInComponent(component)), n.label), reverse=True)
            for neighbor in neighbors:
                stack.append((neighbor, current_node))
        return dfs_result

    def getBestComponentDFS(self, component):
        """
        For each node in the component, perform a DFS and return the lexicographically best DFS code.
        """
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
        
        # Reset visited flags for all nodes in the component
        for node in component:
            node.Visited = False
        
        R = []
        for node in component:
            R_aux = self.DFS(node, component)
            R_aux.sort(key=cmp_to_key(dfs_edge_comparator))
            R.append(R_aux)
        
        R.sort(key=lambda x: (len(x), [(edge[2] if edge[2] is not None else '',
                                          edge[3] if edge[3] is not None else '',
                                          edge[0], edge[1]) for edge in x]))
        return R[0]  # Return the lexicographically smallest DFS code
        
    def get_generalization_level(self, label):
        return self.label_hierarchy.get(label, 0)
    
    def get_best_generalization_label(self, label1, label2):
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
            # Sorting key based on label generalization and edge attributes.
            label1, label2 = edge[2], edge[3]
            return (self.get_generalization_level(label1),
                    self.get_generalization_level(label2),
                    edge[0], edge[1])
        
        for node in self.G_prime.N:
            components = self.extract_components(node)
            
            component_to_dfs = []
            for component in components:
                dfs_code = self.getBestComponentDFS(component)
                dfs_code.sort(key=generalized_sort_key)
                component_to_dfs.append((component, dfs_code))
            
            component_to_dfs.sort(key=lambda item: (
                len({edge[0] for edge in item[1]}.union({edge[1] for edge in item[1] if edge[1] is not None})),
                1 if len(item[1]) == 1 and any(edge[1] is None for edge in item[1]) else 0,
                [(edge[2] if edge[2] is not None else '',
                  edge[3] if edge[3] is not None else '',
                  edge[0], edge[1]) for edge in item[1]]
            ))
            
            sorted_components = [item[0] for item in component_to_dfs]
            sorted_NCC = [item[1] for item in component_to_dfs]
            
            neighborhood = Neighborhood(sorted_components, sorted_NCC)
            self.G_prime.neighborhoods[node] = neighborhood
        
        print("** END NEIGHBORHOODS EXTRACTION AND CODING **")
        
    def compare_ncp(self, label1, label2):
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
        level = self.get_generalization_level(label)
        max_level = self.get_generalization_level("*")
        return level / max_level
        
    def cost_aux(self, neighborhood_u, comp_u, dfs_u, neighborhood_v, comp_v, dfs_v, alpha, beta, gamma):
        comp_edge_cost = abs(neighborhood_u.getNumberOfEdges(comp_u) - neighborhood_v.getNumberOfEdges(comp_v))
        comp_vertex_addition_cost = abs(len(comp_u) - len(comp_v))
        comp_label_cost = sum(self.compare_ncp(edge_u[2], edge_v[2]) + self.compare_ncp(edge_u[3], edge_v[3])
                               for edge_u, edge_v in zip(dfs_u, dfs_v))
        match_cost = alpha * comp_label_cost + beta * comp_edge_cost + gamma * comp_vertex_addition_cost
        return match_cost
        
    def cost(self, neighborhood_v, neighborhood_u, alpha, beta, gamma):
        total_match_cost = 0
        matched_u = []
        matched_v = []
        best_match_cost = float('inf')
        best_match_idx_i = None
        best_match_idx_j = None
        
        while len(matched_u) < len(neighborhood_u.components) and len(matched_v) < len(neighborhood_v.components):
            for i, (comp_u, dfs_u) in enumerate(zip(neighborhood_u.components, neighborhood_u.NCC)):
                for j, (comp_v, dfs_v) in enumerate(zip(neighborhood_v.components, neighborhood_v.NCC)):
                    match_cost = self.cost_aux(neighborhood_u, comp_u, dfs_u,
                                               neighborhood_v, comp_v, dfs_v,
                                               alpha, beta, gamma)
                    if match_cost < best_match_cost:
                        best_match_cost = match_cost
                        best_match_idx_j = j
                        best_match_idx_i = i
            if best_match_idx_j is not None:
                matched_u.append(best_match_idx_i)
                matched_v.append(best_match_idx_j)
                # Preserving original logic (note: this line might be a bug, but is kept intact)
                total_match_cost += total_match_cost
        for i, dfs_u in enumerate(neighborhood_u.NCC):
            if i not in matched_u:
                total_match_cost += self.generalization_cost(dfs_u) * alpha
        for i, dfs_v in enumerate(neighborhood_v.NCC):
            if i not in matched_v:
                total_match_cost += self.generalization_cost(dfs_v) * alpha
        return total_match_cost

    def generalization_cost(self, dfs_code):
        cost = 0
        for edge in dfs_code:
            cost += self.compare_ncp(edge[2], '*') + self.compare_ncp(edge[3], '*')
        return cost
    
    def anonymize_neighborhoods(self, candidate_vertices):
        """
        Anonymize the neighborhoods of the given candidate vertices.
        """
        neighborhoods = {v: self.G_prime.neighborhoods[v] for v in candidate_vertices}
        seed_vertex, seed_neighborhood = list(neighborhoods.items())[0]
        candidate_vertex, candidate_neighborhood = list(neighborhoods.items())[1]
        
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
                    unmatched_seed.append(([], []))
            if not unmatched_candidate:
                for component in unmatched_seed:
                    unmatched_candidate.append(([], []))
            
            best_candidate_match = None
            best_seed_match = None        
            
            for seed_comp, seed_dfs in unmatched_seed:
                best_match_cost = float('inf')
                for candidate_comp, candidate_dfs in unmatched_candidate:
                    match_cost = self.cost_aux(seed_neighborhood, seed_comp, seed_dfs,
                                               candidate_neighborhood, candidate_comp, candidate_dfs,
                                               self.alpha, self.beta, self.gamma)
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
                
        best_label = self.get_best_generalization_label(seed_vertex.label, candidate_vertex.label)
        seed_vertex.label = best_label
        candidate_vertex.label = best_label
                
        # # Print all NCCs in a pretty way
        # for vertex in candidate_vertices:
        #     neighborhood = self.G_prime.neighborhoods[vertex]
        #     print(f"\nNeighborhood for vertex {vertex.node_id}:")
        #     for i, ncc in enumerate(neighborhood.NCC):
        #         print(f"  Component {i + 1}:")
        #         for edge in ncc:
        #             print(f"    {edge}")
                    
    def make_isomorphic(self, comp_v, comp_u, seed_vertex, candidate_vertex):
        """
        Make two components isomorphic by modifying their structure directly.
        """
        def bfs_and_match(comp_v, comp_u):
            # --- STEP 0: Balance the components.
            while len(comp_v) < len(comp_u):
                addVertexToComponent(comp_v, seed_vertex)
            while len(comp_u) < len(comp_v):
                addVertexToComponent(comp_u, candidate_vertex)
            n = len(comp_v)
            if n != len(comp_u):
                raise ValueError("Failed to balance components to the same size.")
            # --- STEP 1: Establish a “reference” ordering for comp_v.
            sorted_v = sorted(comp_v, key=lambda node: (len(node.getEdgesInComponent(comp_v)), node.label))
            idx_map_v = {sorted_v[i].node_id: i for i in range(n)}
            A = np.zeros((n, n), dtype=int)
            for i, node in enumerate(sorted_v):
                for neighbor_id in node.getEdgesInComponent(comp_v):
                    if neighbor_id in idx_map_v:
                        j = idx_map_v[neighbor_id]
                        A[i, j] = 1
                        A[j, i] = 1
            # --- STEP 2: Build a cost matrix for matching nodes in comp_v to nodes in comp_u.
            sorted_v_degrees = np.array([len(node.getEdgesInComponent(comp_v)) for node in sorted_v])
            comp_u_degrees = np.array([len(node.getEdgesInComponent(comp_u)) for node in comp_u])
            cost_matrix = np.abs(sorted_v_degrees[:, None] - comp_u_degrees[None, :])
            label_penalty = np.zeros((n, n))
            for i, node_v in enumerate(sorted_v):
                for j, node_u in enumerate(comp_u):
                    if node_v.label != node_u.label:
                        label_penalty[i, j] = 1
            cost_matrix = cost_matrix.astype(float) + label_penalty
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            best_order = [None] * n
            for i, j in zip(row_ind, col_ind):
                best_order[i] = comp_u[j]
            # --- STEP 4: Build adjacency matrix B for the reordered comp_u.
            idx_map_u = {best_order[i].node_id: i for i in range(n)}
            B = np.zeros((n, n), dtype=int)
            for i, node in enumerate(best_order):
                for neighbor_id in node.getEdgesInComponent(comp_u):
                    if neighbor_id in idx_map_u:
                        j = idx_map_u[neighbor_id]
                        B[i, j] = 1
                        B[j, i] = 1
            # --- STEP 5: Enforce label matching between corresponding nodes.
            for i in range(n):
                if sorted_v[i].label != best_order[i].label:
                    common_label = self.get_best_generalization_label(sorted_v[i].label, best_order[i].label)
                    sorted_v[i].label = common_label
                    best_order[i].label = common_label
            # --- STEP 6: Add missing edges only where one component has an edge and the other does not.
            edge_sets_v = {node.node_id: set(node.edges) for node in sorted_v}
            edge_sets_u = {node.node_id: set(node.edges) for node in best_order}
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j] != B[i, j]:
                        if A[i, j] == 1 and B[i, j] == 0:
                            node1 = best_order[i]
                            node2 = best_order[j]
                            if node2.node_id not in edge_sets_u[node1.node_id]:
                                node1.addEdge(node2.node_id)
                                node2.addEdge(node1.node_id)
                                edge_sets_u[node1.node_id].add(node2.node_id)
                                edge_sets_u[node2.node_id].add(node1.node_id)
                                B[i, j] = 1
                                B[j, i] = 1
                        elif B[i, j] == 1 and A[i, j] == 0:
                            node1 = sorted_v[i]
                            node2 = sorted_v[j]
                            if node2.node_id not in edge_sets_v[node1.node_id]:
                                node1.addEdge(node2.node_id)
                                node2.addEdge(node1.node_id)
                                edge_sets_v[node1.node_id].add(node2.node_id)
                                edge_sets_v[node2.node_id].add(node1.node_id)
                                A[i, j] = 1
                                A[j, i] = 1
            # --- STEP 7: Final verification.
            if not np.array_equal(A, B):
                raise ValueError("Components are not isomorphic after optimized matching.")
        
        def addVertexToComponent(component, owning_vertex):
            if component: 
                candidates = [node for node in self.G_prime.N 
                            if not node.Anonymized 
                            and node.node_id != owning_vertex.node_id 
                            and node.node_id not in [node.node_id for node in component]]
            else:
                candidates = [node for node in self.G_prime.N 
                            if not node.Anonymized 
                            and node.node_id != owning_vertex.node_id 
                            and node.node_id not in owning_vertex.edges]
            if candidates:
                selected = min(candidates, key=lambda n: len(n.edges))
            else:
                if component:
                    candidates = [node for node in self.G_prime.N 
                                if node.node_id != owning_vertex.node_id 
                                and node.node_id not in [node.node_id for node in component]]
                else:
                    candidates = [node for node in self.G_prime.N 
                                if node.node_id != owning_vertex.node_id    
                                and node.node_id not in owning_vertex.edges]
                if candidates:
                    selected = min(candidates, key=lambda n: len(n.edges))
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
        
        bfs_and_match(comp_v, comp_u)
