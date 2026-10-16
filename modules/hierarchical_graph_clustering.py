import numpy as np
import networkx as nx
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering

class HierarchicalGraphClustering(AgglomerativeClustering):
    """
    Hierarchical clustering for graphs using Jensen-Shannon divergence.
    
    This class mimics scikit-learn's AgglomerativeClustering and returns 
    the exact same format so the output can be easily visualized 
    using dendrograms and the scipy hierarchy Z matrix.
    """
    def __init__(self, n_clusters=2, *, linkage='average', memory=None, 
                 connectivity=None, compute_full_tree='auto', 
                 distance_threshold=None, compute_distances=True):
        
        # For precomputed metrics, 'ward' linkage is not supported by sklearn.
        # So we default to 'average' linkage.
        # We enforce metric="precomputed".
        super().__init__(
            n_clusters=n_clusters,
            metric="precomputed",
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances
        )

    def _von_newmann_entropy(self, g):
        num_edges = g.size(weight='weight') if nx.is_weighted(g) else g.number_of_edges()
        if num_edges == 0:
            return 0.0
            
        # Ensure toarray() due to recent scipy/networkx compatibility updates
        L = nx.laplacian_matrix(g).toarray()
        
        # Using eigh for symmetric Laplacian matrices is faster and numerically stable
        # eigvals_only=True removes the need to compute eigenvectors
        e = eigh(L / (2 * num_edges), eigvals_only=True)
        e = e[e > 0]
        
        if len(e) == 0:
            return 0.0
            
        return -np.sum(e * np.log2(e))

    def _get_adjacency(self, g, N):
        """
        Orders nodes by degree (descending) and pads the adjacency matrix 
        to a fixed N x N size.
        """
        nodes_order = list(dict(sorted(g.degree, key=lambda x: x[1], reverse=True)).keys())
        A = nx.adjacency_matrix(g, nodelist=nodes_order).toarray()
        M = A.shape[0]
        # Pad with 0s for nodes up to N
        A = np.pad(A, [(0, N - M), (0, N - M)], mode='constant')
        return A

    def _aggregate(self, g1, g2):
        """
        Aggregates two graphs by taking the ceiling of their averaged adjacency matrices.
        """
        N = max(g1.number_of_nodes(), g2.number_of_nodes())
        A1 = self._get_adjacency(g1, N)
        A2 = self._get_adjacency(g2, N)
        
        A_agg = np.ceil((A1 + A2) / 2)
        return nx.from_numpy_array(A_agg)

    def fit(self, X, y=None):
        """
        Fit the hierarchical clustering on the data.
        
        Parameters
        ----------
        X : list of networkx.Graph
            The graphs to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.
        """
        n_samples = len(X)
        self.distance_matrix_ = np.zeros((n_samples, n_samples))
        
        # Precompute individual entropies to optimize JS divergence calculation
        entropies = [self._von_newmann_entropy(g) for g in X]
        
        # O(N^2) pairwise distance via Jensen-Shannon Divergence
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                g1 = X[i]
                g2 = X[j]
                
                entropy_agg = self._von_newmann_entropy(self._aggregate(g1, g2))
                val = entropy_agg - (entropies[i] + entropies[j]) / 2
                
                # Prevent negative values due to floating point inaccuracies
                div = np.sqrt(max(0.0, val))
                
                self.distance_matrix_[i, j] = div
                self.distance_matrix_[j, i] = div
                
        # Scikit-learn AgglomerativeClustering with precomputed metric 
        # requires passing distance_matrix_ to fit()
        super().fit(self.distance_matrix_, y)
        
        # Compute quality function q(C)
        self._compute_quality(X)
        
        return self

    def _entropy_from_adjacency(self, A):
        sum_weights = np.sum(A) / 2.0
        if sum_weights <= 1e-9:
            return 0.0
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        L = D - A
        e = eigh(L / (2 * sum_weights), eigvals_only=True)
        e = e[e > 0]
        if len(e) == 0:
            return 0.0
        return -np.sum(e * np.log2(e))

    def _compute_quality(self, X):
        n_samples = len(X)
        N_max = max(g.number_of_nodes() for g in X) if n_samples > 0 else 0
        
        # Original layers
        C_dict = {i: self._get_adjacency(g, N_max) for i, g in enumerate(X)}
        
        if len(C_dict) == 0:
            self.q_values_ = []
            return
            
        A_sum = sum(C_dict.values())
        h_A = self._entropy_from_adjacency(A_sum)
        
        self.q_values_ = []
        
        def calculate_q(current_C):
            if h_A > 0:
                H_C = np.mean([self._entropy_from_adjacency(A) for A in current_C])
                return 1.0 - H_C / h_A
            return 0.0
            
        # Step 0: No merges
        self.q_values_.append(calculate_q(C_dict.values()))
            
        # Iterate over merges
        if hasattr(self, 'children_'):
            for step, (child1, child2) in enumerate(self.children_):
                new_idx = n_samples + step
                # Sum the adjacencies for the merged cluster
                C_dict[new_idx] = C_dict[child1] + C_dict[child2]
                
                # Remove old clusters
                del C_dict[child1]
                del C_dict[child2]
                
                # Calculate new H_C
                self.q_values_.append(calculate_q(C_dict.values()))

    def fit_predict(self, X, y=None):
        """
        Fit and return the result of clustering.
        """
        self.fit(X, y)
        return self.labels_

class ScalableHierarchicalGraphClustering(HierarchicalGraphClustering):
    """
    Scalable Hierarchical clustering for graphs using Jensen-Shannon divergence
    and LSH (Locality Sensitive Hashing) to find the closest graph pairs faster.
    """
    def __init__(self, n_clusters=2, *, linkage='average', memory=None, 
                 connectivity=None, compute_full_tree='auto', 
                 distance_threshold=None, compute_distances=True,
                 B=10, N_projections=5, epsilon=1e-5):
        
        super().__init__(
            n_clusters=n_clusters, linkage=linkage, memory=memory, 
            connectivity=connectivity, compute_full_tree=compute_full_tree, 
            distance_threshold=distance_threshold, compute_distances=compute_distances
        )
        self.B = B
        self.N_projections = N_projections
        self.epsilon = epsilon
        
    def _matrix_prod_trace(self, A, B):
        # Tr(A * B^T) = sum_{i,j} A_{i,j} B_{i,j} since B is symmetric adjacency
        return np.sum(A * B)

    def _hash_p_stable(self, projection, r, b):
        return int(np.floor((projection + b) / r))

    def fit(self, X, y=None):
        """
        Fit the scalable hierarchical clustering on the data using LSH.
        """
        n_samples = len(X)
        if n_samples == 0:
            return self

        # Determine N_max
        N_max = max(g.number_of_nodes() for g in X)
        
        # Step 0: Track the active graphs by index and pad them
        active_indices = list(range(n_samples))
        S_graphs = {i: X[i] for i in active_indices}
        S_adj = {i: self._get_adjacency(X[i], N_max) for i in active_indices}
        
        # Step 1: Generate N_projections random Erdos-Renyi graphs
        er_graphs = []
        er_adjs = []
        for _ in range(self.N_projections):
            # p=0.8 as in the notebook example
            erg = nx.erdos_renyi_graph(N_max, 0.8)
            er_graphs.append(erg)
            er_adjs.append(nx.adjacency_matrix(erg).toarray())
            
        # Compute r and b for each projection
        r_list = []
        b_list = []
        for k in range(self.N_projections):
            projections = [self._matrix_prod_trace(S_adj[i], er_adjs[k]) for i in active_indices]
            R_max = max(projections) - min(projections)
            r = R_max / self.B if R_max > 0 else 1.0 # fallback if R_max is 0
            r_list.append(r)
            b_list.append(np.random.uniform(0, r))

        children = []
        distances = []
        
        # For q(C) calculation tracking
        self.q_values_ = []
        A_sum = sum(S_adj.values())
        h_A = self._entropy_from_adjacency(A_sum)
        
        def calculate_q(current_adj_list):
            if h_A > 0:
                H_C = np.mean([self._entropy_from_adjacency(A) for A in current_adj_list])
                return 1.0 - H_C / h_A
            return 0.0
            
        # initial q
        self.q_values_.append(calculate_q(S_adj.values()))

        # Precompute entropies for the JS divergence
        entropies = {i: self._von_newmann_entropy(g) for i, g in S_graphs.items()}
        
        cluster_sizes = {i: 1 for i in active_indices}
        current_idx = n_samples

        # Step 3 Loop
        while len(active_indices) > 1:
            # Step 2: Hashing and building buckets
            buckets = {}
            for i in active_indices:
                A = S_adj[i]
                s_i = ""
                for k in range(self.N_projections):
                    proj = self._matrix_prod_trace(A, er_adjs[k])
                    h_val = self._hash_p_stable(proj, r_list[k], b_list[k])
                    s_i += str(h_val) + "|"
                
                if s_i not in buckets:
                    buckets[s_i] = []
                buckets[s_i].append(i)
                
            # Find the two closest objects within the SAME bucket
            min_dist = float('inf')
            best_pair = None
            
            for s_i, bucket_indices in buckets.items():
                if len(bucket_indices) < 2:
                    continue
                
                # Pairwise JS divergence inside the bucket
                for i_idx in range(len(bucket_indices)):
                    for j_idx in range(i_idx + 1, len(bucket_indices)):
                        idx1 = bucket_indices[i_idx]
                        idx2 = bucket_indices[j_idx]
                        
                        g1 = S_graphs[idx1]
                        g2 = S_graphs[idx2]
                        
                        # Calculate JS divergence
                        A_agg = np.ceil((S_adj[idx1] + S_adj[idx2]) / 2)
                        g_agg = nx.from_numpy_array(A_agg)
                        
                        entropy_agg = self._von_newmann_entropy(g_agg)
                        val = entropy_agg - (entropies[idx1] + entropies[idx2]) / 2
                        div = np.sqrt(max(0.0, val))
                        
                        if div < min_dist:
                            min_dist = div
                            best_pair = (idx1, idx2, g_agg, A_agg, entropy_agg)
            
            # If all buckets had size < 2, we fallback to finding the closest pair globally
            if best_pair is None:
                for i_idx in range(len(active_indices)):
                    for j_idx in range(i_idx + 1, len(active_indices)):
                        idx1 = active_indices[i_idx]
                        idx2 = active_indices[j_idx]
                        
                        g1 = S_graphs[idx1]
                        g2 = S_graphs[idx2]
                        
                        A_agg = np.ceil((S_adj[idx1] + S_adj[idx2]) / 2)
                        g_agg = nx.from_numpy_array(A_agg)
                        
                        entropy_agg = self._von_newmann_entropy(g_agg)
                        val = entropy_agg - (entropies[idx1] + entropies[idx2]) / 2
                        div = np.sqrt(max(0.0, val))
                        
                        if div < min_dist:
                            min_dist = div
                            best_pair = (idx1, idx2, g_agg, A_agg, entropy_agg)
            
            idx1, idx2, g_agg, A_agg, entropy_agg = best_pair
            
            # Merge the best pair
            active_indices.remove(idx1)
            active_indices.remove(idx2)
            
            children.append([idx1, idx2])
            distances.append(min_dist)
            
            S_graphs[current_idx] = g_agg
            
            # Sum for quality function iteration calculation
            S_adj[current_idx] = S_adj[idx1] + S_adj[idx2]
            entropies[current_idx] = entropy_agg
            cluster_sizes[current_idx] = cluster_sizes[idx1] + cluster_sizes[idx2]
            
            active_indices.append(current_idx)
            
            # Remove old from current iteration's quality metric and hash pool
            del S_adj[idx1]
            del S_adj[idx2]
            
            current_q = calculate_q(S_adj.values())
            self.q_values_.append(current_q)
            
            # Early stopping check using epsilon
            if current_q < self.epsilon:
                break
                
            current_idx += 1
            
        # Prepare attributes for scikit-learn compliance
        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
        self.n_clusters_ = len(active_indices)
        
        # Determine labels_
        # Since this is agglomerative, we can just assign an incremental ID for top-level remaining clusters
        self.labels_ = np.zeros(n_samples, dtype=int)
        
        # We trace down from the active_indices to leaf nodes to set labels
        def set_labels(node_id, label):
            if node_id < n_samples:
                self.labels_[node_id] = label
            else:
                child1, child2 = self.children_[node_id - n_samples]
                set_labels(child1, label)
                set_labels(child2, label)
                
        for label, root_id in enumerate(active_indices):
            set_labels(root_id, label)
            
        return self
