import numpy as np
import networkx as nx
import random
from collections import defaultdict

# 1. Regenerar el dataset localmente
np.random.seed(42)
random.seed(42)

graphs = []
true_labels = []

for _ in range(500):
    g = nx.star_graph(9)
    edges = list(g.edges())
    g.remove_edge(*random.choice(edges))
    while True:
        u, v = random.sample(range(10), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
            break
    graphs.append(g)
    true_labels.append(0)

for _ in range(500):
    g = nx.cycle_graph(10)
    edges = list(g.edges())
    g.remove_edge(*random.choice(edges))
    while True:
        u, v = random.sample(range(10), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)
            break
    graphs.append(g)
    true_labels.append(1)

indices = list(range(1000))
random.shuffle(indices)
graphs = [graphs[i] for i in indices]
true_labels = np.array([true_labels[i] for i in indices])

# 2. Funciones LSH
def get_adjacency(g, N):
    g_copy = g.copy()
    g_copy.remove_edges_from(nx.selfloop_edges(g_copy))
    nodes_order = list(sorted(g_copy.nodes()))
    A = nx.adjacency_matrix(g_copy, nodelist=nodes_order, weight=None).toarray().astype(float)
    A = (A > 0).astype(float)
    np.fill_diagonal(A, 0)
    return A

def matrix_prod_trace(A, B):
    return np.trace(np.dot(A.T, B))

def hash_p_stable(projection, r):
    if r <= 0:
        return 0
    b = np.random.uniform(0, r)
    return int(np.floor((projection + b) / r))

def get_erdos_renyi_graph(n, p):
    while True:
        g = nx.erdos_renyi_graph(n, p)
        if g.number_of_edges() > 0:
            return g

def build_lsh_buckets(graphs, num_proj, B, p_er=0.8):
    num_graphs = len(graphs)
    N_pad = 10
    bucket_ids = [""] * num_graphs
    random_graphs = []
    projection_ranges = []

    for k in range(num_proj):
        erg = get_erdos_renyi_graph(N_pad, p_er)
        random_graphs.append(erg)
        A_er = get_adjacency(erg, N_pad)

        projections = []
        for g in graphs:
            A = get_adjacency(g, N_pad)
            proj = matrix_prod_trace(A, A_er)
            projections.append(proj)

        min_proj = min(projections)
        max_proj = max(projections)
        projection_ranges.append((min_proj, max_proj))
        r = (max_proj - min_proj) / B

        for i in range(num_graphs):
            hash_bin = hash_p_stable(projections[i], r)
            bucket_ids[i] += f"{hash_bin}_"

    buckets = defaultdict(list)
    for i, bid in enumerate(bucket_ids):
        buckets[bid].append(i)

    return buckets, bucket_ids

# Probar diferentes configuraciones
for num_proj in [2, 3, 4]:
    for B in [2, 3, 5, 8, 10]:
        np.random.seed(42) # Fijo para comparar las configuraciones
        buckets, bucket_ids = build_lsh_buckets(graphs, num_proj, B)
        
        non_empty = {bid: m for bid, m in buckets.items() if len(m) > 0}
        
        # Encontrar los 3 buckets más grandes
        sorted_buckets = sorted(non_empty.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"--- Config: num_proj={num_proj}, B={B} ---")
        print(f"Total buckets no vacíos: {len(non_empty)}")
        for idx, (bid, members) in enumerate(sorted_buckets[:3]):
            stars = sum(1 for i in members if true_labels[i] == 0)
            cycles = sum(1 for i in members if true_labels[i] == 1)
            print(f"  Bucket {idx+1} ({bid}): Total={len(members)} (Estrellas: {stars}, Ciclos: {cycles})")
        print()
