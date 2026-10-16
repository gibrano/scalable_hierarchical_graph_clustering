import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from hierarchical_graph_clustering import HierarchicalGraphClustering, ScalableHierarchicalGraphClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Create some simple graphs
graphs = []
edges = [
    [['1', '2'], ['1', '3'], ['2', '3']],
    [['a', 'b'], ['b', 'c']],
    [['1', '2'], ['1', '3'], ['z', '1'], ['a', '1']],
    [['a', 'b'], ['b', 'c'], ['a', 'c'], ['c', 'z']]
]
for edge_list in edges:
    g = nx.Graph()
    g.add_edges_from(edge_list)
    graphs.append(g)

# Initialize clustering models
model_exact = HierarchicalGraphClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
model_scalable = ScalableHierarchicalGraphClustering(distance_threshold=0, n_clusters=None, compute_distances=True, B=10, N_projections=5, epsilon=1e-5)

# Fit the models
print("Fitting exact model...")
model_exact = model_exact.fit(graphs)

print("Fitting scalable model...")
model_scalable = model_scalable.fit(graphs)

# Print results
print("\nExact Clustering:")
print("Labels:", model_exact.labels_)
print("Children:\n", model_exact.children_)
print("Distances:\n", model_exact.distances_)

print("\nScalable Clustering:")
print("Labels:", model_scalable.labels_)
print("Children:\n", model_scalable.children_)
print("Distances:\n", model_scalable.distances_)

# Plotting
plt.figure(figsize=(16, 10))

# Exact Model
plt.subplot(2, 2, 1)
plt.title("Exact Hierarchical Graph Clustering Dendrogram")
plot_dendrogram(model_exact, truncate_mode="level", p=3)

plt.subplot(2, 2, 2)
plt.title("Exact Quality Function q(C) over iterations")
plt.plot(range(len(model_exact.q_values_)), model_exact.q_values_, marker='o', color='b')
plt.xlabel("Merge iteration")
plt.ylabel("q(C)")
plt.grid(True)

# Scalable Model
plt.subplot(2, 2, 3)
plt.title("Scalable Hierarchical Graph Clustering Dendrogram")
if hasattr(model_scalable, 'children_'):
    plot_dendrogram(model_scalable, truncate_mode="level", p=3)
else:
    plt.text(0.5, 0.5, 'No merges performed', horizontalalignment='center', verticalalignment='center')


plt.subplot(2, 2, 4)
plt.title("Scalable Quality Function q(C) over iterations")
if hasattr(model_scalable, 'q_values_'):
    plt.plot(range(len(model_scalable.q_values_)), model_scalable.q_values_, marker='o', color='g')
plt.xlabel("Merge iteration")
plt.ylabel("q(C)")
plt.grid(True)

plt.tight_layout()
print("Saving dendrogram and quality plot to result_plot.png")
plt.savefig("result_plot.png")
