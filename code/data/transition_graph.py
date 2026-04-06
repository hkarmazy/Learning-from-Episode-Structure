"""Build and verify small-world transition graphs for play sequence generation."""

import numpy as np
import networkx as nx


def build_sw_graph(n_classes: int = 32, k: int = 4, beta: float = 0.15,
                   seed: int = 42) -> nx.Graph:
    """Build a Watts-Strogatz small-world graph.

    Args:
        n_classes: Number of nodes (toy classes).
        k: Each node connected to k nearest neighbors in ring topology.
        beta: Rewiring probability — calibrated from published play data.
        seed: Random seed for reproducibility.

    Returns:
        A networkx Graph with small-world topology.
    """
    return nx.watts_strogatz_graph(n_classes, k, beta, seed=seed)


def sw_transition_matrix(graph: nx.Graph, self_loop_weight: float = 0.3) -> np.ndarray:
    """Convert SW graph to a row-stochastic transition matrix.

    Each node has a self-loop (probability of staying on the same toy)
    plus uniform weight over its SW neighbors.

    Args:
        graph: Small-world graph.
        self_loop_weight: Probability of self-transition.

    Returns:
        Row-stochastic transition matrix [n_classes, n_classes].
    """
    n = graph.number_of_nodes()
    T = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        neighbors = list(graph.neighbors(i))
        if len(neighbors) == 0:
            T[i, i] = 1.0
            continue
        T[i, i] = self_loop_weight
        neighbor_weight = (1.0 - self_loop_weight) / len(neighbors)
        for j in neighbors:
            T[i, j] = neighbor_weight

    return T


def verify_sw_properties(graph: nx.Graph, n_classes: int = 32, k: int = 4) -> dict:
    """Verify that graph has small-world properties: C >> C_random, L ≈ L_random.

    Returns:
        Dict with clustering coefficient, path length, and their random baselines.
    """
    C = nx.average_clustering(graph)
    L = nx.average_shortest_path_length(graph)

    # Random graph baselines (Erdos-Renyi with same edge density)
    C_random = k / n_classes  # expected clustering for random graph
    L_random = np.log(n_classes) / np.log(k)  # expected path length for random graph

    return {
        'clustering': C,
        'clustering_random': C_random,
        'path_length': L,
        'path_length_random': L_random,
        'is_small_world': C > 2 * C_random and L < 2 * L_random,
    }


if __name__ == '__main__':
    G = build_sw_graph()
    props = verify_sw_properties(G)
    print("Small-world verification:")
    for key, val in props.items():
        print(f"  {key}: {val}")
    T = sw_transition_matrix(G)
    print(f"\nTransition matrix shape: {T.shape}")
    print(f"Row sums (should be 1.0): {T.sum(axis=1)[:5]}")
