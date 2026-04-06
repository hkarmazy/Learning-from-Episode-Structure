"""Embedding space visualization — UMAP, attractor landscape, SW proximity (Stream 4).

Three complementary visualizations:
  1. UMAP projection — cluster structure
  2. Attractor basin landscape — potential field contour
  3. SW neighbor proximity matrix — heatmap
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract embeddings and labels from a data loader.

    Returns:
        embeddings: numpy array [N, embed_dim]
        labels: numpy array [N]
    """
    model.eval()
    all_emb, all_lab = [], []
    for batch in loader:
        seq = batch['sequence'].to(device)
        _, emb = model(seq)
        all_emb.append(emb.cpu().numpy())
        all_lab.append(batch['label'].numpy())
    return np.concatenate(all_emb), np.concatenate(all_lab)


def plot_umap(embeddings, labels, n_classes: int = 32, title: str = '',
              output_path: str = None, rare_classes: list = None):
    """UMAP 2D projection colored by class.

    Rare classes (24-31) plotted with larger markers + labels.
    """
    if not HAS_UMAP:
        print("umap-learn not installed, skipping UMAP plot")
        return

    if rare_classes is None:
        rare_classes = list(range(24, 32))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', n_classes)

    for c in range(n_classes):
        mask = labels == c
        is_rare = c in rare_classes
        size = 40 if is_rare else 10
        alpha = 0.9 if is_rare else 0.4
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(c)], s=size, alpha=alpha,
                   label=f'Class {c}' if is_rare else None)

    ax.set_title(f'UMAP Embedding Space — {title}')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Compute silhouette score
    sil = silhouette_score(embeddings, labels, metric='cosine',
                           sample_size=min(1000, len(labels)))
    return sil


def plot_attractor_landscape(centroids, freq_vec, alpha_loss: float = 1.8,
                             title: str = '', output_path: str = None):
    """Attractor basin landscape as 2D contour plot.

    Projects centroids to 2D via PCA, then computes potential field:
    V(x,y) = -sum_c depth_c * exp(-d^2 / (2σ^2))
    """
    n_classes = centroids.shape[0]

    # PCA to 2D
    pca = PCA(n_components=2)
    centroids_2d = pca.fit_transform(centroids)

    # Compute depth boost per class
    depths = np.log(1.0 + 1.0 / (freq_vec + 1e-7)) ** alpha_loss

    # Create grid
    margin = 2.0
    x_min, x_max = centroids_2d[:, 0].min() - margin, centroids_2d[:, 0].max() + margin
    y_min, y_max = centroids_2d[:, 1].min() - margin, centroids_2d[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Potential field
    sigma = 1.0
    V = np.zeros(grid.shape[0])
    for c in range(n_classes):
        d2 = ((grid - centroids_2d[c]) ** 2).sum(axis=1)
        V -= depths[c] * np.exp(-d2 / (2 * sigma ** 2))

    V = V.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(xx, yy, V, levels=50, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Potential V(x,y)')

    # Plot centroids
    rare_classes = list(range(24, 32))
    for c in range(n_classes):
        is_rare = c in rare_classes
        color = 'red' if is_rare else 'white'
        size = 80 if is_rare else 30
        ax.scatter(centroids_2d[c, 0], centroids_2d[c, 1],
                   c=color, s=size, edgecolors='black', linewidth=1, zorder=5)
        if is_rare:
            ax.annotate(str(c), (centroids_2d[c, 0], centroids_2d[c, 1]),
                        fontsize=8, ha='center', va='bottom', color='white')

    ax.set_title(f'Attractor Basin Landscape — {title}')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def sw_proximity_analysis(centroids, transition_matrix, output_path: str = None):
    """Analyze and visualize SW neighbor proximity in embedding space.

    Tests whether SW-neighbor centroid distances are significantly smaller
    than non-neighbor distances.

    Returns:
        Dict with test results and effect size.
    """
    n = centroids.shape[0]

    # Compute pairwise centroid distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    # Identify SW edges
    T_sym = transition_matrix + transition_matrix.T
    sw_mask = T_sym > 0.02

    sw_dists = []
    non_sw_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            if sw_mask[i, j]:
                sw_dists.append(dist_matrix[i, j])
            else:
                non_sw_dists.append(dist_matrix[i, j])

    sw_dists = np.array(sw_dists)
    non_sw_dists = np.array(non_sw_dists)

    # Mann-Whitney U test
    if len(sw_dists) > 0 and len(non_sw_dists) > 0:
        stat, p_value = mannwhitneyu(sw_dists, non_sw_dists, alternative='less')
        # Rank-biserial correlation (effect size)
        n1, n2 = len(sw_dists), len(non_sw_dists)
        r = 1 - (2 * stat) / (n1 * n2)
    else:
        stat, p_value, r = 0, 1.0, 0.0

    results = {
        'sw_mean_dist': sw_dists.mean() if len(sw_dists) > 0 else 0,
        'non_sw_mean_dist': non_sw_dists.mean() if len(non_sw_dists) > 0 else 0,
        'mann_whitney_U': stat,
        'p_value': p_value,
        'rank_biserial_r': r,
    }

    # Heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(dist_matrix, ax=ax, cmap='coolwarm_r',
                xticklabels=range(n), yticklabels=range(n))

    # Overlay SW edges
    for i in range(n):
        for j in range(n):
            if sw_mask[i, j] and i != j:
                ax.plot(j + 0.5, i + 0.5, 'k.', markersize=3)

    ax.set_title('Centroid Distance Matrix (dots = SW neighbors)')
    ax.set_xlabel('Class (by Zipf rank)')
    ax.set_ylabel('Class (by Zipf rank)')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    return results
