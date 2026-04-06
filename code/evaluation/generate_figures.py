"""Generate all evaluation figures from trained visual models.

Reads results from runs_visual/ and produces 6 figures in figures/.

Usage:
    python -m evaluation.generate_figures [--runs_dir runs_visual] [--output_dir figures]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cifar_class_selection import CIFAR100_CLASS_NAMES, N_CLASSES, RARE_CLASSES

CONDITIONS = ['uniform', 'zipf', 'sw', 'zipf_sw']
CONDITION_LABELS = {
    'uniform': 'C1: Uniform',
    'zipf': 'C2: Zipf',
    'sw': 'C3: SW',
    'zipf_sw': 'C4: Zipf+SW',
}
CONDITION_COLORS = {
    'uniform': '#4C72B0',
    'zipf': '#DD8452',
    'sw': '#55A868',
    'zipf_sw': '#C44E52',
}


def load_metrics_csv(run_dir):
    """Load per-epoch metrics from CSV."""
    csv_path = os.path.join(run_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_results_json(run_dir):
    """Load final test results."""
    path = os.path.join(run_dir, 'results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_runs(runs_dir, seed=0):
    """Find run directories for each condition."""
    runs = {}
    for cond in CONDITIONS:
        run_dir = os.path.join(runs_dir, f'{cond}_seed{seed}')
        if os.path.exists(run_dir):
            runs[cond] = run_dir
    return runs


# ── Figure 1: Class Distributions ──────────────────────────────────────────

def fig_class_distributions(runs, output_dir):
    """Bar chart showing training token counts per class for each condition."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        if cond not in runs:
            ax.set_title(f'{CONDITION_LABELS[cond]} (not found)')
            continue

        metrics_path = os.path.join(runs[cond], 'metrics.csv')
        results = load_results_json(runs[cond])

        # Count tokens per class from the dataset
        # We can infer from Zipf frequencies
        from data.synthetic_generator import zipf_frequencies
        n_episodes = 200
        episode_len = 180

        if cond in ('zipf', 'zipf_sw'):
            freq = zipf_frequencies(N_CLASSES, 2.5)
        else:
            freq = np.ones(N_CLASSES) / N_CLASSES

        token_counts = freq * n_episodes * episode_len
        colors = ['#C44E52' if i in RARE_CLASSES else '#4C72B0' for i in range(N_CLASSES)]

        ax.bar(range(N_CLASSES), token_counts, color=colors, alpha=0.8)
        ax.set_title(CONDITION_LABELS[cond], fontsize=13, fontweight='bold')
        ax.set_xlabel('Class rank')
        ax.set_ylabel('Training tokens')
        ax.set_xticks(range(0, N_CLASSES, 4))
        ax.set_xticklabels(range(0, N_CLASSES, 4))

    fig.suptitle('Training Sample Distribution per Class', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_class_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ── Figure 2: Learning Curves ──────────────────────────────────────────────

def fig_learning_curves(runs, output_dir):
    """Loss and accuracy curves over epochs for all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for cond in CONDITIONS:
        if cond not in runs:
            continue
        metrics = load_metrics_csv(runs[cond])
        if not metrics:
            continue

        epochs = [m['epoch'] for m in metrics]
        color = CONDITION_COLORS[cond]
        label = CONDITION_LABELS[cond]

        # Train loss (use log scale since C3 has very large loss values)
        axes[0].plot(epochs, [m['loss_cls'] for m in metrics],
                     color=color, label=label, linewidth=1.5)
        # Val accuracy (all)
        axes[1].plot(epochs, [m['val_acc_all'] for m in metrics],
                     color=color, label=label, linewidth=1.5)
        # Val accuracy (rare)
        axes[2].plot(epochs, [m['val_acc_rare'] for m in metrics],
                     color=color, label=label, linewidth=1.5)

    axes[0].set_title('Classification Loss (L_cls)', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Accuracy (All Classes)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    axes[2].set_title('Validation Accuracy (Rare Classes)', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    fig.suptitle('Learning Curves During Training', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ── Figures 3-5: Test Accuracy Bars ────────────────────────────────────────

def _accuracy_bar(runs, output_dir, key, title, filename):
    """Generic accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conds_present = [c for c in CONDITIONS if c in runs]
    vals = []
    for cond in conds_present:
        results = load_results_json(runs[cond])
        if results:
            vals.append(results[key])
        else:
            vals.append(0)

    bars = ax.bar(
        [CONDITION_LABELS[c] for c in conds_present],
        [v * 100 for v in vals],
        color=[CONDITION_COLORS[c] for c in conds_present],
        alpha=0.85, edgecolor='black', linewidth=0.5
    )

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val * 100:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, min(100, max(v * 100 for v in vals) + 10))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def fig_test_accuracy_all(runs, output_dir):
    _accuracy_bar(runs, output_dir, 'test_acc_all',
                  'Test Accuracy — All Classes', 'fig_test_accuracy_all.png')


def fig_test_accuracy_rare(runs, output_dir):
    _accuracy_bar(runs, output_dir, 'test_acc_rare',
                  'Test Accuracy — Rare Classes (ranks 24-31)', 'fig_test_accuracy_rare.png')


def fig_test_accuracy_common(runs, output_dir):
    _accuracy_bar(runs, output_dir, 'test_acc_common',
                  'Test Accuracy — Common Classes (ranks 0-23)', 'fig_test_accuracy_common.png')


# ── Figure 6: 3D Embedding Visualization ──────────────────────────────────

def fig_3d_embeddings(runs, output_dir):
    """3D PCA of learned embeddings, one panel per condition."""
    from torch.utils.data import DataLoader
    from data.cifar_generator import CIFARPlaySequenceDataset
    from model.visual_embedding_model import VisualPlayEmbedder

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    fig = plt.figure(figsize=(16, 12))

    for idx, cond in enumerate(CONDITIONS):
        if cond not in runs:
            continue

        run_dir = runs[cond]
        model_path = os.path.join(run_dir, 'model_final.pt')
        if not os.path.exists(model_path):
            continue

        # Load model
        model = VisualPlayEmbedder(n_classes=N_CLASSES, context_window=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # Get test data (small subset for visualization)
        ds = CIFARPlaySequenceDataset(
            cond, 50, n_classes=N_CLASSES, episode_len=180,
            seed=9999, feature_cache_dir='data/cifar_cache', split='test',
        )
        loader = DataLoader(ds, batch_size=50)

        all_emb = []
        all_cls = []
        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(device)
                class_seq = batch['class_seq']
                _, emb = model(features)  # [B, T, 64]
                all_emb.append(emb.cpu().reshape(-1, emb.shape[-1]))
                all_cls.append(class_seq.reshape(-1))

        all_emb = torch.cat(all_emb).numpy()
        all_cls = torch.cat(all_cls).numpy()

        # PCA to 3D
        pca = PCA(n_components=3)
        coords = pca.fit_transform(all_emb)

        # Subsample for clarity
        n_plot = min(3000, len(coords))
        rng = np.random.RandomState(42)
        sel = rng.choice(len(coords), n_plot, replace=False)

        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        is_rare = np.isin(all_cls[sel], RARE_CLASSES)

        # Common classes: small, transparent
        ax.scatter(coords[sel][~is_rare, 0], coords[sel][~is_rare, 1],
                   coords[sel][~is_rare, 2], c='steelblue', s=3, alpha=0.15)
        # Rare classes: larger, more visible
        ax.scatter(coords[sel][is_rare, 0], coords[sel][is_rare, 1],
                   coords[sel][is_rare, 2], c='crimson', s=8, alpha=0.5)

        # Plot centroids
        for c in range(N_CLASSES):
            mask = all_cls == c
            if mask.sum() > 0:
                centroid = coords[mask].mean(axis=0)
                color = 'darkred' if c in RARE_CLASSES else 'navy'
                ax.scatter(*centroid, c=color, s=60, marker='D', edgecolors='white', linewidth=0.5)

        ax.set_title(CONDITION_LABELS[cond], fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.set_zlabel('PC3', fontsize=8)

    fig.suptitle('3D PCA Embedding Space (blue=common, red=rare, diamonds=centroids)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_3d_embeddings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', default='runs_visual')
    parser.add_argument('--output_dir', default='figures')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = find_runs(args.runs_dir, seed=args.seed)
    print(f"Found runs: {list(runs.keys())}")

    fig_class_distributions(runs, args.output_dir)
    fig_learning_curves(runs, args.output_dir)
    fig_test_accuracy_all(runs, args.output_dir)
    fig_test_accuracy_rare(runs, args.output_dir)
    fig_test_accuracy_common(runs, args.output_dir)
    fig_3d_embeddings(runs, args.output_dir)

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
