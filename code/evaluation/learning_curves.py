"""Learning curve plotting from CSV logs (Stream 2).

Generates the 2x2 panel figure:
  Top-left:    val_acc_rare for all 4 conditions vs epoch (main figure)
  Top-right:   val_acc_all for all 4 conditions vs epoch
  Bottom-left: loss curves per condition
  Bottom-right: surprise_value over training (C4 only)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONDITION_LABELS = {
    'uniform': 'C1: Uniform',
    'zipf': 'C2: Zipfian',
    'sw': 'C3: Small-World',
    'zipf_sw': 'C4: Zipf+SW',
}
CONDITION_COLORS = {
    'uniform': '#888888',
    'zipf': '#e6550d',
    'sw': '#3182bd',
    'zipf_sw': '#e7298a',
}


def load_metrics(runs_dir: str, conditions: list = None, seeds: list = None):
    """Load metrics CSVs from all runs.

    Returns:
        Dict[condition] -> list of DataFrames (one per seed).
    """
    if conditions is None:
        conditions = ['uniform', 'zipf', 'sw', 'zipf_sw']
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    data = {}
    for cond in conditions:
        data[cond] = []
        for seed in seeds:
            csv_path = os.path.join(runs_dir, f'{cond}_seed{seed}', 'metrics.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                data[cond].append(df)
    return data


def plot_learning_curves(data: dict, output_path: str = None):
    """Generate the 2x2 learning curve panel.

    Args:
        data: Dict[condition] -> list of DataFrames.
        output_path: If provided, save figure to this path.
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('val_acc_rare', 'Validation Accuracy (Rare Classes)', axes[0, 0]),
        ('val_acc_all', 'Validation Accuracy (All Classes)', axes[0, 1]),
        ('train_loss', 'Training Loss', axes[1, 0]),
    ]

    for metric, title, ax in metrics:
        for cond in ['uniform', 'zipf', 'sw', 'zipf_sw']:
            if cond not in data or len(data[cond]) == 0:
                continue

            # Stack all seeds and compute mean/std
            dfs = data[cond]
            values = np.array([df[metric].values for df in dfs if metric in df.columns])
            if len(values) == 0:
                continue

            epochs = dfs[0]['epoch'].values
            mean = values.mean(axis=0)
            std = values.std(axis=0)

            color = CONDITION_COLORS[cond]
            label = CONDITION_LABELS[cond]
            ax.plot(epochs, mean, color=color, label=label, linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.15)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)

    # Bottom-right: surprise value (C4 only)
    ax = axes[1, 1]
    if 'zipf_sw' in data and len(data['zipf_sw']) > 0:
        dfs = data['zipf_sw']
        values = np.array([df['surprise_value'].values for df in dfs
                           if 'surprise_value' in df.columns])
        if len(values) > 0:
            epochs = dfs[0]['epoch'].values
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            ax.plot(epochs, mean, color=CONDITION_COLORS['zipf_sw'], linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=CONDITION_COLORS['zipf_sw'], alpha=0.15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Surprise Multiplier')
    ax.set_title('Surprise Value (C4: Zipf+SW)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='surprise=1')
    ax.legend(fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves to {output_path}")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', type=str, default='runs')
    parser.add_argument('--output', type=str, default='figures/learning_curves.png')
    args = parser.parse_args()

    data = load_metrics(args.runs_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_learning_curves(data, args.output)
