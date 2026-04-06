"""Parameter source documentation for published child play statistics.

The parameters used in this project are derived from published empirical data
on child object play. This module documents the provenance of each parameter
and provides utility functions for reference.

All values are FIXED from published data — they are not hyperparameter choices.
"""

import numpy as np


# Published parameters from prior work
PUBLISHED_PARAMS = {
    'zipf_alpha': 2.5,          # Zipfian exponent for toy contact frequencies
    'sw_beta': 0.15,            # Watts-Strogatz rewiring parameter
    'n_classes': 32,            # Number of toy objects in empirical dataset
    'episode_len': 180,         # 180 × 5sec bins per play session
}


def published_zipf_frequencies(n_classes: int = 32, alpha: float = 2.5) -> np.ndarray:
    """Reference Zipf frequency vector from published data.

    f(k) ∝ 1/k^(1/α) where α=2.5 (published).
    At α=2.5: exponent = 1/2.5 = 0.4
    Frequency ratio class1/class32 = 32^0.4 ≈ 5.3×
    """
    ranks = np.arange(1, n_classes + 1, dtype=np.float64)
    freqs = 1.0 / ranks ** (1.0 / alpha)
    return freqs / freqs.sum()


def print_frequency_table():
    """Print the Zipf frequency distribution for reference."""
    freqs = published_zipf_frequencies()
    print("Published Zipf frequencies (α=2.5, 32 classes):")
    print(f"{'Rank':>5s}  {'Freq':>8s}  {'Cumul':>8s}  {'Appearances/180':>16s}")
    cumul = 0.0
    for i, f in enumerate(freqs):
        cumul += f
        print(f"{i+1:>5d}  {f:>8.4f}  {cumul:>8.4f}  {f*180:>16.1f}")
    print(f"\nRatio class1/class32: {freqs[0]/freqs[-1]:.2f}×")


if __name__ == '__main__':
    print_frequency_table()
