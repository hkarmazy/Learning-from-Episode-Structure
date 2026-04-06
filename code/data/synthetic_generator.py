"""Synthetic play sequence generator with Zipf and small-world modes.

Generates episodes mimicking child object play with controllable
frequency distribution (Zipf α=2.5) and transition topology (small-world).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from data.transition_graph import build_sw_graph, sw_transition_matrix


def zipf_frequencies(n_classes: int = 32, alpha: float = 2.5) -> np.ndarray:
    """Compute normalized Zipf frequency vector.

    Convention: f(k) ∝ 1/k^(1/α).  Higher α → more uniform.
    At α=2.5: exponent = 0.4, ratio class1/class32 ≈ 5.3×.

    Returns:
        Normalized frequency vector of length n_classes.
    """
    ranks = np.arange(1, n_classes + 1, dtype=np.float64)
    freqs = 1.0 / ranks ** (1.0 / alpha)
    return freqs / freqs.sum()


class PlaySequenceGenerator(Dataset):
    """Generate synthetic play episodes under four experimental conditions.

    Conditions:
        'uniform'  (C1): equal class frequencies, random i.i.d. transitions
        'zipf'     (C2): Zipf α=2.5 frequencies, random i.i.d. transitions
        'sw'       (C3): uniform frequencies, small-world transitions
        'zipf_sw'  (C4): Zipf α=2.5 frequencies + small-world transitions
    """

    def __init__(self, condition: str, n_episodes: int, n_classes: int = 32,
                 episode_len: int = 180, zipf_alpha: float = 2.5,
                 sw_beta: float = 0.15, focal_boost: float = 0.3,
                 seed: int = 42):
        assert condition in ('uniform', 'zipf', 'sw', 'zipf_sw')
        self.condition = condition
        self.n_episodes = n_episodes
        self.n_classes = n_classes
        self.episode_len = episode_len
        self.focal_boost = focal_boost
        self.rng = np.random.RandomState(seed)

        # Frequency distribution
        use_zipf = condition in ('zipf', 'zipf_sw')
        if use_zipf:
            self.freq = zipf_frequencies(n_classes, zipf_alpha)
        else:
            self.freq = np.ones(n_classes) / n_classes

        # Transition matrix
        use_sw = condition in ('sw', 'zipf_sw')
        if use_sw:
            graph = build_sw_graph(n_classes, beta=sw_beta, seed=seed)
            self.transition = sw_transition_matrix(graph)
        else:
            # Random i.i.d. — transitions are just re-sampling from freq
            self.transition = None

        # Pre-generate all episodes.
        # Each episode has a pre-assigned focal class (the label), sampled
        # from the frequency distribution. This ensures all 32 classes
        # appear as labels with the correct Zipf-distributed frequency.
        self.episodes = []
        for _ in range(n_episodes):
            focal_class = self.rng.choice(self.n_classes, p=self.freq)
            seq = self._generate_episode(focal_class)
            self.episodes.append({
                'sequence': torch.tensor(seq, dtype=torch.long),
                'label': torch.tensor(focal_class, dtype=torch.long),
            })

    def _generate_episode(self, focal_class: int) -> np.ndarray:
        """Generate a single episode with the focal class recurring throughout.

        The focal class has elevated contact frequency across the entire
        episode — modelling how a child repeatedly returns to their focal toy.
        At each timestep, there is a focal_boost probability of returning to
        the focal class; otherwise, the normal transition/frequency rule applies.

        Args:
            focal_class: The pre-assigned class label for this episode.
        """
        seq = np.empty(self.episode_len, dtype=np.int64)
        fb = self.focal_boost

        # First time step: always the focal class
        seq[0] = focal_class

        # One-hot for focal class (used to boost probabilities)
        focal_one_hot = np.zeros(self.n_classes)
        focal_one_hot[focal_class] = 1.0

        for t in range(1, self.episode_len):
            if self.transition is not None:
                # Small-world: transition from current class
                base_probs = self.transition[seq[t - 1]]
                # Modulate by Zipf frequencies if in zipf_sw mode
                if self.condition == 'zipf_sw':
                    base_probs = base_probs * self.freq
                    base_probs = base_probs / base_probs.sum()
            else:
                # i.i.d. sampling from frequency distribution
                base_probs = self.freq

            # Mix in focal class boost: child keeps returning to focal toy
            probs = (1.0 - fb) * base_probs + fb * focal_one_hot
            probs = probs / probs.sum()
            seq[t] = self.rng.choice(self.n_classes, p=probs)

        return seq

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        return self.episodes[idx]

    def get_freq_vector(self) -> np.ndarray:
        """Return the frequency distribution used for this condition."""
        return self.freq.copy()

    def get_transition_matrix(self) -> np.ndarray:
        """Return the transition matrix (or None for i.i.d. conditions)."""
        if self.transition is not None:
            return self.transition.copy()
        return None


if __name__ == '__main__':
    for cond in ('uniform', 'zipf', 'sw', 'zipf_sw'):
        ds = PlaySequenceGenerator(cond, n_episodes=10)
        sample = ds[0]
        print(f"{cond:>8s}: seq shape={sample['sequence'].shape}, "
              f"label={sample['label'].item()}, "
              f"freq range=[{ds.freq.min():.4f}, {ds.freq.max():.4f}]")
