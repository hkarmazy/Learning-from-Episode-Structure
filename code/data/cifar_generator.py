"""CIFAR-100 visual play sequence generator.

Generates episodes using the same Zipf + small-world logic as
synthetic_generator.py, but each token is a pre-extracted CNN feature
vector from a real CIFAR-100 image.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic_generator import zipf_frequencies
from data.transition_graph import build_sw_graph, sw_transition_matrix
from data.cifar_features import extract_features


class CIFARPlaySequenceDataset(Dataset):
    """Visual play sequence dataset backed by CIFAR-100 features.

    Each episode is a sequence of 180 image feature vectors with
    corresponding class indices. Zipf and SW control token frequencies
    and transitions, just like PlaySequenceGenerator.

    Conditions:
        'uniform'  (C1): equal class frequencies, random i.i.d. transitions
        'zipf'     (C2): Zipf α=2.5 frequencies, random i.i.d. transitions
        'sw'       (C3): uniform frequencies, small-world transitions
        'zipf_sw'  (C4): Zipf α=2.5 frequencies + small-world transitions
    """

    def __init__(self, condition: str, n_episodes: int, n_classes: int = 32,
                 episode_len: int = 180, zipf_alpha: float = 2.5,
                 sw_beta: float = 0.15, seed: int = 42,
                 feature_cache_dir: str = 'data/cifar_cache',
                 split: str = 'train'):
        assert condition in ('uniform', 'zipf', 'sw', 'zipf_sw')
        self.condition = condition
        self.n_episodes = n_episodes
        self.n_classes = n_classes
        self.episode_len = episode_len
        self.rng = np.random.RandomState(seed)

        # Load pre-extracted features
        self.features_by_rank = extract_features(
            split=split, cache_dir=feature_cache_dir
        )

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
            self.transition = None

        # Pre-generate all episodes
        self.episodes = []
        for _ in range(n_episodes):
            class_seq = self._generate_class_sequence()
            feat_seq = self._sample_features(class_seq)
            self.episodes.append({
                'features': feat_seq,       # [T, 512]
                'class_seq': torch.tensor(class_seq, dtype=torch.long),  # [T]
            })

    def _generate_class_sequence(self) -> np.ndarray:
        """Generate a sequence of class indices following Zipf + SW rules."""
        seq = np.empty(self.episode_len, dtype=np.int64)

        # First token sampled from frequency distribution
        seq[0] = self.rng.choice(self.n_classes, p=self.freq)

        for t in range(1, self.episode_len):
            if self.transition is not None:
                base_probs = self.transition[seq[t - 1]]
                if self.condition == 'zipf_sw':
                    base_probs = base_probs * self.freq
                    base_probs = base_probs / base_probs.sum()
            else:
                base_probs = self.freq
            seq[t] = self.rng.choice(self.n_classes, p=base_probs)

        return seq

    def _sample_features(self, class_seq: np.ndarray) -> torch.Tensor:
        """For each class in the sequence, randomly sample an image feature."""
        feat_list = []
        for cls in class_seq:
            pool = self.features_by_rank[int(cls)]  # [N_images, 512]
            idx = self.rng.randint(0, pool.shape[0])
            feat_list.append(pool[idx])
        return torch.stack(feat_list)  # [T, 512]

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        return self.episodes[idx]

    def get_freq_vector(self) -> np.ndarray:
        return self.freq.copy()

    def get_transition_matrix(self):
        if self.transition is not None:
            return self.transition.copy()
        return None
