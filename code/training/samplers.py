"""Custom samplers for Zipfian and small-world data loading."""

import numpy as np
from torch.utils.data import Sampler


class ZipfianSampler(Sampler):
    """Sampler that draws episodes with Zipfian frequency bias.

    Over-samples episodes whose label is a rare class,
    matching the Zipf α=2.5 distribution inversely (so rare classes
    appear more often in training batches).
    """

    def __init__(self, dataset, alpha: float = 2.5, seed: int = 42):
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        self.n = len(dataset)

        # Compute inverse-Zipf weights: rarer labels get higher sampling weight
        labels = np.array([dataset[i]['label'].item() for i in range(self.n)])
        unique, counts = np.unique(labels, return_counts=True)
        freq = np.zeros(self.n)
        label_freq = dict(zip(unique, counts))
        for i in range(self.n):
            freq[i] = label_freq.get(labels[i], 1)

        # Inverse frequency weighting
        self.weights = 1.0 / (freq + 1e-7)
        self.weights /= self.weights.sum()

    def __iter__(self):
        indices = self.rng.choice(self.n, size=self.n, replace=True, p=self.weights)
        return iter(indices.tolist())

    def __len__(self):
        return self.n


class SmallWorldSampler(Sampler):
    """Sampler that respects small-world transition structure.

    Samples episodes in an order that follows the SW transition graph:
    the next episode's label is drawn according to the transition
    probability from the current episode's label.
    """

    def __init__(self, dataset, transition_matrix: np.ndarray = None, seed: int = 42):
        self.dataset = dataset
        self.n = len(dataset)
        self.rng = np.random.RandomState(seed)
        self.transition = transition_matrix

        # Build label-to-indices mapping
        self.label_indices = {}
        for i in range(self.n):
            label = dataset[i]['label'].item()
            if label not in self.label_indices:
                self.label_indices[label] = []
            self.label_indices[label].append(i)

    def __iter__(self):
        indices = []
        # Start from a random episode
        current_idx = self.rng.randint(self.n)
        indices.append(current_idx)
        current_label = self.dataset[current_idx]['label'].item()

        for _ in range(self.n - 1):
            if self.transition is not None:
                # Transition to next label via SW graph
                probs = self.transition[current_label]
                next_label = self.rng.choice(len(probs), p=probs)
            else:
                next_label = self.rng.randint(max(self.label_indices.keys()) + 1)

            # Pick a random episode with that label
            if next_label in self.label_indices and self.label_indices[next_label]:
                candidates = self.label_indices[next_label]
                current_idx = candidates[self.rng.randint(len(candidates))]
            else:
                current_idx = self.rng.randint(self.n)

            indices.append(current_idx)
            current_label = self.dataset[current_idx]['label'].item()

        return iter(indices)

    def __len__(self):
        return self.n
