"""One-time CIFAR-100 feature extraction using pretrained ResNet-18.

Downloads CIFAR-100, filters to the 32 selected classes, extracts
512-d feature vectors, and caches them to disk.

For training images, multiple augmented versions are extracted per image
(random crop, horizontal flip, color jitter) to increase visual diversity
when Zipf-frequent classes reuse images.

Usage:
    python -m data.cifar_features [--cache_dir data/cifar_cache] [--n_augment 6]
"""

import argparse
import os

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from data.cifar_class_selection import CIFAR100_CLASS_INDICES, CIFAR_IDX_TO_RANK, N_CLASSES


def build_feature_extractor(device):
    """Return a frozen ResNet-18 without the final FC layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  # remove classifier, output is 512-d
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _clean_transform():
    """Standard ImageNet-normalized transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _augment_transform():
    """Augmented transform: random crop, flip, color jitter, then normalize."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_features(split='train', cache_dir='data/cifar_cache',
                     device=None, n_augment=6):
    """Extract and cache ResNet-18 features for the 32 selected CIFAR-100 classes.

    For the train split, extracts 1 clean + (n_augment-1) augmented versions
    per image, giving n_augment * 500 = 3000 features per class (default).
    For the test split, extracts 1 clean version per image (100 per class).

    Args:
        split: 'train' or 'test'.
        cache_dir: Directory to save cached features.
        device: torch device.
        n_augment: Number of feature versions per train image (1 clean + n-1 augmented).

    Returns:
        Dict mapping rank (0-31) → tensor [N_images, 512].
    """
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    cache_path = os.path.join(cache_dir, f'{split}_features.pt')
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, map_location='cpu', weights_only=True)

    os.makedirs(cache_dir, exist_ok=True)

    is_train = (split == 'train')

    # For train: extract clean + augmented versions; for test: clean only
    clean_transform = _clean_transform()
    augment_transform = _augment_transform() if is_train else None
    n_passes = n_augment if is_train else 1  # 1 clean + (n-1) augmented for train

    # First load with clean transform to filter indices
    dataset_clean = datasets.CIFAR100(root=cache_dir, train=is_train,
                                      download=True, transform=clean_transform)
    selected_set = set(CIFAR100_CLASS_INDICES)
    filtered_indices = [i for i, (_, label) in enumerate(dataset_clean)
                        if label in selected_set]

    model = build_feature_extractor(device)
    features_by_rank = {r: [] for r in range(N_CLASSES)}

    total_images = len(filtered_indices) * n_passes
    print(f"Extracting {split} features: {len(filtered_indices)} images × "
          f"{n_passes} versions = {total_images} feature vectors...")

    for pass_idx in range(n_passes):
        if pass_idx == 0:
            # Clean pass (no augmentation)
            tfm = clean_transform
            label = "clean"
        else:
            tfm = augment_transform
            label = f"aug-{pass_idx}"

        dataset = datasets.CIFAR100(root=cache_dir, train=is_train,
                                    download=False, transform=tfm)
        subset = Subset(dataset, filtered_indices)
        loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)

        print(f"  Pass {pass_idx + 1}/{n_passes} ({label})...")
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                feats = model(images).cpu()  # [B, 512]
                for feat, lab in zip(feats, labels):
                    rank = CIFAR_IDX_TO_RANK[lab.item()]
                    features_by_rank[rank].append(feat)

    # Stack into tensors
    result = {}
    for rank in range(N_CLASSES):
        result[rank] = torch.stack(features_by_rank[rank])
        print(f"  Rank {rank:2d}: {result[rank].shape[0]} features")

    torch.save(result, cache_path)
    print(f"Saved to {cache_path}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', default='data/cifar_cache')
    parser.add_argument('--n_augment', type=int, default=6,
                        help='Number of feature versions per train image '
                             '(1 clean + n-1 augmented). Default: 6')
    args = parser.parse_args()

    for split in ('train', 'test'):
        n_aug = args.n_augment if split == 'train' else 1
        extract_features(split=split, cache_dir=args.cache_dir,
                         n_augment=n_aug)
