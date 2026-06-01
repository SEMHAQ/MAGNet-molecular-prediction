"""
Data loading utilities for CompBioChem project.

Includes:
- BiologicalDataset: Abstract base for biological data (placeholder)
- SampleBiologicalDataset: Synthetic demo dataset for testing pipelines
- MolecularGraphDataset: Graph-structured molecular data for GNN models
- create_dataloaders: Helper for train/val/test DataLoader creation
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from pathlib import Path


class BiologicalDataset(Dataset):
    """
    Base dataset for biological data.

    This is a placeholder. Subclass and implement __getitem__ for your data.
    """

    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = None
        self.labels = None

    def __len__(self) -> int:
        if self.data is None:
            return 0
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Implement data loading logic in a subclass")


class SampleBiologicalDataset(Dataset):
    """
    Synthetic biological dataset for pipeline testing and development.

    Generates random feature vectors with binary labels. Useful for verifying
    the training loop, metrics, and visualization code before real data is available.

    Args:
        n_samples: Number of samples to generate.
        n_features: Feature dimension.
        n_classes: Number of classes (default: 2 for binary classification).
        noise: Label noise ratio (0.0 = clean, 0.5 = maximum noise).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 128,
        n_classes: int = 2,
        noise: float = 0.1,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)

        # Generate cluster-based data: each class has a distinct centroid
        centroids = rng.randn(n_classes, n_features).astype(np.float32)
        samples_per_class = n_samples // n_classes

        features_list = []
        labels_list = []
        for c in range(n_classes):
            n = samples_per_class if c < n_classes - 1 else n_samples - len(labels_list)
            class_data = centroids[c] + rng.randn(n, n_features).astype(np.float32) * 0.5
            features_list.append(class_data)
            labels_list.append(np.full(n, c, dtype=np.int64))

        self.data = np.vstack(features_list)
        self.labels = np.concatenate(labels_list)

        # Apply label noise
        if noise > 0:
            n_flip = int(len(self.labels) * noise)
            flip_idx = rng.choice(len(self.labels), n_flip, replace=False)
            self.labels[flip_idx] = 1 - self.labels[flip_idx]

        # Shuffle
        perm = rng.permutation(len(self.labels))
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


class MolecularGraphDataset(Dataset):
    """
    Dataset for molecular graph data (GNN input format).

    Stores node features, adjacency matrices, and graph-level labels.
    Suitable for molecular property prediction tasks where molecules are
    represented as graphs (atoms = nodes, bonds = edges).

    Args:
        node_features_list: List of node feature arrays, one per molecule.
            Each array has shape (n_atoms, node_feature_dim).
        adjacency_list: List of adjacency matrices, one per molecule.
            Each array has shape (n_atoms, n_atoms).
        labels: Array of graph-level labels, shape (n_molecules,).
        max_nodes: Maximum number of nodes (for padding). None = use max in data.
    """

    def __init__(
        self,
        node_features_list: List[np.ndarray],
        adjacency_list: List[np.ndarray],
        labels: np.ndarray,
        max_nodes: Optional[int] = None,
    ):
        assert len(node_features_list) == len(adjacency_list) == len(labels)
        self.node_features_list = node_features_list
        self.adjacency_list = adjacency_list
        self.labels = labels

        self.max_nodes = max_nodes or max(
            nf.shape[0] for nf in node_features_list
        )
        self.node_dim = node_features_list[0].shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_features: (max_nodes, node_dim) - zero-padded.
            adj: (max_nodes, max_nodes) - zero-padded with self-loops.
            label: scalar tensor.
        """
        nf = self.node_features_list[idx]
        adj = self.adjacency_list[idx]
        n = nf.shape[0]

        # Pad node features
        padded_nf = np.zeros((self.max_nodes, self.node_dim), dtype=np.float32)
        padded_nf[:n, :] = nf

        # Pad adjacency and add self-loops
        padded_adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        padded_adj[:n, :n] = adj
        np.fill_diagonal(padded_adj[:n, :n], 1.0)  # self-loops

        return (
            torch.tensor(padded_nf, dtype=torch.float32),
            torch.tensor(padded_adj, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    @staticmethod
    def generate_synthetic(
        n_molecules: int = 500,
        min_atoms: int = 5,
        max_atoms: int = 20,
        node_dim: int = 9,
        n_classes: int = 2,
        seed: int = 42,
    ) -> "MolecularGraphDataset":
        """
        Generate a synthetic molecular graph dataset for testing.

        Creates random molecular graphs with atom-like features and bond-like
        adjacency matrices. Labels are partially determined by graph size
        to simulate a learnable pattern.

        Args:
            n_molecules: Number of molecules.
            min_atoms: Minimum atoms per molecule.
            max_atoms: Maximum atoms per molecule.
            node_dim: Node feature dimension (default 9 for basic atom features).
            n_classes: Number of classes.
            seed: Random seed.

        Returns:
            MolecularGraphDataset instance.
        """
        rng = np.random.RandomState(seed)

        node_features_list = []
        adjacency_list = []
        labels = []

        for i in range(n_molecules):
            n_atoms = rng.randint(min_atoms, max_atoms + 1)

            # Random node features (simulating atom descriptors)
            nf = rng.randn(n_atoms, node_dim).astype(np.float32)

            # Random adjacency (sparse graph, ~30% edge probability)
            adj = (rng.rand(n_atoms, n_atoms) < 0.3).astype(np.float32)
            adj = np.triu(adj, k=1)
            adj = adj + adj.T  # symmetric

            node_features_list.append(nf)
            adjacency_list.append(adj)

            # Label based on graph size + noise (learnable pattern)
            label = int(n_atoms > (min_atoms + max_atoms) / 2)
            if rng.rand() < 0.1:  # 10% noise
                label = 1 - label
            labels.append(label)

        return MolecularGraphDataset(
            node_features_list=node_features_list,
            adjacency_list=adjacency_list,
            labels=np.array(labels, dtype=np.int64),
            max_nodes=max_atoms,
        )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_bioinformatics_data(data_dir: str) -> dict:
    """
    Load bioinformatics data from directory.

    Args:
        data_dir: Directory containing data files.

    Returns:
        Dictionary containing loaded data.
    """
    data = {}

    # TODO: Implement actual data loading
    # Example:
    # data['sequences'] = np.load(os.path.join(data_dir, 'sequences.npy'))
    # data['features'] = np.load(os.path.join(data_dir, 'features.npy'))
    # data['labels'] = np.load(os.path.join(data_dir, 'labels.npy'))

    return data
