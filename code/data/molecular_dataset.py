"""
Molecular property prediction datasets for CompBioChem.

Implements synthetic molecular datasets that simulate real-world benchmarks
(BBBP, BACE, HIV, Tox21) for testing the MAGNet architecture.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import json


class SyntheticMolecularDataset(Dataset):
    """
    Synthetic molecular dataset for property prediction.

    Generates molecular graphs with realistic structural patterns:
    - Variable number of atoms (5-50)
    - Realistic bond patterns (single, double, triple)
    - Atom features based on element type
    - Property labels correlated with molecular structure

    Args:
        n_molecules: Number of molecules to generate.
        property_type: Type of property to predict ('toxicity', 'solubility', 'binding').
        difficulty: Difficulty level ('easy', 'medium', 'hard').
        seed: Random seed for reproducibility.
    """

    # Element features (simplified)
    ELEMENT_FEATURES = {
        'C': [1, 0, 0, 0, 0, 0, 0, 0, 0],   # Carbon
        'N': [0, 1, 0, 0, 0, 0, 0, 0, 0],   # Nitrogen
        'O': [0, 0, 1, 0, 0, 0, 0, 0, 0],   # Oxygen
        'S': [0, 0, 0, 1, 0, 0, 0, 0, 0],   # Sulfur
        'F': [0, 0, 0, 0, 1, 0, 0, 0, 0],   # Fluorine
        'Cl': [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Chlorine
        'Br': [0, 0, 0, 0, 0, 0, 1, 0, 0],  # Bromine
        'I': [0, 0, 0, 0, 0, 0, 0, 1, 0],   # Iodine
        'P': [0, 0, 0, 0, 0, 0, 0, 0, 1],   # Phosphorus
    }

    # Bond types
    BOND_TYPES = {
        'single': [1, 0, 0, 0],
        'double': [0, 1, 0, 0],
        'triple': [0, 0, 1, 0],
        'aromatic': [0, 0, 0, 1],
    }

    def __init__(
        self,
        n_molecules: int = 1000,
        property_type: str = 'toxicity',
        difficulty: str = 'medium',
        seed: int = 42,
    ):
        self.n_molecules = n_molecules
        self.property_type = property_type
        self.difficulty = difficulty

        rng = np.random.RandomState(seed)

        # Generate molecules
        self.node_features_list = []
        self.adjacency_list = []
        self.labels = []
        self.mol_sizes = []

        # Property-specific parameters
        prop_params = self._get_property_params(property_type, difficulty)

        for i in range(n_molecules):
            # Random molecule size
            n_atoms = rng.randint(5, 51)

            # Generate molecular structure
            node_features, adj = self._generate_molecule(n_atoms, rng, prop_params)

            # Generate label based on structural features
            label = self._compute_label(node_features, adj, prop_params, rng)

            self.node_features_list.append(node_features)
            self.adjacency_list.append(adj)
            self.labels.append(label)
            self.mol_sizes.append(n_atoms)

        self.labels = np.array(self.labels, dtype=np.int64)
        self.max_nodes = max(self.mol_sizes)

    def _get_property_params(self, property_type: str, difficulty: str) -> Dict:
        """Get parameters for different property types."""
        params = {
            'toxicity': {
                'toxic_elements': ['Cl', 'Br', 'I'],
                'toxic_motifs': ['aromatic'],
                'noise_level': 0.1 if difficulty == 'easy' else 0.2 if difficulty == 'medium' else 0.3,
            },
            'solubility': {
                'hydrophilic': ['N', 'O'],
                'hydrophobic': ['C', 'S'],
                'noise_level': 0.1 if difficulty == 'easy' else 0.2 if difficulty == 'medium' else 0.3,
            },
            'binding': {
                'binding_elements': ['N', 'O', 'S'],
                'noise_level': 0.1 if difficulty == 'easy' else 0.2 if difficulty == 'medium' else 0.3,
            },
        }
        return params.get(property_type, params['toxicity'])

    def _generate_molecule(
        self,
        n_atoms: int,
        rng: np.random.RandomState,
        prop_params: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single molecular graph."""
        elements = list(self.ELEMENT_FEATURES.keys())

        # Select elements (biased by property type)
        if self.property_type == 'toxicity':
            weights = [0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        elif self.property_type == 'solubility':
            weights = [0.4, 0.2, 0.2, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        else:  # binding
            weights = [0.5, 0.15, 0.15, 0.1, 0.03, 0.03, 0.02, 0.01, 0.01]

        selected_elements = rng.choice(elements, size=n_atoms, p=weights)

        # Create node features
        node_features = np.array([
            self.ELEMENT_FEATURES[elem] + [
                rng.randint(0, 4),  # Hybridization
                rng.randint(0, 2),  # Aromatic
                rng.randint(0, 5),  # Degree
                rng.randint(0, 3),  # Formal charge
                rng.randint(0, 4),  # Num H
            ]
            for elem in selected_elements
        ], dtype=np.float32)

        # Generate adjacency matrix (tree-like + some rings)
        adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)

        # Create a connected graph (tree backbone)
        for i in range(1, n_atoms):
            parent = rng.randint(0, i)
            bond_type = rng.choice(['single', 'double', 'triple', 'aromatic'],
                                   p=[0.7, 0.15, 0.05, 0.1])
            bond_feat = self.BOND_TYPES[bond_type]
            adj[parent, i] = 1.0
            adj[i, parent] = 1.0

        # Add some cycles (rings)
        n_rings = rng.randint(0, min(3, max(1, n_atoms // 5)))
        for _ in range(n_rings):
            if n_atoms >= 5:
                start = rng.randint(0, n_atoms - 3)
                max_ring = min(6, n_atoms - start)
                if max_ring >= 3:
                    ring_size = rng.randint(3, max_ring + 1)
                    for j in range(ring_size):
                        next_j = (j + 1) % ring_size
                        adj[start + j, start + next_j] = 1.0
                        adj[start + next_j, start + j] = 1.0

        return node_features, adj

    def _compute_label(
        self,
        node_features: np.ndarray,
        adj: np.ndarray,
        prop_params: Dict,
        rng: np.random.RandomState,
    ) -> int:
        """Compute property label based on molecular features."""
        n_atoms = node_features.shape[0]

        if self.property_type == 'toxicity':
            # Toxicity correlates with halogen content and aromaticity
            halogen_cols = [4, 5, 6, 7]  # F, Cl, Br, I
            halogen_count = node_features[:, halogen_cols].sum()
            aromatic_count = node_features[:, 9].sum()  # Aromatic feature
            score = (halogen_count * 2 + aromatic_count) / (n_atoms + 1)
            threshold = 0.3

        elif self.property_type == 'solubility':
            # Solubility correlates with N, O content
            hydrophilic_cols = [1, 2]  # N, O
            hydrophilic_count = node_features[:, hydrophilic_cols].sum()
            score = hydrophilic_count / (n_atoms + 1)
            threshold = 0.2

        else:  # binding
            # Binding correlates with heteroatom content
            heteroatom_cols = [1, 2, 3]  # N, O, S
            heteroatom_count = node_features[:, heteroatom_cols].sum()
            score = heteroatom_count / (n_atoms + 1)
            threshold = 0.25

        # Add noise
        if rng.rand() < prop_params['noise_level']:
            score = 1.0 - score

        return int(score > threshold)

    def __len__(self) -> int:
        return self.n_molecules

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
        padded_nf = np.zeros((self.max_nodes, nf.shape[1]), dtype=np.float32)
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

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return {
            'n_molecules': self.n_molecules,
            'property_type': self.property_type,
            'difficulty': self.difficulty,
            'avg_atoms': np.mean(self.mol_sizes),
            'std_atoms': np.std(self.mol_sizes),
            'min_atoms': min(self.mol_sizes),
            'max_atoms': max(self.mol_sizes),
            'positive_ratio': self.labels.mean(),
            'node_dim': self.node_features_list[0].shape[1],
        }


def create_molecular_datasets(
    n_train: int = 2000,
    n_val: int = 400,
    n_test: int = 400,
    property_type: str = 'toxicity',
    difficulty: str = 'medium',
    seed: int = 42,
) -> Tuple[SyntheticMolecularDataset, SyntheticMolecularDataset, SyntheticMolecularDataset]:
    """
    Create train/val/test molecular datasets.

    Args:
        n_train: Number of training molecules.
        n_val: Number of validation molecules.
        n_test: Number of test molecules.
        property_type: Type of property to predict.
        difficulty: Difficulty level.
        seed: Base random seed.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = SyntheticMolecularDataset(
        n_molecules=n_train,
        property_type=property_type,
        difficulty=difficulty,
        seed=seed,
    )

    val_dataset = SyntheticMolecularDataset(
        n_molecules=n_val,
        property_type=property_type,
        difficulty=difficulty,
        seed=seed + 1,
    )

    test_dataset = SyntheticMolecularDataset(
        n_molecules=n_test,
        property_type=property_type,
        difficulty=difficulty,
        seed=seed + 2,
    )

    return train_dataset, val_dataset, test_dataset
