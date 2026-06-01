"""
Utility functions for CompBioChem project.
"""

import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        path: Checkpoint path
        
    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


def save_results(results: Dict, path: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary of results
        path: Save path
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> Dict:
    """
    Load experiment results from JSON file.
    
    Args:
        path: Results file path
        
    Returns:
        Dictionary of results
    """
    with open(path, 'r') as f:
        results = json.load(f)
    return results


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(log_dir: str, experiment_name: str):
    """
    Setup logging directory.
    
    Args:
        log_dir: Base log directory
        experiment_name: Name of experiment
        
    Returns:
        Path to experiment log directory
    """
    exp_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
