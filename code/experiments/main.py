"""
Main experiment script for CompBioChem project.
Target journal: Computational Biology and Chemistry (Elsevier, IF 3.29)
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import random
from pathlib import Path


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "cuda") -> torch.device:
    """Get PyTorch device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(args):
    """Main function."""
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['experiment']['seed'])
    
    # Get device
    device = get_device(config['experiment']['device'])
    print(f"Using device: {device}")
    
    # Create output directories
    for dir_key in ['checkpoint_dir', 'log_dir', 'result_dir']:
        os.makedirs(config['output'][dir_key], exist_ok=True)
    
    print("Experiment setup complete!")
    print(f"Experiment name: {config['experiment']['name']}")
    
    # TODO: Add your experiment code here
    # 1. Load data
    # 2. Build model
    # 3. Train model
    # 4. Evaluate model
    # 5. Save results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CompBioChem Experiment")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "test"],
                        help="Experiment mode")
    args = parser.parse_args()
    
    main(args)
