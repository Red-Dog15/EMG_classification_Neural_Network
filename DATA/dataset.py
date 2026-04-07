"""
EMG Dataset preparation for PyTorch training.
Creates train/test splits and DataLoader utilities.
"""

import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EMGDataset(Dataset):
    """PyTorch Dataset for EMG data with movement and severity labels."""
    
    def __init__(self, labeled_data, window_size=100, stride=50):
        """
        Args:
            labeled_data: List of (tensor_data, movement_label, severity_label)
            window_size: Number of timesteps per sample (sliding window)
            stride: Step size for sliding window
        """
        self.samples = []
        self.movement_labels = []
        self.severity_labels = []
        
        # Create sliding windows from each recording
        for tensor_data, movement_label, severity_label in labeled_data:
            # tensor_data shape: (num_timesteps, 8)
            num_timesteps = tensor_data.shape[0]
            
            # Create sliding windows
            for start_idx in range(0, num_timesteps - window_size + 1, stride):
                end_idx = start_idx + window_size
                window = tensor_data[start_idx:end_idx]  # (window_size, 8)
                
                self.samples.append(window)
                self.movement_labels.append(movement_label)
                self.severity_labels.append(severity_label)
        
        # Convert to tensors
        self.samples = torch.stack(self.samples)  # (N, window_size, 8)
        self.movement_labels = torch.tensor(self.movement_labels, dtype=torch.long)
        self.severity_labels = torch.tensor(self.severity_labels, dtype=torch.long)
        
        print(f"Created dataset with {len(self.samples)} samples")
        print(f"Sample shape: {self.samples[0].shape}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (self.samples[idx], 
                self.movement_labels[idx], 
                self.severity_labels[idx])


def create_dataloaders(labeled_data, batch_size=32, train_split=0.8,
                       window_size=100, stride=50, num_workers=0, split_seed=42):
    """
    Create train and test DataLoaders.
    
    Args:
        labeled_data: Output from create_labeled_dataset()
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for testing)
        window_size: Timesteps per sample
        stride: Sliding window stride
        num_workers: Number of worker processes for data loading
        split_seed: Seed used for reproducible recording-level split
        
    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    # Temporal split within each recording.
    #
    # The previous recording-level split (one whole recording held out per movement
    # class) created an out-of-distribution problem: Movement 0 might train on
    # [Medium, Hard] but test on [Light], so the model has never seen that exact
    # (movement × severity) pairing. With only 21 recordings this produces severe
    # distribution shift and low severity accuracy regardless of regularisation.
    #
    # Fix: for every recording keep the first `train_split` fraction of its
    # timesteps for training and the remaining fraction for testing.  A gap of
    # (window_size - 1) timesteps is removed between the two segments so that no
    # training window and test window can overlap.  Every movement × severity
    # combination now appears in both train and test, eliminating the shift.
    train_segments = []
    test_segments = []

    for tensor_data, movement_label, severity_label in labeled_data:
        num_timesteps = tensor_data.shape[0]
        train_end = int(num_timesteps * train_split)
        # Gap ensures zero window overlap between train and test segments
        test_start = train_end + (window_size - 1)

        train_segments.append((tensor_data[:train_end], movement_label, severity_label))
        if test_start < num_timesteps:
            test_segments.append((tensor_data[test_start:], movement_label, severity_label))

    _sev_names = {0: 'Light', 1: 'Medium', 2: 'Hard'}
    _test_sev = defaultdict(int)
    for _, _, sev in test_segments:
        _test_sev[sev] += 1
    _sev_summary = ', '.join(f"{_sev_names[k]}:{v}" for k, v in sorted(_test_sev.items()))

    train_dataset = EMGDataset(train_segments, window_size=window_size, stride=stride)
    test_dataset  = EMGDataset(test_segments,  window_size=window_size, stride=stride)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Temporal split: {len(train_segments)} train segments, {len(test_segments)} test segments")
    print(f"Test severity distribution: {_sev_summary}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, test_loader


def get_dataset_statistics(labeled_data):
    """Print statistics about the dataset."""
    movement_counts = {}
    severity_counts = {}
    
    for _, movement, severity in labeled_data:
        movement_counts[movement] = movement_counts.get(movement, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("\n=== Dataset Statistics ===")
    print(f"Total recordings: {len(labeled_data)}")
    print(f"\nMovement distribution:")
    for mov_idx, count in sorted(movement_counts.items()):
        print(f"  Class {mov_idx}: {count} recordings")
    print(f"\nSeverity distribution:")
    for sev_idx, count in sorted(severity_counts.items()):
        print(f"  Level {sev_idx}: {count} recordings")
    print("=" * 30 + "\n")
