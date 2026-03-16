"""
EMG Dataset preparation for PyTorch training.
Creates train/test splits and DataLoader utilities.
"""

import random
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
    # Split at the recording level (stratified by movement class) to prevent
    # data leakage from overlapping windows spanning the train/test boundary.
    groups = defaultdict(list)
    for item in labeled_data:
        groups[item[1]].append(item)  # item[1] is movement_label

    rng = random.Random(split_seed)
    train_recordings, test_recordings = [], []
    for class_items in groups.values():
        shuffled = list(class_items)
        rng.shuffle(shuffled)
        # At least 1 test recording per class, at least 1 train recording per class
        n_test = max(1, min(len(shuffled) - 1, round(len(shuffled) * (1 - train_split))))
        test_recordings.extend(shuffled[:n_test])
        train_recordings.extend(shuffled[n_test:])

    train_dataset = EMGDataset(train_recordings, window_size=window_size, stride=stride)
    test_dataset  = EMGDataset(test_recordings,  window_size=window_size, stride=stride)
    
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
    
    print(f"Train recordings: {len(train_recordings)}, Test recordings: {len(test_recordings)}")
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
