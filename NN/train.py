"""
Training script for EMG Multi-Task Classification Model.
Trains a model to predict both movement type and contraction severity.
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from datetime import datetime

# Optional tensorboard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Training metrics will only be printed to console.")
    print("Install with: pip install tensorboard")

from DATA.Data_Conversion import create_labeled_dataset, get_num_classes, MOVEMENT_LABELS, SEVERITY_LABELS
from DATA.dataset import create_dataloaders, get_dataset_statistics
from NN.network import create_model

# Import shared configuration to ensure consistency with evaluation
try:
    from config import (
        WINDOW_SIZE, STRIDE, TRAIN_SPLIT, SPLIT_SEED,
        NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
        MOVEMENT_LOSS_WEIGHT, SEVERITY_LOSS_WEIGHT,
        EARLY_STOPPING_MONITOR, EARLY_STOPPING_MIN_DELTA
    )
    print(f"✓ Using shared config: window_size={WINDOW_SIZE}, stride={STRIDE}")
except ImportError:
    # Fallback defaults if config.py not found
    WINDOW_SIZE = 100
    STRIDE = 50
    TRAIN_SPLIT = 0.8
    SPLIT_SEED = 42
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    MOVEMENT_LOSS_WEIGHT = 1.0
    SEVERITY_LOSS_WEIGHT = 1.0
    EARLY_STOPPING_MONITOR = 'loss'
    EARLY_STOPPING_MIN_DELTA = 0.0
    print("⚠ config.py not found, using default settings")

SCRIPTS_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPTS_ROOT, 'NN', 'models')


class MultiTaskLoss(nn.Module):
    """Combined loss for movement and severity prediction."""
    
    def __init__(self, movement_weight=1.0, severity_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.movement_loss = nn.CrossEntropyLoss()
        self.severity_loss = nn.CrossEntropyLoss()
        self.movement_weight = movement_weight
        self.severity_weight = severity_weight
        
    def forward(self, movement_logits, severity_logits, movement_targets, severity_targets):
        loss_movement = self.movement_loss(movement_logits, movement_targets)
        loss_severity = self.severity_loss(severity_logits, severity_targets)
        
        total_loss = (self.movement_weight * loss_movement + 
                     self.severity_weight * loss_severity)
        
        return total_loss, loss_movement, loss_severity


def calculate_accuracy(logits, targets):
    """Calculate classification accuracy."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).float().sum()
    accuracy = correct / len(targets)
    return accuracy.item()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_movement_loss = 0.0
    total_severity_loss = 0.0
    total_movement_acc = 0.0
    total_severity_acc = 0.0
    num_batches = 0
    
    for batch_idx, (data, movement_labels, severity_labels) in enumerate(train_loader):
        data = data.to(device)
        movement_labels = movement_labels.to(device)
        severity_labels = severity_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        movement_logits, severity_logits = model(data)
        
        # Calculate loss
        loss, loss_movement, loss_severity = criterion(
            movement_logits, severity_logits,
            movement_labels, severity_labels
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracies
        movement_acc = calculate_accuracy(movement_logits, movement_labels)
        severity_acc = calculate_accuracy(severity_logits, severity_labels)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_movement_loss += loss_movement.item()
        total_severity_loss += loss_severity.item()
        total_movement_acc += movement_acc
        total_severity_acc += severity_acc
        num_batches += 1
    
    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'movement_loss': total_movement_loss / num_batches,
        'severity_loss': total_severity_loss / num_batches,
        'movement_acc': total_movement_acc / num_batches,
        'severity_acc': total_severity_acc / num_batches
    }
    
    return avg_metrics


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    
    total_loss = 0.0
    total_movement_loss = 0.0
    total_severity_loss = 0.0
    total_movement_acc = 0.0
    total_severity_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, movement_labels, severity_labels in test_loader:
            data = data.to(device)
            movement_labels = movement_labels.to(device)
            severity_labels = severity_labels.to(device)
            
            # Forward pass
            movement_logits, severity_logits = model(data)
            
            # Calculate loss
            loss, loss_movement, loss_severity = criterion(
                movement_logits, severity_logits,
                movement_labels, severity_labels
            )
            
            # Calculate accuracies
            movement_acc = calculate_accuracy(movement_logits, movement_labels)
            severity_acc = calculate_accuracy(severity_logits, severity_labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_movement_loss += loss_movement.item()
            total_severity_loss += loss_severity.item()
            total_movement_acc += movement_acc
            total_severity_acc += severity_acc
            num_batches += 1
    
    # Average metrics
    avg_metrics = {
        'loss': total_loss / num_batches,
        'movement_loss': total_movement_loss / num_batches,
        'severity_loss': total_severity_loss / num_batches,
        'movement_acc': total_movement_acc / num_batches,
        'severity_acc': total_severity_acc / num_batches
    }
    
    return avg_metrics


def train_model(
    model_type='full',
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_monitor=EARLY_STOPPING_MONITOR,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    movement_loss_weight=MOVEMENT_LOSS_WEIGHT,
    severity_loss_weight=SEVERITY_LOSS_WEIGHT,
    window_size=WINDOW_SIZE,  # From shared config
    stride=STRIDE,            # From shared config
    train_split=TRAIN_SPLIT,  # From shared config
    save_dir=None,
    device=None
):
    """
    Main training function.
    
    Args:
        model_type: 'full', 'standard_cnn', or 'lightweight'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Stop training if no test-loss improvement for N epochs
        early_stopping_monitor: Metric used for early stopping and legacy best checkpoint
        early_stopping_min_delta: Minimum absolute improvement to reset patience
        movement_loss_weight: Weight for movement classification loss
        severity_loss_weight: Weight for severity classification loss
        window_size: Timesteps per sample
        stride: Sliding window stride
        train_split: Train/test split ratio
        save_dir: Directory to save models (defaults to Scripts/NN/models)
        device: Device to train on (auto-detect if None)
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if save_dir is None:
        save_dir = DEFAULT_MODEL_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and prepare data
    print("\n=== Loading Data ===")
    labeled_data = create_labeled_dataset()
    get_dataset_statistics(labeled_data)
    
    train_loader, test_loader = create_dataloaders(
        labeled_data,
        batch_size=batch_size,
        train_split=train_split,
        window_size=window_size,
        stride=stride,
        num_workers=0,
        split_seed=SPLIT_SEED
    )
    
    # Create model
    print("\n=== Creating Model ===")
    num_movements, num_severities = get_num_classes()
    model = create_model(
        model_type=model_type,
        num_channels=8,
        num_movements=num_movements,
        num_severities=num_severities
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = MultiTaskLoss(
        movement_weight=movement_loss_weight,
        severity_weight=severity_loss_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard logging (optional)
    writer = None
    log_dir = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'./runs/emg_training_{timestamp}'
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled: {log_dir}")
    
    # Training loop
    print("\n=== Starting Training ===")
    best_by_metric = {
        'loss': {'value': float('inf'), 'epoch': -1, 'metrics': None, 'mode': 'min'},
        'movement_acc': {'value': float('-inf'), 'epoch': -1, 'metrics': None, 'mode': 'max'},
        'severity_acc': {'value': float('-inf'), 'epoch': -1, 'metrics': None, 'mode': 'max'},
    }

    if early_stopping_monitor not in best_by_metric:
        raise ValueError(
            f"Invalid early_stopping_monitor '{early_stopping_monitor}'. "
            f"Choose from {list(best_by_metric.keys())}."
        )

    def _is_improved(metric_name, current_value):
        mode = best_by_metric[metric_name]['mode']
        best_value = best_by_metric[metric_name]['value']
        delta = early_stopping_min_delta
        if mode == 'min':
            return current_value < (best_value - delta)
        return current_value > (best_value + delta)

    def _build_checkpoint(epoch_idx, metrics_dict):
        return {
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_metrics': metrics_dict,
            'model_type': model_type,
            'window_size': window_size,
            'num_movements': num_movements,
            'num_severities': num_severities,
            'movement_loss_weight': movement_loss_weight,
            'severity_loss_weight': severity_loss_weight,
            'early_stopping_monitor': early_stopping_monitor,
            'early_stopping_min_delta': early_stopping_min_delta,
        }

    epochs_without_improvement = 0
    train_start_time = time.perf_counter()
    
    final_epoch = 0
    for epoch in range(num_epochs):
        final_epoch = epoch + 1
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(test_metrics['loss'])
        
        # Logging
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Movement Acc: {train_metrics['movement_acc']:.4f}, "
              f"Severity Acc: {train_metrics['severity_acc']:.4f}")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"Movement Acc: {test_metrics['movement_acc']:.4f}, "
              f"Severity Acc: {test_metrics['severity_acc']:.4f}")
        
        # TensorBoard logging (if available)
        if writer is not None:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/test', test_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train_movement', train_metrics['movement_acc'], epoch)
            writer.add_scalar('Accuracy/test_movement', test_metrics['movement_acc'], epoch)
            writer.add_scalar('Accuracy/train_severity', train_metrics['severity_acc'], epoch)
            writer.add_scalar('Accuracy/test_severity', test_metrics['severity_acc'], epoch)
        
        # Save best model for each key metric
        for metric_name in best_by_metric:
            current_value = test_metrics[metric_name]
            if _is_improved(metric_name, current_value):
                best_by_metric[metric_name]['value'] = current_value
                best_by_metric[metric_name]['epoch'] = epoch + 1
                best_by_metric[metric_name]['metrics'] = copy.deepcopy(test_metrics)
                checkpoint = _build_checkpoint(epoch, test_metrics)
                save_path = os.path.join(save_dir, f'best_model_{model_type}_by_{metric_name}.pth')
                torch.save(checkpoint, save_path)
                print(f"Saved best-by-{metric_name} model to {save_path}")

        # Backward-compatible alias tracks the selected early-stopping monitor
        if best_by_metric[early_stopping_monitor]['epoch'] == epoch + 1:
            alias_checkpoint = _build_checkpoint(epoch, test_metrics)
            alias_path = os.path.join(save_dir, f'best_model_{model_type}.pth')
            torch.save(alias_checkpoint, alias_path)
            print(f"Updated monitor-best alias to {alias_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(no improvement for {early_stopping_patience} epochs)"
            )
            break

    total_training_time_sec = time.perf_counter() - train_start_time
    
    # Save final model
    final_checkpoint = {
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
        'best_by_metric': best_by_metric,
        'best_test_metrics': best_by_metric[early_stopping_monitor]['metrics'],
        'best_test_loss': best_by_metric['loss']['value'],
        'best_epoch': best_by_metric[early_stopping_monitor]['epoch'],
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_monitor': early_stopping_monitor,
        'early_stopping_min_delta': early_stopping_min_delta,
        'movement_loss_weight': movement_loss_weight,
        'severity_loss_weight': severity_loss_weight,
        'training_time_sec': total_training_time_sec,
        'model_type': model_type,
        'window_size': window_size,
        'num_movements': num_movements,
        'num_severities': num_severities
    }
    final_path = os.path.join(save_dir, f'final_model_{model_type}.pth')
    torch.save(final_checkpoint, final_path)
    
    if writer is not None:
        writer.close()
    
    print(f"\n=== Training Complete ===")
    print(f"Optimization objective: {early_stopping_monitor}")
    print(
        f"Loss weights -> movement: {movement_loss_weight:.2f}, "
        f"severity: {severity_loss_weight:.2f}"
    )
    print(f"Best test loss: {best_by_metric['loss']['value']:.4f} (epoch {best_by_metric['loss']['epoch']})")
    print(
        f"Best movement accuracy: {best_by_metric['movement_acc']['value']*100:.2f}% "
        f"(epoch {best_by_metric['movement_acc']['epoch']})"
    )
    print(
        f"Best severity accuracy: {best_by_metric['severity_acc']['value']*100:.2f}% "
        f"(epoch {best_by_metric['severity_acc']['epoch']})"
    )
    selected_best = best_by_metric[early_stopping_monitor]['metrics']
    if selected_best is not None:
        print("Selected best checkpoint metrics:")
        print(f"  Movement Accuracy: {selected_best['movement_acc']*100:.2f}%")
        print(f"  Severity Accuracy: {selected_best['severity_acc']*100:.2f}%")
    print(f"Total training time: {total_training_time_sec:.2f}s ({total_training_time_sec/60:.2f} min)")
    print(f"Models saved to: {save_dir}")
    if log_dir is not None:
        print(f"TensorBoard logs: {log_dir}")
    
    return model, (best_by_metric[early_stopping_monitor]['metrics'] or test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMG multi-task model")
    parser.add_argument(
        "--model-type",
        choices=["full", "standard_cnn", "lightweight", "nn_a", "nn_b", "nn_c"],
        default=None,
        help="Model architecture to train"
    )
    args = parser.parse_args()

    if args.model_type is None:
        print("Select model architecture to train:")
        print("1. NN-A: full (CNN+GRU)")
        print("2. NN-B: standard_cnn (CNN-only)")
        print("3. NN-C: lightweight")

        selected = input("Enter choice (1-3) or press Enter for 1: ").strip()
        model_map = {
            "1": "full",
            "2": "standard_cnn",
            "3": "lightweight",
            "": "full"
        }
        model_type = model_map.get(selected, "full")
    else:
        alias_map = {
            "nn_a": "full",
            "nn_b": "standard_cnn",
            "nn_c": "lightweight"
        }
        model_type = alias_map.get(args.model_type, args.model_type)

    print(f"\nTraining model type: {model_type}")
    print(f"Window config: size={WINDOW_SIZE}, stride={STRIDE}")
    model, metrics = train_model(
        model_type=model_type,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
        # window_size, stride, train_split use shared config defaults
    )
    
    print(f"\nBest Test Metrics:")
    print(f"  Movement Accuracy: {metrics['movement_acc']*100:.2f}%")
    print(f"  Severity Accuracy: {metrics['severity_acc']*100:.2f}%")
