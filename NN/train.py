"""
Training script for EMG Multi-Task Classification Model.
Trains a model to predict both movement type and contraction severity.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    window_size=100,
    stride=50,
    train_split=0.8,
    save_dir='./models',
    device=None
):
    """
    Main training function.
    
    Args:
        model_type: 'full' or 'lightweight'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        window_size: Timesteps per sample
        stride: Sliding window stride
        train_split: Train/test split ratio
        save_dir: Directory to save models
        device: Device to train on (auto-detect if None)
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        num_workers=0
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
    criterion = MultiTaskLoss(movement_weight=1.0, severity_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard logging (optional)
    writer = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'./runs/emg_training_{timestamp}'
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging enabled: {log_dir}")
    
    # Training loop
    print("\n=== Starting Training ===")
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
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
        
        # Save best model
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_metrics': test_metrics,
                'model_type': model_type,
                'window_size': window_size,
                'num_movements': num_movements,
                'num_severities': num_severities
            }
            save_path = os.path.join(save_dir, f'best_model_{model_type}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_metrics': test_metrics,
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
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")
    
    return model, test_metrics


if __name__ == "__main__":
    # Train the full model
    print("Training Full Model...")
    model, metrics = train_model(
        model_type='full',
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001,
        window_size=100,
        stride=50
    )
    
    print(f"\nFinal Test Metrics:")
    print(f"  Movement Accuracy: {metrics['movement_acc']*100:.2f}%")
    print(f"  Severity Accuracy: {metrics['severity_acc']*100:.2f}%")
