"""
Inference/Prediction module for trained EMG models.
Load trained models and make predictions on new EMG data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import pandas as pd

from NN.network import create_model
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS


def _infer_model_type_from_path(checkpoint_path):
    """Infer model type from checkpoint filename as a fallback."""
    name = os.path.basename(str(checkpoint_path)).lower()
    if "standard_cnn" in name:
        return "standard_cnn"
    if "lightweight" in name:
        return "lightweight"
    if "full" in name:
        return "full"
    return "full"


def load_trained_model(checkpoint_path, device=None):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        device: Device to load model on (auto-detect if None)
        
    Returns:
        model: Loaded PyTorch model in eval mode
        checkpoint: Full checkpoint dictionary with metadata
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved parameters; fallback to filename inference for
    # legacy checkpoints that may not contain model_type metadata.
    model_type = checkpoint.get('model_type') or _infer_model_type_from_path(checkpoint_path)
    model = create_model(
        model_type=model_type,
        num_channels=8,
        num_movements=checkpoint.get('num_movements', 7),
        num_severities=checkpoint.get('num_severities', 3)
    )
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}'. "
            f"Resolved model_type='{model_type}'. "
            "Checkpoint architecture may not match selected model file."
        ) from exc
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Resolved model type: {model_type}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    if 'test_metrics' in checkpoint:
        metrics = checkpoint['test_metrics']
        print(f"Test Movement Accuracy: {metrics['movement_acc']*100:.2f}%")
        print(f"Test Severity Accuracy: {metrics['severity_acc']*100:.2f}%")
    
    return model, checkpoint


def predict_from_tensor(model, emg_data, window_size=100, device=None):
    """
    Make predictions on EMG tensor data.
    
    Args:
        model: Trained model
        emg_data: Tensor of shape (num_samples, 8) or (window_size, 8)
        window_size: Expected input window size
        device: Device to run inference on
        
    Returns:
        dict with keys:
            - movement_pred: Predicted movement class index
            - movement_name: Name of predicted movement
            - movement_probs: Probability distribution over movements
            - severity_pred: Predicted severity class index
            - severity_name: Name of predicted severity
            - severity_probs: Probability distribution over severities
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Ensure correct shape
    if emg_data.dim() == 2:
        # If more samples than window_size, take last window_size samples
        if emg_data.shape[0] > window_size:
            emg_data = emg_data[-window_size:, :]
        # If fewer samples, pad with zeros
        elif emg_data.shape[0] < window_size:
            padding = torch.zeros(window_size - emg_data.shape[0], 8)
            emg_data = torch.cat([padding, emg_data], dim=0)
    
    # Add batch dimension: (1, window_size, 8)
    if emg_data.dim() == 2:
        emg_data = emg_data.unsqueeze(0)
    
    emg_data = emg_data.to(device)
    
    with torch.no_grad():
        movement_logits, severity_logits = model(emg_data)
        
        # Get probabilities
        movement_probs = torch.softmax(movement_logits, dim=1)
        severity_probs = torch.softmax(severity_logits, dim=1)
        
        # Get predictions
        movement_pred = torch.argmax(movement_probs, dim=1).item()
        severity_pred = torch.argmax(severity_probs, dim=1).item()
    
    results = {
        'movement_pred': movement_pred,
        'movement_name': MOVEMENT_LABELS[movement_pred],
        'movement_probs': movement_probs.cpu().numpy()[0],
        'movement_confidence': movement_probs[0, movement_pred].item(),
        'severity_pred': severity_pred,
        'severity_name': SEVERITY_LABELS[severity_pred],
        'severity_probs': severity_probs.cpu().numpy()[0],
        'severity_confidence': severity_probs[0, severity_pred].item()
    }
    
    return results


def predict_from_csv(model, csv_path, window_size=100, stride=50, device=None):
    """
    Make predictions on EMG data from CSV file using sliding windows.
    Aggregates predictions across all windows for more robust results.
    
    Args:
        model: Trained model
        csv_path: Path to CSV file with 8 EMG channels
        window_size: Window size for prediction
        stride: Stride for sliding windows (default: 50 for 50% overlap)
        device: Device to run inference on
        
    Returns:
        dict: Aggregated prediction results across all windows
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Convert to tensor
    emg_full = torch.tensor(df.values, dtype=torch.float32)
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create sliding windows
    num_samples = emg_full.shape[0]
    windows = []
    
    for start_idx in range(0, num_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = emg_full[start_idx:end_idx, :]
        windows.append(window)
    
    # If no complete windows, use last window_size samples
    if len(windows) == 0:
        if num_samples >= window_size:
            windows.append(emg_full[-window_size:, :])
        else:
            # Pad if too short
            padding = torch.zeros(window_size - num_samples, 8)
            windows.append(torch.cat([padding, emg_full], dim=0))
    
    # Stack windows into batch
    batch = torch.stack(windows).to(device)
    
    # Predict on all windows
    with torch.no_grad():
        movement_logits, severity_logits = model(batch)
        
        # Get probabilities for each window
        movement_probs = torch.softmax(movement_logits, dim=1)
        severity_probs = torch.softmax(severity_logits, dim=1)
        
        # Average probabilities across all windows
        avg_movement_probs = movement_probs.mean(dim=0)
        avg_severity_probs = severity_probs.mean(dim=0)
        
        # Get final predictions from averaged probabilities
        movement_pred = torch.argmax(avg_movement_probs).item()
        severity_pred = torch.argmax(avg_severity_probs).item()
    
    results = {
        'movement_pred': movement_pred,
        'movement_name': MOVEMENT_LABELS[movement_pred],
        'movement_probs': avg_movement_probs.cpu().numpy(),
        'movement_confidence': avg_movement_probs[movement_pred].item(),
        'severity_pred': severity_pred,
        'severity_name': SEVERITY_LABELS[severity_pred],
        'severity_probs': avg_severity_probs.cpu().numpy(),
        'severity_confidence': avg_severity_probs[severity_pred].item(),
        'num_windows': len(windows)
    }
    
    return results


def predict_streaming(model, emg_buffer, window_size=100, device=None):
    """
    Make predictions on streaming EMG data.
    Useful for real-time classification.
    
    Args:
        model: Trained model
        emg_buffer: Circular buffer or list of recent EMG samples (N, 8)
        window_size: Window size for prediction
        device: Device to run inference on
        
    Returns:
        dict: Prediction results
    """
    # Convert buffer to tensor
    if isinstance(emg_buffer, list):
        emg_buffer = np.array(emg_buffer)
    
    if isinstance(emg_buffer, np.ndarray):
        emg_tensor = torch.tensor(emg_buffer, dtype=torch.float32)
    else:
        emg_tensor = emg_buffer
    
    # Make prediction
    results = predict_from_tensor(model, emg_tensor, window_size, device)
    
    return results


def print_prediction(results, verbose=True):
    """
    Pretty print prediction results.
    
    Args:
        results: Dictionary from predict_* functions
        verbose: If True, print probability distributions
    """
    print("\n=== Prediction Results ===")
    print(f"Movement: {results['movement_name']} (confidence: {results['movement_confidence']*100:.1f}%)")
    print(f"Severity: {results['severity_name']} (confidence: {results['severity_confidence']*100:.1f}%)")
    
    if verbose:
        print("\nMovement probabilities:")
        for idx, prob in enumerate(results['movement_probs']):
            print(f"  {MOVEMENT_LABELS[idx]:20s}: {prob*100:5.1f}%")
        
        print("\nSeverity probabilities:")
        for idx, prob in enumerate(results['severity_probs']):
            print(f"  {SEVERITY_LABELS[idx]:20s}: {prob*100:5.1f}%")
    
    print("=" * 30)

def save_prediction(results, file_out):
    """
    Pretty print prediction results.
    
    Args:
        results: result from predict_* functions
    """
    
# Example usage
if __name__ == "__main__":
    # Example: Load model and make prediction
    model_path = "./NN/models/best_model_full.pth"
    
    if os.path.exists(model_path):
        # Load model
        model, checkpoint = load_trained_model(model_path)
        
        # Example 1: Predict from CSV
        csv_path = "./DATA/Example_data/S1_Hard_C7_R1.csv"
        if os.path.exists(csv_path):
            print(f"\nPredicting from CSV: {csv_path}")
            results = predict_from_csv(model, csv_path, window_size=100)
            print_prediction(results, verbose=True)
        
        # Example 2: Predict from tensor (simulated data)
        print("\nPredicting from random simulated data...")
        simulated_data = torch.randn(100, 8)  # Random EMG-like data
        results = predict_from_tensor(model, simulated_data, window_size=100)
        print_prediction(results, verbose=False)
        
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train.py")
