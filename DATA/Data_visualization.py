import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Data_Conversion as DC
from NN.predict import load_trained_model, predict_from_tensor
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS

# Load the trained model once
MODEL_PATH = "NN/models/best_model_full.pth"
model, _ = load_trained_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

def plot_window_with_prediction(window, true_movement_idx, true_severity_idx, file_name, window_size=100):
    """
    Plot EMG window with NN prediction overlay.
    Colors change based on whether prediction is correct.
    
    Args:
        window: Tensor of shape (seq_len, channels)
        true_movement_idx: Ground truth movement index (0-6)
        true_severity_idx: Ground truth severity index (0-2)
        file_name: Name for saving the plot
        window_size: Window size for prediction (default 100)
    """
    # Get a window for prediction (take last window_size samples)
    seq_len, channels = window.shape
    
    if seq_len >= window_size:
        pred_window = window[-window_size:, :]
        plot_start = seq_len - window_size
    else:
        pred_window = window
        plot_start = 0
    
    # Get NN prediction
    prediction = predict_from_tensor(model, pred_window, window_size=window_size)
    
    # Check if prediction is correct
    movement_correct = prediction['movement_pred'] == true_movement_idx
    severity_correct = prediction['severity_pred'] == true_severity_idx
    both_correct = movement_correct and severity_correct
    
    # Set color based on correctness
    if both_correct:
        color = 'green'
        status = "✓ CORRECT"
    elif movement_correct:
        color = 'orange'
        status = "⚠ Movement OK, Severity Wrong"
    else:
        color = 'red'
        status = "✗ INCORRECT"
    
    # Create plot
    fig, axs = plt.subplots(channels, 1, figsize=(12, 2*channels), sharex=True)
    
    for c in range(channels):
        # Plot entire signal in light gray
        axs[c].plot(window[:, c], color='lightgray', linewidth=0.8, label='Signal')
        
        # Highlight prediction window with color based on correctness
        if seq_len >= window_size:
            axs[c].plot(range(plot_start, seq_len), 
                       window[plot_start:, c], 
                       color=color, linewidth=1.5, label='Prediction Window')
        
        axs[c].set_ylabel(f"EMG Ch{c}", fontweight='bold')
        axs[c].grid(True, alpha=0.3)
        if c == 0:
            axs[c].legend(loc='upper right', fontsize=8)
    
    # Add title with prediction info
    true_movement = MOVEMENT_LABELS[true_movement_idx]
    true_severity = SEVERITY_LABELS[true_severity_idx]
    pred_movement = prediction['movement_name']
    pred_severity = prediction['severity_name']
    
    fig.suptitle(
        f"{status}\n"
        f"True: {true_movement} ({true_severity}) | "
        f"Predicted: {pred_movement} ({pred_severity})\n"
        f"Movement Conf: {prediction['movement_confidence']:.2%} | "
        f"Severity Conf: {prediction['severity_confidence']:.2%}",
        fontsize=12, fontweight='bold', color=color
    )
    
    plt.xlabel("Sample Index", fontweight='bold')
    plt.tight_layout()
    
    # Save to results/
    plt.savefig(f"DATA/Results/{file_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: DATA/Results/{file_name}.png - {status}")
    plt.close()
    
# Ensure folder exists
os.makedirs("DATA/Results", exist_ok=True)

# Load datasets
print("\nLoading datasets...")
tensors = DC.load_all_datasets()

# Generate visualizations for each movement/severity combination
print("\nGenerating visualizations with NN predictions...\n")

# Example: Take a 200-sample window from each movement
window_samples = 200

# Light intensity movements
for movement_idx in range(7):
    tensor_data = tensors["Light"][movement_idx]
    # Take a window from the middle of the signal
    start_idx = len(tensor_data) // 2
    window = tensor_data[start_idx:start_idx + window_samples]
    
    plot_window_with_prediction(
        window, 
        true_movement_idx=movement_idx,
        true_severity_idx=0,  # Light = 0
        file_name=f"Light_{MOVEMENT_LABELS[movement_idx]}"
    )

# Medium intensity movements
for movement_idx in range(7):
    tensor_data = tensors["Medium"][movement_idx]
    start_idx = len(tensor_data) // 2
    window = tensor_data[start_idx:start_idx + window_samples]
    
    plot_window_with_prediction(
        window,
        true_movement_idx=movement_idx,
        true_severity_idx=1,  # Medium = 1
        file_name=f"Medium_{MOVEMENT_LABELS[movement_idx]}"
    )

# Hard intensity movements
for movement_idx in range(7):
    tensor_data = tensors["Hard"][movement_idx]
    start_idx = len(tensor_data) // 2
    window = tensor_data[start_idx:start_idx + window_samples]
    
    plot_window_with_prediction(
        window,
        true_movement_idx=movement_idx,
        true_severity_idx=2,  # Hard = 2
        file_name=f"Hard_{MOVEMENT_LABELS[movement_idx]}"
    )

print("\nAll visualizations complete!")