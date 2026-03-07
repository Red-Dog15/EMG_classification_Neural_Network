import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DATA import Data_Conversion as DC
from NN.predict import load_trained_model, predict_from_tensor
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS, create_labeled_dataset
from DATA.dataset import EMGDataset
from torch.utils.data import random_split

# Import shared configuration to match training settings
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import WINDOW_SIZE, STRIDE, TRAIN_SPLIT, SPLIT_SEED, print_config_summary
    # Use shared config for consistency
    _USING_SHARED_CONFIG = True
except ImportError:
    # Fallback if config.py not found
    WINDOW_SIZE = 100
    STRIDE = 50
    TRAIN_SPLIT = 0.8
    SPLIT_SEED = 42
    _USING_SHARED_CONFIG = False
    print("⚠ Warning: config.py not found. Using local defaults.")

# EMG Data Specifications from LibEMG ContractionIntensity Dataset
# Source: https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt
SAMPLING_RATE_HZ = 1000  # 1 kHz sampling frequency
TIME_PER_SAMPLE_MS = 1000 / SAMPLING_RATE_HZ  # 1.0 ms per sample
TIME_PER_SAMPLE_S = 1 / SAMPLING_RATE_HZ  # 0.001 s per sample

# Path configuration (cwd-independent)
SCRIPTS_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(SCRIPTS_ROOT, "NN", "models")
RESULTS_DIR = os.path.join(SCRIPTS_ROOT, "DATA", "Results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "Raw_Data")
PRED_RESULTS_DIR = os.path.join(RESULTS_DIR, "Predicted_Data")
HEATMAP_RESULTS_DIR = os.path.join(RESULTS_DIR, "Heatmaps")

# ============================================================================
# HEATMAP EVALUATION CONTROLS
# 
# ⚠️ CRITICAL: Window settings (size/stride) are now imported from config.py
# This ensures training and evaluation use identical window configurations!
#
# TO INCREASE/DECREASE THE NUMBER OF WINDOWS FOR TRAINING & EVALUATION:
# Edit Scripts/config.py and change these variables:
#
#   1. STRIDE (most common way to increase windows):
#      - STRIDE = 50  → ~1,239 windows (current)
#      - STRIDE = 25  → ~2,457 windows (2x more)
#      - STRIDE = 10  → ~6,111 windows (5x more)
#      - STRIDE = 1   → ~60,921 windows (50x more, very slow!)
#
#   2. WINDOW_SIZE (affects model input):
#      - WINDOW_SIZE = 100 → ~1,239 windows (current)
#      - WINDOW_SIZE = 50  → ~2,499 windows (2x more)
#      - Note: Changing this requires retraining models!
#
#   3. TRAIN_SPLIT (changes train/test ratio):
#      - TRAIN_SPLIT = 0.8 → 80% train, 20% test (current)
#      - TRAIN_SPLIT = 0.7 → 70% train, 30% test (more test samples)
#
# Run 'python Scripts/config.py' to see the impact of different settigs!n
#
# Edit these variables to control how many test windows are used when comparing
# multiple NN architectures in the heatmap.
#
# HEATMAP_TRAIN_SPLIT & HEATMAP_SPLIT_SEED:
# - Controls the train/test split (0.8 = 80% train, 20% test)
# - Seed ensures the same split is created on every run (reproducibility)
#
# HEATMAP_WINDOW_SIZE & HEATMAP_STRIDE:
# - Window size: How many consecutive EMG samples form one input (100 samples = 100ms)
# - Stride: Step size when sliding the window (50 = 50% overlap)
# - MUST MATCH training settings for fair evaluation
#
# HEATMAP_MAX_TEST_SAMPLES (THE KEY CONTROL FOR EVALUATION SAMPLE COUNT):
# - None       -> Use ALL test windows (no subsampling)
# - int N      -> Use at most N test windows (if available)
# 
# UNDERSTANDING DATA FLOW:
# 1. You have 21 CSV files (7 movements × 3 severities)
# 2. Each CSV has ~3000 timesteps × 8 channels = 3000 raw samples
# 3. Total raw timesteps: 21 × 3000 = 63,000 timesteps
# 4. Sliding windows convert these to overlapping windows:
#    - Window size = 100 timesteps (100ms @ 1kHz sampling)
#    - Stride = 50 (50% overlap between windows)
#    - Each CSV → ~60 windows: (3000-100)/50 + 1 ≈ 60
#    - Total windows: 21 × 60 ≈ 1,260 windows
# 5. Train/test split (80/20):
#    - Training: ~1,008 windows (80%)
#    - Testing: ~252 windows (20%)
# 6. HEATMAP_MAX_TEST_SAMPLES controls which test windows are evaluated:
#    - If >= 252: Uses all ~252 test windows (no effect)
#    - If < 252: Uses random subset (THIS is how you control evaluation size)
#
# EXAMPLE WITH YOUR DATA (~252 test windows available):
#   - HEATMAP_MAX_TEST_SAMPLES = None    --> Uses all ~252 windows
#   - HEATMAP_MAX_TEST_SAMPLES = 1000    --> Uses all ~252 windows (limit has no effect)
#   - HEATMAP_MAX_TEST_SAMPLES = 200     --> Uses random 200 windows (79% of test set)
#   - HEATMAP_MAX_TEST_SAMPLES = 100     --> Uses random 100 windows (40% of test set)
#   - HEATMAP_MAX_TEST_SAMPLES = 50      --> Uses random 50 windows (20% of test set)
#
# HEATMAP_SUBSET_STRATEGY:
# - "random"   -> Deterministic random subset (same samples each run, but shuffled)
# - "first"    -> Take the first N windows in order
# ============================================================================
# IMPORTANT: These now default to shared config values from config.py
# This ensures evaluation uses the SAME window settings as training!
HEATMAP_TRAIN_SPLIT = TRAIN_SPLIT     # Train/test split ratio (from config.py)
HEATMAP_SPLIT_SEED = SPLIT_SEED       # Random seed for reproducible splits (from config.py)
HEATMAP_WINDOW_SIZE = WINDOW_SIZE     # EMG samples per window (from config.py)
HEATMAP_STRIDE = STRIDE               # Stride for sliding windows (from config.py)
HEATMAP_MAX_TEST_SAMPLES = 1000       # Max test windows to evaluate (see explanation above)
HEATMAP_SUBSET_STRATEGY = "random"    # "random" or "first"

# ============================================================================
# MODEL REGISTRY - Configure available NN architectures
# Each NN architecture (A/B/C) represents a different neural network design
# They can use either "best" or "final" trained model checkpoints
# ============================================================================
MODEL_REGISTRY = {
    "NN-A": {
        "name": "NN-A (Current Architecture)",
        "path_best": os.path.join(MODELS_DIR, "best_model_full.pth"),
        "path_final": os.path.join(MODELS_DIR, "final_model_full.pth"),
        "description": "Current neural network architecture with best/final checkpoints",
        "available": True
    },
    "NN-B": {
        "name": "NN-B (Standard CNN)",
        "path_best": os.path.join(MODELS_DIR, "best_model_standard_cnn.pth"),
        "path_final": os.path.join(MODELS_DIR, "final_model_standard_cnn.pth"),
        "description": "CNN-only baseline inspired by standard CNN EMG layouts",
        "available": True
    },
    "NN-C": {
        "name": "NN-C (Lightweight)",
        "path_best": os.path.join(MODELS_DIR, "best_model_lightweight.pth"),
        "path_final": os.path.join(MODELS_DIR, "final_model_lightweight.pth"),
        "description": "Lightweight CNN architecture for embedded/low-resource deployment",
        "available": True  # Trained and available
    }
}

# Global state for analytics tracking
analytics_data = {}
current_model_info = {}

def samples_to_time_ms(samples):
    """Convert sample indices to time in milliseconds."""
    return samples * TIME_PER_SAMPLE_MS

def samples_to_time_s(samples):
    """Convert sample indices to time in seconds."""
    return samples * TIME_PER_SAMPLE_S


def select_eval_indices(total_size, max_samples=None, strategy="random", seed=42):
    """
    Select test-set indices to evaluate for fair model comparison.

    Args:
        total_size: Total number of available test windows.
        max_samples: Optional cap on number of test windows to use.
        strategy: "random" for seeded random subset, "first" for first-N.
        seed: Seed used for deterministic random sampling.
    
    Returns:
        list: Indices of test samples to evaluate.
    
    How it works:
        - If max_samples is None or >= total_size: returns ALL indices (no subsampling)
        - If max_samples < total_size: returns a subset of max_samples indices
        - Seeded selection ensures the same subset is chosen on repeated runs
    """
    if total_size <= 0:
        return []

    # KEY LOGIC: If max_samples is None or larger than available samples,
    # we return ALL samples. This means setting max_samples=1000 when you only
    # have 800 test samples will use all 800 (no subsampling occurs).
    if max_samples is None or max_samples >= total_size:
        return list(range(total_size))  # No subsampling - uses everything

    if max_samples <= 0:
        return []

    if strategy == "first":
        # Take the first N samples in order
        return list(range(max_samples))

    # RANDOM SUBSET: Uses deterministic seeding so the same random subset
    # is selected on every run (ensures fair comparison across multiple runs).
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.randperm(total_size, generator=generator)[:max_samples]
    return sampled.tolist()

def calculate_dataset_breakdown(window_size=100, stride=50, train_split=0.8):
    """
    Calculate and display how raw CSV data becomes windowed train/test samples.
    
    DATA FLOW:
    1. Raw CSVs: 21 files (7 movements × 3 severities)
    2. Each CSV: ~3000 timesteps × 8 channels
    3. Sliding windows: Convert timesteps into overlapping windows
    4. Train/test split: Divide windows between training and testing
    5. Evaluation: Use subset of test windows for model comparison
    
    Returns:
        dict with breakdown statistics
    """
    # Constants from your dataset
    num_movements = 7
    num_severities = 3
    num_csv_files = num_movements * num_severities  # 21 files
    avg_timesteps_per_csv = 3000  # Each CSV has ~3000 rows
    num_channels = 8
    
    # Calculate windows per CSV file
    # Formula: (total_timesteps - window_size) / stride + 1
    windows_per_csv = (avg_timesteps_per_csv - window_size) // stride + 1
    
    # Total windows from all CSVs
    total_windows = num_csv_files * windows_per_csv
    
    # Train/test split
    train_windows = int(total_windows * train_split)
    test_windows = total_windows - train_windows
    
    breakdown = {
        "num_csv_files": num_csv_files,
        "timesteps_per_csv": avg_timesteps_per_csv,
        "total_raw_timesteps": num_csv_files * avg_timesteps_per_csv,
        "window_size": window_size,
        "stride": stride,
        "windows_per_csv": windows_per_csv,
        "total_windows": total_windows,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "train_split": train_split
    }
    
    return breakdown


def print_dataset_breakdown():
    """Print a human-readable explanation of the data flow."""
    breakdown = calculate_dataset_breakdown(
        window_size=HEATMAP_WINDOW_SIZE,
        stride=HEATMAP_STRIDE,
        train_split=HEATMAP_TRAIN_SPLIT
    )
    
    print("\n" + "="*80)
    print("📊 DATA FLOW: FROM RAW CSV TO TEST WINDOWS")
    print("="*80)
    print(f"\n1️⃣  RAW CSV FILES:")
    print(f"   • {breakdown['num_csv_files']} CSV files (7 movements × 3 severities)")
    print(f"   • ~{breakdown['timesteps_per_csv']:,} timesteps per CSV")
    print(f"   • Total raw timesteps: {breakdown['total_raw_timesteps']:,}")
    print(f"   • Each timestep has 8 EMG channels")
    
    print(f"\n2️⃣  SLIDING WINDOW CONVERSION:")
    print(f"   • Window size: {breakdown['window_size']} timesteps (100ms @ 1kHz)")
    print(f"   • Stride: {breakdown['stride']} timesteps (50% overlap)")
    print(f"   • Windows per CSV: ~{breakdown['windows_per_csv']}")
    print(f"   • Formula: (3000 - {breakdown['window_size']}) ÷ {breakdown['stride']} + 1 = {breakdown['windows_per_csv']}")
    
    print(f"\n3️⃣  TOTAL WINDOWS CREATED:")
    print(f"   • {breakdown['num_csv_files']} CSVs × {breakdown['windows_per_csv']} windows/CSV = {breakdown['total_windows']:,} total windows")
    print(f"   • Each window: ({breakdown['window_size']} timesteps, 8 channels)")
    
    print(f"\n4️⃣  TRAIN/TEST SPLIT ({int(breakdown['train_split']*100)}/{int((1-breakdown['train_split'])*100)}):")
    print(f"   • Training windows: {breakdown['train_windows']:,} ({breakdown['train_split']*100:.0f}%)")
    print(f"   • Test windows: {breakdown['test_windows']:,} ({(1-breakdown['train_split'])*100:.0f}%)")
    
    print(f"\n5️⃣  HEATMAP EVALUATION CONTROL:")
    limit = HEATMAP_MAX_TEST_SAMPLES
    if limit is None:
        print(f"   • HEATMAP_MAX_TEST_SAMPLES = None")
        print(f"   • Will evaluate ALL {breakdown['test_windows']:,} test windows")
    elif limit >= breakdown['test_windows']:
        print(f"   • HEATMAP_MAX_TEST_SAMPLES = {limit:,}")
        print(f"   • ⚠ Limit is >= {breakdown['test_windows']:,} test windows")
        print(f"   • Will evaluate ALL {breakdown['test_windows']:,} test windows (limit has no effect)")
    else:
        pct = (limit / breakdown['test_windows']) * 100
        print(f"   • HEATMAP_MAX_TEST_SAMPLES = {limit:,}")
        print(f"   • Will evaluate {limit:,} / {breakdown['test_windows']:,} test windows ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHT:")
    print(f"   Your {breakdown['total_raw_timesteps']:,} raw timesteps become {breakdown['test_windows']:,} test windows.")
    print(f"   This is because overlapping windows are created from continuous data.")
    print(f"   To control evaluation: Set HEATMAP_MAX_TEST_SAMPLES < {breakdown['test_windows']:,}")
    print("="*80 + "\n")
    
    return breakdown


def prompt_heatmap_sample_limit(default_limit):
    """
    Allow the user to override heatmap sample count at runtime.
    
    How this works:
    - The limit only takes effect if it's SMALLER than the actual test set size
    - If you have 800 test samples and set limit=1000, all 800 are used
    - If you have 800 test samples and set limit=400, only 400 are used (random subset)
    """
    # Show the breakdown first so user understands what they're controlling
    breakdown = print_dataset_breakdown()
    
    print("-"*80)
    print("HEATMAP SAMPLE SIZE CONTROL")
    print("-"*80)
    print(f"Current default: {default_limit if default_limit is not None else 'all test samples'}")
    print(f"Available test windows: ~{breakdown['test_windows']}")
    print("\nEnter:")
    print("  • Press Enter to keep default")
    print("  • all  -> use full test set")
    print(f"  • N    -> use exactly N test windows (max: ~{breakdown['test_windows']})")
    print(f"\n💡 TIP: Set N < {breakdown['test_windows']} to see actual subsampling")

    raw = input("\nHeatmap test sample count override: ").strip().lower()
    if raw == "":
        return default_limit
    if raw == "all":
        return None
    if raw.isdigit() and int(raw) > 0:
        return int(raw)

    print("Invalid input. Keeping default sample count.")
    return default_limit

# ============================================================================
# MODEL AND ANALYTICS INITIALIZATION
# ============================================================================

def initialize_analytics_data(model_path):
    """Initialize analytics data structure for a given model."""
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "window_size": 100
        },
        "predictions": [],
        "movement_stats": {i: {"correct": 0, "total": 0} for i in range(7)},
        "severity_stats": {i: {"correct": 0, "total": 0} for i in range(3)},
        "combined_stats": {"correct": 0, "total": 0}
    }

def select_model():
    """Display available NN architectures and let user select one with checkpoint type."""
    print("\n" + "="*80)
    print("AVAILABLE NEURAL NETWORK ARCHITECTURES")
    print("="*80)
    
    models = list(MODEL_REGISTRY.keys())
    available_models = [k for k in models if MODEL_REGISTRY[k]["available"]]
    
    for i, key in enumerate(models, 1):
        model = MODEL_REGISTRY[key]
        status = "✓ Available" if model["available"] else "⚠ Not Implemented"
        print(f"{i}. {model['name']:35} [{status}]")
        print(f"   {model['description']}")
    
    print("\n" + "-"*80)
    
    # Select architecture
    if len(available_models) == 0:
        print("⚠ No models available!")
        return None, None
    
    choice = input(f"Select NN architecture (1-{len(models)}) or press Enter for NN-A: ").strip()
    
    if choice == "" or choice == "1":
        selected_key = models[0]
    elif choice.isdigit() and 1 <= int(choice) <= len(models):
        selected_key = models[int(choice) - 1]
    else:
        print(f"Invalid choice. Using NN-A.")
        selected_key = models[0]
    
    model_info = MODEL_REGISTRY[selected_key]
    
    if not model_info["available"]:
        print(f"\n⚠ {model_info['name']} is not yet implemented!")
        print("   Please select an available model or implement this architecture first.")
        return None, None
    
    # Select checkpoint type
    print("\n" + "-"*80)
    print("Select trained checkpoint:")
    print("1. Best model (best validation performance)")
    print("2. Final model (final training epoch)")
    
    checkpoint_choice = input("\nSelect checkpoint (1-2) or press Enter for best: ").strip()
    
    if checkpoint_choice == "2":
        model_path = model_info["path_final"]
        checkpoint_type = "final"
    else:
        model_path = model_info["path_best"]
        checkpoint_type = "best"
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Warning: Model file not found at {model_path}")
        print("   The script will fail if this model is needed for prediction.")
        return None, None
    
    print(f"\n✓ Selected: {model_info['name']} ({checkpoint_type} checkpoint)")
    print(f"  Path: {model_path}")
    
    # Create model info dict with selected path
    selected_model_info = {
        "name": model_info["name"],
        "path": model_path,
        "architecture": selected_key,
        "checkpoint": checkpoint_type,
        "description": model_info["description"]
    }
    
    return selected_key, selected_model_info

def plot_raw_window(window, true_movement_idx, true_severity_idx, file_name):
    """
    Plot raw EMG window without predictions.
    
    Args:
        window: Tensor of shape (seq_len, channels)
        true_movement_idx: Ground truth movement index (0-6)
        true_severity_idx: Ground truth severity index (0-2)
        file_name: Name for saving the plot
    """
    seq_len, channels = window.shape
    
    # Create time axis in milliseconds
    time_ms = samples_to_time_ms(torch.arange(seq_len))
    
    # Create plot
    fig, axs = plt.subplots(channels, 1, figsize=(12, 2*channels), sharex=True)
    
    for c in range(channels):
        axs[c].plot(time_ms, window[:, c], color='steelblue', linewidth=1.0)
        axs[c].set_ylabel(f"EMG Ch{c}\n(mV)", fontweight='bold')
        axs[c].grid(True, alpha=0.3)
    
    # Add title
    true_movement = MOVEMENT_LABELS[true_movement_idx]
    true_severity = SEVERITY_LABELS[true_severity_idx]
    
    fig.suptitle(
        f"Raw EMG Signal: {true_movement} ({true_severity})\n"
        f"Duration: {time_ms[-1]:.1f} ms ({samples_to_time_s(seq_len):.3f} s) | "
        f"Samples: {seq_len} @ {SAMPLING_RATE_HZ} Hz",
        fontsize=12, fontweight='bold'
    )
    
    plt.xlabel("Time (milliseconds)", fontweight='bold')
    plt.tight_layout()
    
    # Save to raw data folder
    output_path = os.path.join(RAW_RESULTS_DIR, f"{file_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Raw: {output_path}")
    plt.close()

def plot_window_with_prediction(window, true_movement_idx, true_severity_idx, file_name, window_size=100):
    """
    Plot EMG window with NN prediction overlay.
    Colors change based on whether prediction is correct.
    Tracks analytics data.
    
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
    
    # Update analytics
    analytics_data["predictions"].append({
        "file": file_name,
        "true_movement": MOVEMENT_LABELS[true_movement_idx],
        "true_movement_idx": true_movement_idx,
        "pred_movement": prediction['movement_name'],
        "pred_movement_idx": prediction['movement_pred'],
        "movement_correct": movement_correct,
        "movement_confidence": prediction['movement_confidence'],
        "true_severity": SEVERITY_LABELS[true_severity_idx],
        "true_severity_idx": true_severity_idx,
        "pred_severity": prediction['severity_name'],
        "pred_severity_idx": prediction['severity_pred'],
        "severity_correct": severity_correct,
        "severity_confidence": prediction['severity_confidence'],
        "both_correct": both_correct
    })
    
    # Update statistics
    analytics_data["movement_stats"][true_movement_idx]["total"] += 1
    if movement_correct:
        analytics_data["movement_stats"][true_movement_idx]["correct"] += 1
    
    analytics_data["severity_stats"][true_severity_idx]["total"] += 1
    if severity_correct:
        analytics_data["severity_stats"][true_severity_idx]["correct"] += 1
    
    analytics_data["combined_stats"]["total"] += 1
    if both_correct:
        analytics_data["combined_stats"]["correct"] += 1
    
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
    
    # Create time axis in milliseconds
    time_ms = samples_to_time_ms(torch.arange(seq_len))
    time_ms_pred = samples_to_time_ms(torch.arange(plot_start, seq_len))
    
    # Create plot
    fig, axs = plt.subplots(channels, 1, figsize=(12, 2*channels), sharex=True)
    
    for c in range(channels):
        # Plot entire signal in light gray
        axs[c].plot(time_ms, window[:, c], color='lightgray', linewidth=0.8, label='Signal')
        
        # Highlight prediction window with color based on correctness
        if seq_len >= window_size:
            axs[c].plot(time_ms_pred, 
                       window[plot_start:, c], 
                       color=color, linewidth=1.5, label='Prediction Window')
        
        axs[c].set_ylabel(f"EMG Ch{c}\n(mV)", fontweight='bold')
        axs[c].grid(True, alpha=0.3)
        if c == 0:
            axs[c].legend(loc='upper right', fontsize=8)
    
    # Add title with prediction info
    true_movement = MOVEMENT_LABELS[true_movement_idx]
    true_severity = SEVERITY_LABELS[true_severity_idx]
    pred_movement = prediction['movement_name']
    pred_severity = prediction['severity_name']
    
    # Calculate prediction window duration
    pred_duration_ms = samples_to_time_ms(window_size if seq_len >= window_size else seq_len)
    
    fig.suptitle(
        f"{status}\n"
        f"True: {true_movement} ({true_severity}) | "
        f"Predicted: {pred_movement} ({pred_severity})\n"
        f"Movement Conf: {prediction['movement_confidence']:.2%} | "
        f"Severity Conf: {prediction['severity_confidence']:.2%} | "
        f"Window: {pred_duration_ms:.0f} ms @ {SAMPLING_RATE_HZ} Hz",
        fontsize=12, fontweight='bold', color=color
    )
    
    plt.xlabel("Time (milliseconds)", fontweight='bold')
    plt.tight_layout()
    
    # Save to predicted data folder
    output_path = os.path.join(PRED_RESULTS_DIR, f"{file_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved Predicted: {output_path} - {status}")
    plt.close()

def generate_raw_plots(tensors, window_samples=200):
    """Generate raw EMG signal plots without predictions."""
    print("\n" + "="*80)
    print("GENERATING RAW EMG PLOTS")
    print("="*80)
    
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    
    for severity_name, severity_idx in [("Light", 0), ("Medium", 1), ("Hard", 2)]:
        for movement_idx in range(7):
            tensor_data = tensors[severity_name][movement_idx]
            start_idx = len(tensor_data) // 2
            window = tensor_data[start_idx:start_idx + window_samples]
            
            plot_raw_window(
                window,
                true_movement_idx=movement_idx,
                true_severity_idx=severity_idx,
                file_name=f"{severity_name}_{MOVEMENT_LABELS[movement_idx]}"
            )
    
    print(f"\n✅ Raw plots saved to: {RAW_RESULTS_DIR}")

def generate_predicted_plots(tensors, window_samples=200):
    """Generate EMG plots with NN predictions overlaid."""
    print("\n" + "="*80)
    print("GENERATING PREDICTED EMG PLOTS")
    print("="*80)
    
    os.makedirs(PRED_RESULTS_DIR, exist_ok=True)
    
    for severity_name, severity_idx in [("Light", 0), ("Medium", 1), ("Hard", 2)]:
        for movement_idx in range(7):
            tensor_data = tensors[severity_name][movement_idx]
            start_idx = len(tensor_data) // 2
            window = tensor_data[start_idx:start_idx + window_samples]
            
            plot_window_with_prediction(
                window,
                true_movement_idx=movement_idx,
                true_severity_idx=severity_idx,
                file_name=f"{severity_name}_{MOVEMENT_LABELS[movement_idx]}"
            )
    
    print(f"\n✅ Predicted plots saved to: {PRED_RESULTS_DIR}")

def generate_analytics_report():
    """Generate analytics report from collected prediction data."""
    print("\n" + "="*80)
    print("GENERATING ANALYTICS REPORT")
    print("="*80)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Calculate overall accuracies
    movement_accuracy = analytics_data["movement_stats"]
    severity_accuracy = analytics_data["severity_stats"]
    combined_accuracy = analytics_data["combined_stats"]
    
    # Movement accuracy by class
    movement_class_accuracy = {}
    for idx, stats in movement_accuracy.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            movement_class_accuracy[idx] = {
                "name": MOVEMENT_LABELS[idx],
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"]
            }
    
    # Severity accuracy by class
    severity_class_accuracy = {}
    for idx, stats in severity_accuracy.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            severity_class_accuracy[idx] = {
                "name": SEVERITY_LABELS[idx],
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"]
            }
    
    # Overall metrics
    total_movement_correct = sum(s["correct"] for s in movement_accuracy.values())
    total_movement_samples = sum(s["total"] for s in movement_accuracy.values())
    avg_movement_accuracy = total_movement_correct / total_movement_samples if total_movement_samples > 0 else 0
    
    total_severity_correct = sum(s["correct"] for s in severity_accuracy.values())
    total_severity_samples = sum(s["total"] for s in severity_accuracy.values())
    avg_severity_accuracy = total_severity_correct / total_severity_samples if total_severity_samples > 0 else 0
    
    avg_overall_accuracy = combined_accuracy["correct"] / combined_accuracy["total"] if combined_accuracy["total"] > 0 else 0
    
    # Identify critical classes (accuracy < 70%)
    critical_movements = [
        {"class": data["name"], "accuracy": data["accuracy"], "samples": data["total"]}
        for idx, data in movement_class_accuracy.items()
        if data["accuracy"] < 0.70
    ]
    
    critical_severities = [
        {"class": data["name"], "accuracy": data["accuracy"], "samples": data["total"]}
        for idx, data in severity_class_accuracy.items()
        if data["accuracy"] < 0.70
    ]
    
    # Create analytics report
    analytics_report = {
        "timestamp": analytics_data["metadata"]["timestamp"],
        "model": analytics_data["metadata"]["model_path"],
        "total_samples": combined_accuracy["total"],
        
        "overall_metrics": {
            "movement_accuracy": avg_movement_accuracy,
            "severity_accuracy": avg_severity_accuracy,
            "combined_accuracy": avg_overall_accuracy
        },
        
        "movement_class_performance": movement_class_accuracy,
        "severity_class_performance": severity_class_accuracy,
        
        "critical_classes": {
            "movements": critical_movements,
            "severities": critical_severities
        },
        
        "detailed_predictions": analytics_data["predictions"]
    }
    
    # Save to JSON
    analytics_file = os.path.join(RESULTS_DIR, "analytics_report.json")
    with open(analytics_file, "w") as f:
        json.dump(analytics_report, f, indent=2)
    
    # Save human-readable summary
    summary_file = os.path.join(RESULTS_DIR, "analytics_summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EMG CLASSIFICATION ANALYTICS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {analytics_data['metadata']['timestamp']}\n")
        f.write(f"Model: {analytics_data['metadata']['model_path']}\n")
        f.write(f"Total Samples: {combined_accuracy['total']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL ACCURACY METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Average Movement Accuracy:  {avg_movement_accuracy:.2%}\n")
        f.write(f"Average Severity Accuracy:  {avg_severity_accuracy:.2%}\n")
        f.write(f"Combined Overall Accuracy:  {avg_overall_accuracy:.2%}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MOVEMENT CLASS PERFORMANCE\n")
        f.write("="*80 + "\n")
        for idx in sorted(movement_class_accuracy.keys()):
            data = movement_class_accuracy[idx]
            status = "⚠ CRITICAL" if data["accuracy"] < 0.70 else "✓ Good"
            f.write(f"{data['name']:25} {data['accuracy']:6.2%}  ({data['correct']}/{data['total']})  {status}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SEVERITY CLASS PERFORMANCE\n")
        f.write("="*80 + "\n")
        for idx in sorted(severity_class_accuracy.keys()):
            data = severity_class_accuracy[idx]
            status = "⚠ CRITICAL" if data["accuracy"] < 0.70 else "✓ Good"
            f.write(f"{data['name']:25} {data['accuracy']:6.2%}  ({data['correct']}/{data['total']})  {status}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CRITICAL CLASSES (Accuracy < 70%)\n")
        f.write("="*80 + "\n")
        
        if critical_movements:
            f.write("\nMovements Needing Improvement:\n")
            for item in critical_movements:
                f.write(f"  • {item['class']:25} {item['accuracy']:6.2%}  ({item['samples']} samples)\n")
        else:
            f.write("\n✓ All movement classes performing well (>70% accuracy)\n")
        
        if critical_severities:
            f.write("\nSeverities Needing Improvement:\n")
            for item in critical_severities:
                f.write(f"  • {item['class']:25} {item['accuracy']:6.2%}  ({item['samples']} samples)\n")
        else:
            f.write("\n✓ All severity classes performing well (>70% accuracy)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✅ Analytics saved to:")
    print(f"   • {analytics_file}")
    print(f"   • {summary_file}")
    print("\n" + "="*80)
    print("ANALYTICS SUMMARY")
    print("="*80)
    print(f"Average Movement Accuracy:  {avg_movement_accuracy:.2%}")
    print(f"Average Severity Accuracy:  {avg_severity_accuracy:.2%}")
    print(f"Combined Overall Accuracy:  {avg_overall_accuracy:.2%}")
    
    if critical_movements or critical_severities:
        print("\n⚠ CRITICAL CLASSES DETECTED (Accuracy < 70%):")
        for item in critical_movements:
            print(f"  • Movement: {item['class']} - {item['accuracy']:.2%}")
        for item in critical_severities:
            print(f"  • Severity: {item['class']} - {item['accuracy']:.2%}")
    else:
        print("\n✓ All classes performing well!")
    
    print("="*80)

def generate_model_heatmap(
    train_split=HEATMAP_TRAIN_SPLIT,
    split_seed=HEATMAP_SPLIT_SEED,
    eval_window_size=HEATMAP_WINDOW_SIZE,
    eval_stride=HEATMAP_STRIDE,
    max_test_samples=HEATMAP_MAX_TEST_SAMPLES,
    subset_strategy=HEATMAP_SUBSET_STRATEGY
):
    """
    Generate a heatmap showing accuracy of different models across movement classifications.
    This allows comparison of multiple NN architectures.
    """
    print("\n" + "="*80)
    print("GENERATING MODEL COMPARISON HEATMAP")
    print("="*80)
    
    os.makedirs(HEATMAP_RESULTS_DIR, exist_ok=True)
    
    # Collect accuracy data for each model
    model_accuracy_data = {}

    # ================================================================
    # Build one deterministic held-out test set and reuse it for all models.
    # This guarantees a fair comparison because each model sees the exact same
    # test windows in the exact same order.
    # ================================================================
    print("\nPreparing held-out test set...")
    labeled_data = create_labeled_dataset()
    full_dataset = EMGDataset(labeled_data, window_size=eval_window_size, stride=eval_stride)

    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size

    _, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(split_seed)
    )

    selected_indices = select_eval_indices(
        total_size=len(test_dataset),
        max_samples=max_test_samples,
        strategy=subset_strategy,
        seed=split_seed + 1
    )

    if not selected_indices:
        print("\n⚠ No test samples selected for heatmap evaluation!")
        return

    actual_eval_count = len(selected_indices)
    test_dataset_size = len(test_dataset)
    
    print(f"\n  📊 TEST SET ANALYSIS:")
    print(f"  → Full test set size: {test_dataset_size} windows")
    print(f"  → Max samples requested: {max_test_samples if max_test_samples is not None else 'None (all)'}")
    print(f"  → Actually evaluating: {actual_eval_count} windows")
    print(f"  → Subset strategy: {subset_strategy}")
    
    # Show whether subsampling occurred
    if max_test_samples is None:
        print(f"  → Sampling mode: FULL TEST SET (no limit set)")
    elif max_test_samples >= test_dataset_size:
        print(f"  ⚠ Sampling mode: FULL TEST SET (requested {max_test_samples} but only {test_dataset_size} available)")
        print(f"     TIP: To see subsampling in action, set max_test_samples < {test_dataset_size}")
    else:
        pct = (actual_eval_count / test_dataset_size) * 100
        print(f"  ✓ Sampling mode: RANDOM SUBSET ({pct:.1f}% of test set)")
    
    print(f"\nTesting {len(MODEL_REGISTRY)} NN architectures...")
    
    for model_key, model_config in MODEL_REGISTRY.items():
        if not model_config["available"]:
            print(f"\n  ⊘ Skipping {model_config['name']}: Not yet implemented")
            continue
        
        model_path = model_config["path_best"]  # Use best checkpoint for comparison
        model_name = model_config["name"]
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: Model file not found at {model_path}")
            continue
        
        print(f"\n  • Processing: {model_name}")
        
        try:
            # Load model
            model, _ = load_trained_model(model_path)
            
            # Initialize analytics for this model
            model_analytics = initialize_analytics_data(model_path)
            
            # Evaluate model on the same selected held-out windows.
            for idx in selected_indices:
                window, movement_label, severity_label = test_dataset[idx]
                
                # Get prediction
                prediction = predict_from_tensor(model, window, window_size=eval_window_size)
                
                # Update stats only for movement accuracy (for heatmap)
                movement_correct = prediction['movement_pred'] == movement_label.item()
                
                model_analytics["movement_stats"][movement_label.item()]["total"] += 1
                if movement_correct:
                    model_analytics["movement_stats"][movement_label.item()]["correct"] += 1
            
            # Calculate accuracies for each movement class
            movement_accuracies = []
            print(f"     → Per-class breakdown:")
            for idx in range(7):
                stats = model_analytics["movement_stats"][idx]
                if stats["total"] > 0:
                    accuracy = (stats["correct"] / stats["total"]) * 100
                else:
                    accuracy = 0
                movement_accuracies.append(accuracy)
                # Print detailed breakdown
                print(f"        {MOVEMENT_LABELS[idx]:20s}: {stats['correct']:3d}/{stats['total']:3d} = {accuracy:6.2f}%")
            
            model_accuracy_data[model_name] = movement_accuracies
            
            # Calculate total samples tested
            total_samples = sum(stats["total"] for stats in model_analytics["movement_stats"].values())
            print(f"     ✓ Total: {total_samples} held-out windows | Avg Accuracy: {np.mean(movement_accuracies):.1f}%")
            
        except Exception as e:
            print(f"     ✗ Error processing model: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not model_accuracy_data:
        print("\n⚠ No valid models found for heatmap generation!")
        return
    
    # Create heatmap
    print("\n  Creating visualization...")
    
    model_names = list(model_accuracy_data.keys())
    accuracies = np.array(list(model_accuracy_data.values()))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(accuracies, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([MOVEMENT_LABELS[i] for i in range(7)], rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add accuracy values as text annotations
    for i in range(len(model_names)):
        for j in range(7):
            value = accuracies[i, j]
            color = 'white' if value < 75 else 'black'
            text = ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                          color=color, fontweight='bold', fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Movement Classifications', fontweight='bold', fontsize=12)
    ax.set_ylabel('Models', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance on Test Set (Held-Out Data) - Movement Classification Accuracy (%)\n' +
                 f'Evaluated on {actual_eval_count} windows from test split (seed={split_seed})',
                fontweight='bold', fontsize=13, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    
    # Save figure
    heatmap_file = os.path.join(HEATMAP_RESULTS_DIR, "model_performance_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Heatmap saved to: {heatmap_file}")
    
    # Save data as JSON
    
    # Collect per-class sample counts from the last model evaluated
    sample_counts_per_class = {}
    for idx in range(7):
        sample_counts_per_class[MOVEMENT_LABELS[idx]] = None  # Will be filled if available
    
    heatmap_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "test_set_only",
        "evaluation_details": {
            "train_split_ratio": train_split,
            "test_split_ratio": 1 - train_split,
            "random_seed": split_seed,
            "window_size": eval_window_size,
            "stride": eval_stride,
            "test_samples_available": len(test_dataset),
            "test_samples_evaluated": actual_eval_count,
            "max_test_samples_requested": max_test_samples,
            "subset_strategy": subset_strategy,
            "note": "All models are evaluated on the exact same selected held-out windows"
        },
        "models": model_names,
        "movements": [MOVEMENT_LABELS[i] for i in range(7)],
        "accuracy_matrix": accuracies.tolist(),
        "average_per_model": [float(np.mean(row)) for row in accuracies]
    }
    
    json_file = os.path.join(HEATMAP_RESULTS_DIR, "model_heatmap_data.json")
    with open(json_file, "w") as f:
        json.dump(heatmap_data, f, indent=2)
    
    print(f"✅ Heatmap data saved to: {json_file}")
    
    plt.close()

    print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EMG DATA VISUALIZATION & ANALYTICS TOOL")
    print("="*80)
    
    # Step 1: Select Model
    model_key, model_info = select_model()
    
    if model_info is None:
        print("\n❌ No valid model selected. Exiting.")
        exit(1)
    
    model, _ = load_trained_model(model_info["path"])
    
    # Initialize analytics data with selected model
    analytics_data = initialize_analytics_data(model_info["path"])
    
    print(f"\nTime Conversion: {SAMPLING_RATE_HZ} Hz → {TIME_PER_SAMPLE_MS} ms/sample")
    
    # Step 2: Select Generation Mode
    print("\n" + "="*80)
    print("DATA GENERATION MODES")
    print("="*80)
    print("\nWhat would you like to generate?")
    print("1. Raw EMG plots (without predictions)")
    print("2. Predicted EMG plots (with NN predictions)")
    print("3. Analytics report (requires predicted plots)")
    print("4. Model comparison heatmap (tests all available models)")
    print("5. Full pipeline (all of the above in sequence)")
    print("6. Analytics + Heatmap (skip raw/predicted plots)")
    
    choice = input("\nEnter your choice (1-6) or press Enter for full pipeline: ").strip()
    
    if choice == "" or choice == "5":
        choice = "5"
    
    generate_raw = choice in ["1", "5"]
    generate_predicted = choice in ["2", "5"]
    generate_analytics = choice in ["3", "5", "6"]
    generate_heatmap = choice in ["4", "5", "6"]
    
    # Load datasets if needed for single model analysis
    if generate_raw or generate_predicted or generate_analytics:
        print("\nLoading datasets...")
        tensors = DC.load_all_datasets()
        
        # Window size configuration
        window_samples = 200
        
        # Generate requested outputs
        if generate_raw:
            generate_raw_plots(tensors, window_samples)
        
        if generate_predicted:
            generate_predicted_plots(tensors, window_samples)
        
        if generate_analytics and len(analytics_data["predictions"]) > 0:
            generate_analytics_report()
        elif generate_analytics:
            print("\n⚠ Warning: Analytics require predicted plots to be generated first!")
            print("   Run option 2 or 5 to generate analytics.")
    
    # Generate heatmap (tests all models, independent of above)
    if generate_heatmap:
        heatmap_max_samples = prompt_heatmap_sample_limit(HEATMAP_MAX_TEST_SAMPLES)
        generate_model_heatmap(max_test_samples=heatmap_max_samples)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n📊 Time Conversion Calculations:")
    print(f"   Sampling Rate: {SAMPLING_RATE_HZ} Hz")
    print(f"   Time per sample: {TIME_PER_SAMPLE_MS} ms = {TIME_PER_SAMPLE_S} s")
    print(f"   Formula: Time (ms) = Sample_Index × {TIME_PER_SAMPLE_MS}")
    print(f"   Formula: Time (s) = Sample_Index × {TIME_PER_SAMPLE_S}")
    print(f"\n📁 Output Folders:")
    if generate_raw:
        print(f"   Raw Data: {RAW_RESULTS_DIR}")
    if generate_predicted:
        print(f"   Predicted Data: {PRED_RESULTS_DIR}")
    if generate_analytics:
        print(f"   Analytics: {os.path.join(RESULTS_DIR, 'analytics_report.json')}")
        print(f"   Analytics: {os.path.join(RESULTS_DIR, 'analytics_summary.txt')}")
    if generate_heatmap:
        print(f"   Heatmaps: {HEATMAP_RESULTS_DIR}")
    print("="*80)