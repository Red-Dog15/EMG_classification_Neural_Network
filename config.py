"""
SHARED CONFIGURATION FOR EMG TRAINING AND EVALUATION
=====================================================

⚠️ CRITICAL: train.py and Data_visualization.py MUST use the same window settings!
If they differ, you'll evaluate on different data than you trained on.

This file ensures consistency across training and evaluation.
"""

# ============================================================================
# WINDOW CONFIGURATION - Controls how many windows are created from raw data
# ============================================================================

# WINDOW_SIZE: How many consecutive EMG timesteps form one input sample
# - Larger window = fewer windows, more temporal context
# - Smaller window = more windows, less temporal context
# - Example: 100 timesteps = 100ms @ 1kHz sampling rate
WINDOW_SIZE = 100

# STRIDE: Step size when sliding the window across the data  
# - Controls overlap between consecutive windows
# - Smaller stride = MORE windows (more overlap)
# - Larger stride = FEWER windows (less overlap)
# 
# TO INCREASE TOTAL WINDOWS: Decrease this value!
# Examples with 3000 timesteps per CSV:
#   Stride=50  → (3000-100)/50  + 1 = 59 windows per CSV  → 1,239 total windows
#   Stride=25  → (3000-100)/25  + 1 = 117 windows per CSV → 2,457 total windows
#   Stride=10  → (3000-100)/10  + 1 = 291 windows per CSV → 6,111 total windows
#   Stride=1   → (3000-100)/1   + 1 = 2,901 windows per CSV → 60,921 total windows
STRIDE = 10

# TRAIN_SPLIT: Fraction of windows used for training vs testing
# - 0.8 = 80% training, 20% testing
# - Decrease to get more test samples (but less training data)
TRAIN_SPLIT = 0.8

# SPLIT_SEED: Random seed for reproducible train/test splits
SPLIT_SEED = 42

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

# Increased epochs to give stricter split training enough optimization time.
NUM_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Multi-task loss weighting — equal weight so severity head receives full gradient
MOVEMENT_LOSS_WEIGHT = 1.0
SEVERITY_LOSS_WEIGHT = 1.0

# L2 regularization strength applied to all model weights via Adam weight_decay.
# Prevents the model from memorising the small training set.
WEIGHT_DECAY = 1e-4

# Early stopping monitor options: "loss", "movement_acc", "severity_acc"
# movement_acc: saves the epoch that peaks on test movement accuracy (recommended).
# Do NOT use "loss" here — test loss diverges from epoch 1 so the monitor would
# always save the epoch-1 near-random model.
EARLY_STOPPING_MONITOR = "movement_acc"
EARLY_STOPPING_MIN_DELTA = 0.001

# ============================================================================
# CALCULATION HELPER
# ============================================================================

def calculate_windows_per_csv(csv_length=3000, window_size=WINDOW_SIZE, stride=STRIDE):
    """Calculate how many windows are created from one CSV file."""
    if csv_length < window_size:
        return 0
    return (csv_length - window_size) // stride + 1


def calculate_total_windows(num_csvs=21, csv_length=3000):
    """Calculate total windows from all CSV files."""
    windows_per_csv = calculate_windows_per_csv(csv_length, WINDOW_SIZE, STRIDE)
    return num_csvs * windows_per_csv


def print_config_summary():
    """Print current configuration and its impact."""
    num_csvs = 21  # 7 movements × 3 severities
    csv_length = 3000
    
    windows_per_csv = calculate_windows_per_csv(csv_length)
    total_windows = calculate_total_windows(num_csvs, csv_length)
    train_windows = int(total_windows * TRAIN_SPLIT)
    test_windows = total_windows - train_windows
    
    print("\n" + "="*80)
    print("📊 CURRENT WINDOW CONFIGURATION")
    print("="*80)
    print(f"\nWindow Settings:")
    print(f"  • WINDOW_SIZE = {WINDOW_SIZE} timesteps ({WINDOW_SIZE}ms @ 1kHz)")
    print(f"  • STRIDE = {STRIDE} timesteps ({(STRIDE/WINDOW_SIZE)*100:.0f}% overlap)")
    print(f"  • TRAIN_SPLIT = {TRAIN_SPLIT} ({TRAIN_SPLIT*100:.0f}% train, {(1-TRAIN_SPLIT)*100:.0f}% test)")
    
    print(f"\nResulting Windows:")
    print(f"  • Windows per CSV: {windows_per_csv}")
    print(f"  • Total windows: {total_windows:,} (from {num_csvs} CSVs)")
    print(f"  • Training windows: {train_windows:,}")
    print(f"  • Test windows: {test_windows:,}")
    
    print("\n" + "="*80)
    print("🎯 TO INCREASE/DECREASE WINDOW COUNT:")
    print("="*80)
    
    # Show alternatives
    for new_stride in [1, 10, 25, 50, 100]:
        alt_windows_per_csv = calculate_windows_per_csv(csv_length, WINDOW_SIZE, new_stride)
        alt_total = num_csvs * alt_windows_per_csv
        alt_test = int(alt_total * (1 - TRAIN_SPLIT))
        change = "←" if new_stride == STRIDE else ""
        print(f"  • STRIDE={new_stride:3d} → {alt_total:5,} total windows ({alt_test:4,} test) {change}")
    
    print("\n💡 Remember: Smaller stride = MORE windows = Longer training time")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_config_summary()
