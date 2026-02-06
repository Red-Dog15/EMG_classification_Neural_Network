import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import os
import sys
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import Data_Conversion as DC
from NN.predict import load_trained_model, predict_from_tensor
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS

# EMG Data Specifications from LibEMG ContractionIntensity Dataset
# Source: https://github.com/LibEMG/ContractionIntensity/blob/main/Info.txt
SAMPLING_RATE_HZ = 1000  # 1 kHz sampling frequency
TIME_PER_SAMPLE_MS = 1000 / SAMPLING_RATE_HZ  # 1.0 ms per sample
TIME_PER_SAMPLE_S = 1 / SAMPLING_RATE_HZ  # 0.001 s per sample

# Analytics tracking
analytics_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "model_path": "NN/models/best_model_full.pth",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_size": 100
    },
    "predictions": [],
    "movement_stats": {i: {"correct": 0, "total": 0} for i in range(7)},
    "severity_stats": {i: {"correct": 0, "total": 0} for i in range(3)},
    "combined_stats": {"correct": 0, "total": 0}
}

def samples_to_time_ms(samples):
    """Convert sample indices to time in milliseconds."""
    return samples * TIME_PER_SAMPLE_MS

def samples_to_time_s(samples):
    """Convert sample indices to time in seconds."""
    return samples * TIME_PER_SAMPLE_S

# Load the trained model once
MODEL_PATH = "NN/models/best_model_full.pth"
model, _ = load_trained_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
print(f"Time Conversion: {SAMPLING_RATE_HZ} Hz → {TIME_PER_SAMPLE_MS} ms/sample")

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
    plt.savefig(f"DATA/Results/Raw_Data/{file_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved Raw: DATA/Results/Raw_Data/{file_name}.png")
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
    plt.savefig(f"DATA/Results/Predicted_Data/{file_name}.png", dpi=300, bbox_inches="tight")
    print(f"Saved Predicted: DATA/Results/Predicted_Data/{file_name}.png - {status}")
    plt.close()

def generate_raw_plots(tensors, window_samples=200):
    """Generate raw EMG signal plots without predictions."""
    print("\n" + "="*80)
    print("GENERATING RAW EMG PLOTS")
    print("="*80)
    
    os.makedirs("DATA/Results/Raw_Data", exist_ok=True)
    
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
    
    print(f"\n✅ Raw plots saved to: DATA/Results/Raw_Data/")

def generate_predicted_plots(tensors, window_samples=200):
    """Generate EMG plots with NN predictions overlaid."""
    print("\n" + "="*80)
    print("GENERATING PREDICTED EMG PLOTS")
    print("="*80)
    
    os.makedirs("DATA/Results/Predicted_Data", exist_ok=True)
    
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
    
    print(f"\n✅ Predicted plots saved to: DATA/Results/Predicted_Data/")

def generate_analytics_report():
    """Generate analytics report from collected prediction data."""
    print("\n" + "="*80)
    print("GENERATING ANALYTICS REPORT")
    print("="*80)
    
    os.makedirs("DATA/Results", exist_ok=True)
    
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
    analytics_file = "DATA/Results/analytics_report.json"
    with open(analytics_file, "w") as f:
        json.dump(analytics_report, f, indent=2)
    
    # Save human-readable summary
    summary_file = "DATA/Results/analytics_summary.txt"
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EMG DATA VISUALIZATION & ANALYTICS TOOL")
    print("="*80)
    
    # User prompts for what to generate
    print("\nWhat would you like to generate?")
    print("1. Raw EMG plots (without predictions)")
    print("2. Predicted EMG plots (with NN predictions)")
    print("3. Analytics report")
    print("4. All of the above")
    
    choice = input("\nEnter your choice (1-4) or press Enter for all: ").strip()
    
    if choice == "":
        choice = "4"
    
    generate_raw = choice in ["1", "4"]
    generate_predicted = choice in ["2", "4"]
    generate_analytics = choice in ["3", "4"]
    
    # Load datasets once
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
        print("   Run option 2 or 4 to generate analytics.")
    
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
        print(f"   Raw Data: DATA/Results/Raw_Data/")
    if generate_predicted:
        print(f"   Predicted Data: DATA/Results/Predicted_Data/")
    if generate_analytics:
        print(f"   Analytics: DATA/Results/")
    print("="*80)