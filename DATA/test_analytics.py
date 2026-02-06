"""Quick test to generate analytics without all visualizations"""
import matplotlib
matplotlib.use('Agg')
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

# Load model
MODEL_PATH = "NN/models/best_model_full.pth"
model, _ = load_trained_model(MODEL_PATH)

# Analytics tracking
analytics_data = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "window_size": 100
    },
    "predictions": [],
    "movement_stats": {i: {"correct": 0, "total": 0} for i in range(7)},
    "severity_stats": {i: {"correct": 0, "total": 0} for i in range(3)},
    "combined_stats": {"correct": 0, "total": 0}
}

print("Loading datasets...")
tensors = DC.load_all_datasets()

print("\nRunning predictions...")
window_samples = 200

# Test on all movements/severities
for severity_name, severity_idx in [("Light", 0), ("Medium", 1), ("Hard", 2)]:
    for movement_idx in range(7):
        tensor_data = tensors[severity_name][movement_idx]
        start_idx = len(tensor_data) // 2
        window = tensor_data[start_idx:start_idx + window_samples]
        
        # Get prediction
        prediction = predict_from_tensor(model, window[-100:], window_size=100)
        
        movement_correct = prediction['movement_pred'] == movement_idx
        severity_correct = prediction['severity_pred'] == severity_idx
        both_correct = movement_correct and severity_correct
        
        # Update analytics
        analytics_data["predictions"].append({
            "movement": MOVEMENT_LABELS[movement_idx],
            "severity": severity_name,
            "movement_correct": movement_correct,
            "severity_correct": severity_correct,
            "both_correct": both_correct
        })
        
        analytics_data["movement_stats"][movement_idx]["total"] += 1
        if movement_correct:
            analytics_data["movement_stats"][movement_idx]["correct"] += 1
        
        analytics_data["severity_stats"][severity_idx]["total"] += 1
        if severity_correct:
            analytics_data["severity_stats"][severity_idx]["correct"] += 1
        
        analytics_data["combined_stats"]["total"] += 1
        if both_correct:
            analytics_data["combined_stats"]["correct"] += 1
        
        status = "✓" if both_correct else ("⚠" if movement_correct else "✗")
        print(f"{status} {severity_name:6} {MOVEMENT_LABELS[movement_idx]:20}")

# Calculate metrics
movement_accuracy = analytics_data["movement_stats"]
severity_accuracy = analytics_data["severity_stats"]
combined_accuracy = analytics_data["combined_stats"]

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

total_movement_correct = sum(s["correct"] for s in movement_accuracy.values())
total_movement_samples = sum(s["total"] for s in movement_accuracy.values())
avg_movement_accuracy = total_movement_correct / total_movement_samples

total_severity_correct = sum(s["correct"] for s in severity_accuracy.values())
total_severity_samples = sum(s["total"] for s in severity_accuracy.values())
avg_severity_accuracy = total_severity_correct / total_severity_samples

avg_overall_accuracy = combined_accuracy["correct"] / combined_accuracy["total"]

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

# Save analytics
os.makedirs("DATA/Results", exist_ok=True)

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

with open("DATA/Results/analytics_report.json", "w") as f:
    json.dump(analytics_report, f, indent=2)

with open("DATA/Results/analytics_summary.txt", "w", encoding='utf-8') as f:
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

print("\n✅ Analytics saved to:")
print("   • DATA/Results/analytics_report.json")
print("   • DATA/Results/analytics_summary.txt")
print("="*80)
