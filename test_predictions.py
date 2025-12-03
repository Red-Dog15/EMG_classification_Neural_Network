"""
Quick test script to verify predictions across all files.
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from NN.predict import load_trained_model, predict_from_csv

# Load model
model, _ = load_trained_model("./models/final_model_full.pth")

# Test files with their expected labels
test_files = [
    ("./DATA/Example_data/S1_Hard_C1_R1.csv", "No_Movement", "Hard"),
    ("./DATA/Example_data/S1_Hard_C2_R1.csv", "Wrist_Flexion", "Hard"),
    ("./DATA/Example_data/S1_Hard_C3_R1.csv", "Wrist_Extension", "Hard"),
    ("./DATA/Example_data/S1_Hard_C4_R1.csv", "Wrist_Pronation", "Hard"),
    ("./DATA/Example_data/S1_Hard_C5_R1.csv", "Wrist_Supination", "Hard"),
    ("./DATA/Example_data/S1_Hard_C6_R1.csv", "Chuck_Grip", "Hard"),
    ("./DATA/Example_data/S1_Hard_C7_R1.csv", "Hand_Open", "Hard"),
]

print("Testing predictions on Hard intensity files...\n")
correct = 0
total = 0

for csv_path, expected_movement, expected_severity in test_files:
    if os.path.exists(csv_path):
        result = predict_from_csv(model, csv_path, window_size=100)
        movement_match = "✓" if result['movement_name'] == expected_movement else "✗"
        severity_match = "✓" if result['severity_name'] == expected_severity else "✗"
        
        print(f"{os.path.basename(csv_path):25s} | Expected: {expected_movement:20s} | Got: {result['movement_name']:20s} {movement_match} | Severity: {result['severity_name']} {severity_match}")
        
        if result['movement_name'] == expected_movement:
            correct += 1
        total += 1

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
