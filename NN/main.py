import os, sys

# Add parent directory of this file (Scripts/) to module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from DATA.Data_Conversion import (
    tensors_dict, 
    create_labeled_dataset, 
    get_num_classes,
    MOVEMENT_LABELS,
    SEVERITY_LABELS
)
from DATA.dataset import get_dataset_statistics
from NN.train import train_model
from NN.predict import load_trained_model, predict_from_csv, print_prediction


def main():
    """
    Main entry point for EMG classification.
    Demonstrates the complete workflow from data loading to prediction.
    """
    print("="*50)
    print("EMG Movement & Severity Classification")
    print("="*50)
    
    # 1. Show available data
    print("\n--- Step 1: Data Overview ---")
    print(f"Available intensities: {list(tensors_dict.keys())}")
    print(f"Movements per intensity: {len(tensors_dict['Light'])}")
    print(f"\nMovement classes: {list(MOVEMENT_LABELS.values())}")
    print(f"Severity levels: {list(SEVERITY_LABELS.values())}")
    
    # Show first tensor shape
    sample_tensor = tensors_dict["Hard"][0]
    print(f"\nExample tensor shape (Hard, No Movement): {sample_tensor.shape}")
    print(f"  - {sample_tensor.shape[0]} time samples")
    print(f"  - {sample_tensor.shape[1]} EMG channels")
    
    # 2. Create labeled dataset
    print("\n--- Step 2: Creating Labeled Dataset ---")
    labeled_data = create_labeled_dataset()
    get_dataset_statistics(labeled_data)
    
    num_movements, num_severities = get_num_classes()
    print(f"Total classes - Movements: {num_movements}, Severities: {num_severities}")
    
    # 3. Training option
    print("\n--- Step 3: Training ---")
    print("To train the model, uncomment the training section below or run:")
    print("  python NN/train.py")
    print("\nThis will:")
    print("  - Create sliding windows from the EMG data")
    print("  - Split data into train/test sets")
    print("  - Train a CNN+GRU model to predict movement and severity")
    print("  - Save the best model to ./models/")
    print("  - Log training metrics to TensorBoard")
    
    # Uncomment to train:
    # train_mode = input("\nTrain model now? (y/n): ").lower()
    # if train_mode == 'y':
    #     model, metrics = train_model(
    #         model_type='full',
    #         num_epochs=30,
    #         batch_size=32,
    #         learning_rate=0.001,
    #         window_size=100,
    #         stride=50
    #     )
    #     print(f"\nTraining complete!")
    #     print(f"Movement Accuracy: {metrics['movement_acc']*100:.2f}%")
    #     print(f"Severity Accuracy: {metrics['severity_acc']*100:.2f}%")
    
    # 4. Prediction demo
    print("\n--- Step 4: Prediction (Demo) ---")
    model_path = "./models/final_model_full.pth"
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model, checkpoint = load_trained_model(model_path)
        
        # Predict on example file
        csv_path = "./DATA/Example_data/S1_Hard_C7_R1.csv"
        if os.path.exists(csv_path):
            print(f"\nMaking prediction on: {csv_path}")
            results = predict_from_csv(model, csv_path, window_size=100)
            print_prediction(results, verbose=True)
    else:
        print(f"No trained model found at {model_path}")
        print("Please train a model first!")
    
    print("\n" + "="*50)
    print("Workflow complete!")
    print("="*50)


if __name__ == "__main__":
    main()