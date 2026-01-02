# EMG Classification Neural Network

Real-time classification system for hand movements and contraction intensity using surface electromyography (sEMG) signals.

## Project Overview

This undergraduate thesis project implements a multi-task deep learning model that predicts:
1. **Movement Type** (7 classes): No Movement, Wrist Flexion, Wrist Extension, Wrist Pronation, Wrist Supination, Chuck Grip, Hand Open
2. **Contraction Intensity** (3 levels): Light, Medium, Hard

## Results

**Final Model Performance** (30 epochs):
- Movement Classification: **98.05%** accuracy
- Severity Classification: **97.92%** accuracy
- Model Parameters: 165,802
- Inference Time: ~10ms per prediction (CPU)

## Real-Time Capabilities

The system supports continuous real-time classification:
- **Sliding window approach**: Processes last 100 samples from continuous EMG stream
- **Low latency**: <10ms prediction time enables smooth control
- **Adaptive transitions**: Model naturally handles movement transitions in the data stream
- **No retraining required**: Works on continuous input without special configuration

## Quick Start

### Training
```bash
python NN/train.py
```

### Prediction
```bash
python NN/main.py
```

## Files

- `DATA/Data_Conversion.py` - Data loading and preprocessing
- `DATA/Data_Mapping.py` - MyoSuite conversion utilities (NEW)
- `DATA/dataset.py` - PyTorch dataset with sliding windows
- `NN/network.py` - CNN-GRU model architecture
- `NN/train.py` - Training script
- `NN/predict.py` - Inference utilities
- `NN/main.py` - Main prediction script
- `Output/NNO.txt` - Neural network output storage
- `METHODOLOGY.md` - Complete technical documentation

## Requirements

```bash
pip install torch numpy pandas tensorboard
```

## Model Architecture

- **Input**: 100 timesteps × 8 EMG channels
- **Feature Extraction**: CNN (spatial) + Bidirectional GRU (temporal)
- **Output**: Dual heads for movement and severity classification
- **Training**: Multi-task learning with combined loss


## Notes on Model Selection

**Final Model vs Best Model**:
- `final_model_full.pth`: Model after complete training (30 epochs)
- `best_model_full.pth`: Model with lowest validation loss during training

For deployment, the **final model** typically shows:
- Higher confidence scores (more decisive predictions)
- Better generalization to real-time data
- Smoother transitions between movements

Use `final_model_full.pth` for real-time applications unless overfitting is observed.

## MyoSuite Integration (In Progress)

The system is being extended to interface with MyoSuite robotic simulation:

**Output Data Structure**:
```python
{
    'movement_pred': 6,                 # Hand_Open
    'movement_name': 'Hand_Open',
    'movement_probs': [0.0, 0.0, ..., 0.83],  # 7 movement probabilities
    'movement_confidence': 0.828,
    'severity_pred': 2,                 # Hard
    'severity_name': 'Hard', 
    'severity_probs': [0.0, 0.0, 1.0],  # 3 severity probabilities
    'severity_confidence': 0.9999,
    'num_windows': 59
}
```

**Conversion Pipeline**:
1. Parse NN output from `Output/NNO.txt`
2. Map movements to muscle activation patterns
3. Scale activations by severity/intensity
4. Blend using probability weights for smooth control
5. Format for MyoSuite muscle actuators

See `DATA/Data_Mapping.py` for implementation details.

## Citation

Dataset: LibEMG Contraction Intensity Dataset  
https://github.com/LibEMG/ContractionIntensity

