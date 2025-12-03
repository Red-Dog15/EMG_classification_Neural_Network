# EMG Classification System - Technical Methodology

## 1. Overview

This system implements a deep learning approach for classifying hand movements and contraction intensity from surface electromyography (sEMG) signals. The model performs multi-task learning to simultaneously predict:

1. **Movement Classification**: 7 hand/wrist movements
2. **Severity Classification**: 3 contraction intensity levels

## 2. Dataset Structure

### 2.1 Raw Data Format

**Input Files**: CSV files containing 8-channel EMG recordings
- **Format**: `S{subject}_{severity}_C{class}_R{repetition}.csv`
- **Channels**: 8 EMG electrodes (Channel_1 through Channel_8)
- **Sampling**: ~200 timesteps per recording
- **Values**: Normalized EMG amplitude (-1 to 1 range)

**Example**:
```csv
Channel_1,Channel_2,Channel_3,Channel_4,Channel_5,Channel_6,Channel_7,Channel_8
0.015106,-0.12955,-0.0047303,-0.044709,0.033417,0.0062562,-0.019074,-0.024872
...
```

### 2.2 Class Labels

**Movement Classes (7 total)**:
- Class 0: No Movement
- Class 1: Wrist Flexion
- Class 2: Wrist Extension
- Class 3: Wrist Pronation
- Class 4: Wrist Supination
- Class 5: Chuck Grip
- Class 6: Hand Open

**Severity Levels (3 total)**:
- Level 0: Light contraction
- Level 1: Medium contraction
- Level 2: Hard contraction

### 2.3 Dataset Composition

**Current Dataset**:
- 21 total recordings
- 3 severity levels × 7 movements = 21 combinations
- Each recording: ~200 samples × 8 channels

**Distribution**:
- Balanced across movements (3 recordings per movement class)
- Balanced across severities (7 recordings per severity level)

## 3. Data Processing Pipeline

### 3.1 Data Loading (Data_Conversion.py)

**Step 1: CSV to DataFrame**
```python
df = pd.read_csv(csv_path)  # Shape: (num_samples, 8)
```

**Step 2: DataFrame to PyTorch Tensor**
```python
tensor = torch.tensor(df.values, dtype=torch.float32)
# Shape: (num_samples, 8)
```

**Step 3: Label Assignment**
```python
labeled_data = [(tensor, movement_idx, severity_idx), ...]
# Each tuple contains:
#   - tensor: (num_samples, 8) EMG data
#   - movement_idx: 0-6 (movement class)
#   - severity_idx: 0-2 (severity level)
```

### 3.2 Sliding Window Approach (dataset.py)

To create fixed-length inputs for the neural network, a sliding window technique is applied:

**Parameters**:
- `window_size = 100`: Each sample contains 100 consecutive timesteps
- `stride = 50`: Windows overlap by 50% (50 timesteps)

**Process**:
```
Original recording: [200 samples × 8 channels]

Window 1:  samples[0:100]   → (100, 8)
Window 2:  samples[50:150]  → (100, 8)  [50% overlap]
Window 3:  samples[100:200] → (100, 8)
```

**Result**:
- From 21 recordings → ~1,240 training samples
- Each sample: (100 timesteps, 8 channels)

### 3.3 Train/Test Split

**Split Ratio**: 80% training, 20% testing
- Training samples: 992
- Testing samples: 248

**Method**: Random split with fixed seed (seed=42) for reproducibility

### 3.4 Batch Processing

**Batch Size**: 32 samples per batch
- Training batches: 31 batches per epoch
- Testing batches: 8 batches per epoch

## 4. Neural Network Architecture

### 4.1 Model Type: Multi-Task CNN-GRU

The network uses a **shared backbone** with **task-specific heads** for multi-task learning.

### 4.2 Architecture Breakdown

```
Input: (batch_size, 100, 8)
  ↓
┌─────────────────────────────────┐
│   CNN Feature Extraction        │
│                                 │
│  Conv1D(8→32, k=5) + BatchNorm │
│  ReLU + MaxPool(2)              │
│  → Shape: (batch, 32, 50)       │
│                                 │
│  Conv1D(32→64, k=3) + BatchNorm│
│  ReLU + MaxPool(2)              │
│  → Shape: (batch, 64, 25)       │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│   Temporal Feature Extraction   │
│                                 │
│  Bidirectional GRU              │
│  - Input size: 64               │
│  - Hidden size: 64              │
│  - Num layers: 2                │
│  - Bidirectional: True          │
│  → Output: (batch, 25, 128)     │
│  → Last timestep: (batch, 128)  │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│   Shared Feature Layer          │
│                                 │
│  Linear(128 → 128)              │
│  ReLU + Dropout(0.3)            │
└─────────────────────────────────┘
  ↓
  ├──────────────────┬──────────────────┐
  ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐
│Movement Head │  │Severity Head │
│              │  │              │
│Linear(128→64)│  │Linear(128→64)│
│ReLU          │  │ReLU          │
│Dropout(0.3)  │  │Dropout(0.3)  │
│Linear(64→7)  │  │Linear(64→3)  │
└──────────────┘  └──────────────┘
  ↓                  ↓
Movement Logits    Severity Logits
(batch, 7)         (batch, 3)
```

### 4.3 Layer Details

**1. Convolutional Layers (Spatial Feature Extraction)**

Purpose: Extract patterns across the 8 EMG channels

- **Conv1D Layer 1**:
  - Input channels: 8 (EMG electrodes)
  - Output channels: 32
  - Kernel size: 5 (captures 5-timestep patterns)
  - Padding: 2 (maintains temporal dimension)
  
- **Conv1D Layer 2**:
  - Input channels: 32
  - Output channels: 64
  - Kernel size: 3
  - Padding: 1

- **MaxPool**: Reduces sequence length by 2× after each conv layer

**2. Recurrent Layers (Temporal Feature Extraction)**

Purpose: Capture temporal dependencies in EMG signals

- **GRU (Gated Recurrent Unit)**:
  - Type: Bidirectional (processes sequence forward and backward)
  - Hidden size: 64 per direction (128 total)
  - Num layers: 2 (stacked)
  - Dropout: 0.3 (between layers)
  
  Advantages over LSTM:
  - Fewer parameters (faster training)
  - Better for shorter sequences
  - Less prone to overfitting

**3. Task-Specific Heads**

Purpose: Separate classification for movement and severity

- **Movement Head**:
  - Input: 128 features
  - Hidden: 64 features
  - Output: 7 classes (movement types)
  
- **Severity Head**:
  - Input: 128 features
  - Hidden: 64 features
  - Output: 3 classes (severity levels)

### 4.4 Model Parameters

**Total Parameters**: 165,802
- Convolutional layers: ~20,000
- GRU layers: ~130,000
- Fully connected layers: ~15,000

**Model Size**: ~650 KB (saved as .pth file)

## 5. Training Process

### 5.1 Training Strategy: Multi-Task Learning

The model is trained to minimize loss on both tasks simultaneously.

**Advantages**:
- Shared representations between tasks
- More efficient use of limited data
- Regularization effect (reduces overfitting)

### 5.2 Loss Function

**Multi-Task Loss**:
```python
Total_Loss = λ₁ × Movement_Loss + λ₂ × Severity_Loss
```

Where:
- `λ₁ = 1.0` (movement weight)
- `λ₂ = 1.0` (severity weight)

**Component Losses**:
- Movement Loss: CrossEntropyLoss over 7 classes
- Severity Loss: CrossEntropyLoss over 3 classes

**Cross-Entropy Loss Formula**:
```
L = -∑ yᵢ log(ŷᵢ)
```
Where:
- `yᵢ`: True label (one-hot encoded)
- `ŷᵢ`: Predicted probability (from softmax)

### 5.3 Optimization

**Optimizer**: Adam (Adaptive Moment Estimation)
- Learning rate: 0.001
- Beta1: 0.9 (default)
- Beta2: 0.999 (default)
- Epsilon: 1e-8 (default)

**Learning Rate Scheduling**:
- Strategy: ReduceLROnPlateau
- Monitor: Test loss
- Reduction factor: 0.5 (halve learning rate)
- Patience: 5 epochs (wait 5 epochs before reducing)

### 5.4 Regularization Techniques

**1. Dropout (p=0.3)**
- Applied after shared features
- Applied in task-specific heads
- Randomly zeroes 30% of neurons during training
- Prevents overfitting

**2. Batch Normalization**
- Applied after each convolutional layer
- Normalizes activations
- Accelerates training
- Provides mild regularization

**3. Early Stopping (implicit)**
- Saves best model based on test loss
- Prevents overtraining

### 5.5 Training Loop

**Epochs**: 30 (one complete pass through training data)

**Per Epoch**:
1. **Training Phase**:
   ```python
   for batch in train_loader:
       # Forward pass
       movement_pred, severity_pred = model(batch_data)
       
       # Calculate loss
       loss = multi_task_loss(predictions, targets)
       
       # Backward pass
       loss.backward()
       optimizer.step()
   ```

2. **Evaluation Phase**:
   ```python
   with torch.no_grad():  # No gradient computation
       for batch in test_loader:
           predictions = model(batch_data)
           metrics = calculate_metrics(predictions, targets)
   ```

3. **Metrics Logging**:
   - Training loss (both tasks)
   - Testing loss (both tasks)
   - Training accuracy (movement & severity)
   - Testing accuracy (movement & severity)

4. **Model Checkpointing**:
   - Save if test loss improves
   - Store model weights, optimizer state, metrics

### 5.6 Metrics

**Accuracy Calculation**:
```python
predictions = argmax(logits)  # Get class with highest probability
accuracy = (predictions == targets).mean()
```

**Tracked Metrics**:
- Movement classification accuracy (% correct)
- Severity classification accuracy (% correct)
- Combined loss (weighted sum)
- Individual task losses

## 6. Inference Process

### 6.1 Model Loading

```python
checkpoint = torch.load('best_model_full.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode
```

### 6.2 Prediction Pipeline

**Input**: New EMG recording (CSV file or tensor)

**Steps**:
1. Load CSV → Tensor (N, 8)
2. Extract window (last 100 samples)
3. Add batch dimension → (1, 100, 8)
4. Forward pass through model
5. Apply softmax to get probabilities
6. Take argmax for final prediction

**Output**:
```python
{
    'movement_pred': 6,                    # Class index
    'movement_name': 'Hand_Open',          # Class name
    'movement_confidence': 0.923,          # Probability
    'severity_pred': 2,
    'severity_name': 'Hard',
    'severity_confidence': 0.887
}
```

## 7. Implementation Details

### 7.1 Software Stack

**Core Libraries**:
- PyTorch 2.0+: Deep learning framework
- NumPy: Numerical operations
- Pandas: Data loading and manipulation

**Optional**:
- TensorBoard: Training visualization
- Matplotlib: Plotting (future work)

### 7.2 Hardware Requirements

**Minimum**:
- CPU: Any modern processor
- RAM: 4GB
- Storage: 1GB

**Recommended**:
- GPU: CUDA-compatible (NVIDIA)
- RAM: 8GB+
- Training time reduction: 5-10× faster

### 7.3 File Structure

```
Scripts/
├── DATA/
│   ├── Data_Conversion.py    # CSV → Tensor + labels
│   ├── dataset.py            # Sliding window + DataLoader
│   └── Example_data/         # CSV files
│
├── NN/
│   ├── network.py            # Model architecture
│   ├── train.py              # Training script
│   ├── predict.py            # Inference utilities
│   └── main.py               # Demo workflow
│
├── models/                   # Saved checkpoints (.pth)
└── runs/                     # TensorBoard logs
```

## 8. Expected Performance

### 8.1 Accuracy Ranges

Based on current dataset (21 recordings):

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Movement Accuracy | 85-95% | Higher with more data |
| Severity Accuracy | 80-90% | Harder task (subtle differences) |
| Training Time (CPU) | 5-10 min | 30 epochs |
| Training Time (GPU) | 1-2 min | 30 epochs |

### 8.2 Factors Affecting Performance

**Positive Factors**:
- Balanced dataset
- Multi-task learning
- Dropout regularization
- Data augmentation (sliding windows)

**Limiting Factors**:
- Small dataset (only 21 recordings)
- Single subject (S1)
- No cross-subject validation

## 9. Advantages & Limitations

### 9.1 Advantages

1. **Multi-Task Learning**: Efficient use of data, shared representations
2. **Temporal Modeling**: GRU captures time-dependent EMG patterns
3. **Moderate Complexity**: 165K parameters - trainable on CPU
4. **Real-Time Capable**: Fast inference (~10ms per prediction)
5. **Interpretable**: Separate heads for each task

### 9.2 Limitations

1. **Data Scarcity**: Only 21 recordings (needs more subjects)
2. **No Data Augmentation**: Could add noise, time warping
3. **Fixed Window**: 100-timestep window may not fit all use cases
4. **Single Subject**: Model may not generalize to new users
5. **No Online Learning**: Cannot adapt to user during deployment

## 10. Future Improvements

### 10.1 Data Collection
- Multiple subjects (10+ participants)
- Multiple sessions per subject
- More repetitions per movement

### 10.2 Model Enhancements
- Attention mechanisms (focus on important timesteps)
- ResNet-style skip connections
- Transformer encoder (alternative to GRU)
- Continuous severity prediction (regression)

### 10.3 Training Improvements
- Cross-validation (k-fold)
- Data augmentation (noise, scaling, time warping)
- Class weighting (if imbalanced)
- Hyperparameter optimization (Optuna, Ray Tune)

### 10.4 Deployment
- Model quantization (INT8 for microcontrollers)
- ONNX export (framework-agnostic)
- Real-time streaming inference
- Mobile app integration

## 11. References

### 11.1 Dataset Source
- LibEMG Contraction Intensity Dataset
- GitHub: https://github.com/LibEMG/ContractionIntensity

### 11.2 Key Concepts
- **Multi-Task Learning**: Learning multiple related tasks simultaneously
- **Sliding Window**: Overlapping segments for data augmentation
- **Bidirectional GRU**: Processes sequences in both directions
- **Cross-Entropy Loss**: Standard loss for classification tasks
- **Adam Optimizer**: Adaptive learning rate optimization

### 11.3 Model Architecture Inspiration
- Convolutional features: Standard for time-series analysis
- GRU temporal modeling: Common in EMG/EEG classification
- Multi-task heads: Proven effective for related tasks
