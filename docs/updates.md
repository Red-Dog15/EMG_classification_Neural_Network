 Project Updates

Development log for EMG Classification Neural Network

## Update 1.0 - Initial Implementation (December 3, 2025)

### Achievements
- ✓ Implemented multi-task CNN-GRU model for EMG classification
- ✓ Achieved 98.05% movement accuracy and 97.92% severity accuracy
- ✓ Created complete training and inference pipeline
- ✓ Validated real-time prediction capabilities (<10ms latency)

### Model Architecture
- CNN layers for spatial feature extraction (8 EMG channels)
- Bidirectional GRU for temporal modeling (100-timestep windows)
- Dual classification heads (7 movements × 3 severity levels)
- Total parameters: 165,802

### Dataset
- 21 recordings (LibEMG Contraction Intensity)
- 7 movement classes × 3 intensity levels
- Sliding window augmentation: 1,240 samples
- 80/20 train/test split

### Real-Time Implementation Notes
- Model processes 100-sample sliding windows
- Works on continuous EMG streams without modification
- Handles movement transitions naturally through overlapping windows
- No special configuration needed for real-time vs batch processing

### Technical Decisions
- **Using final_model_full.pth for deployment**: Higher confidence scores, better suited for continuous prediction
- **Window size (100 samples)**: Balances temporal context with responsiveness
- **50% window overlap**: Provides smooth predictions during transitions

### Next Steps
- [ ] Test with real-time EMG hardware integration
- [ ] Evaluate transition smoothing strategies
- [ ] Collect multi-subject data for generalization
- [ ] Implement model quantization for embedded deployment

---

## Update 2.0 - MyoSuite Integration Pipeline (January 2, 2026)

### Achievements
- ✓ Defined neural network output data structure specification
- ✓ Designed Data_Mapping module architecture for MyoSuite conversion
- ✓ Implemented utility functions for NN output parsing
- ✓ Created severity-to-intensity conversion framework

### NN Output Format Specification
Model predictions now output structured dictionaries containing:
```python
{
    'movement_pred': int,              # Predicted movement class (0-6)
    'movement_name': str,              # Human-readable movement name
    'movement_probs': ndarray,         # Probability distribution [7 values]
    'movement_confidence': float,      # Confidence score (0-1)
    'severity_pred': int,              # Predicted severity class (0-2)
    'severity_name': str,              # Severity level name
    'severity_probs': ndarray,         # Probability distribution [3 values]
    'severity_confidence': float,      # Confidence score (0-1)
    'num_windows': int                 # Number of windows processed
}
```

### Data Mapping Module Components

**Implemented Functions**:
- `data_parser(file)`: Reads NN output from text file
- `Get_Probable_Movements(data)`: Filters movements by probability threshold (>10%)
- `Severity_Converter(severity_level, max_severity)`: Scales severity to intensity multiplier (0-1)
- `activation_blender(probabilities, weights)`: Framework for probability-weighted muscle activation

**Implemented Classes**:
- `Muscle_Mapping`: Main class for MyoSuite conversion pipeline
  - Designed for single-instance reuse (prevents memory buildup)
  - Methods: `Muscle_activation_Index()`, `get_Activation_Pattern()`, `MyoSuiteFormatter()`

### Conversion Pipeline Architecture
Defined transformation stages:
1. **Parse NN Output**: Extract predictions from file/dictionary
2. **Movement Mapping**: Convert discrete movements to muscle activation patterns
3. **Intensity Scaling**: Apply severity-based magnitude scaling
4. **Probability Blending**: Weighted combination of multiple movement patterns
5. **Format Output**: Generate MyoSuite-compatible muscle activation array

### Design Decisions
- **Single instance pattern**: Muscle_Mapping class reused throughout runtime
- **Probability thresholding**: Movements <10% probability filtered out
- **Modular architecture**: Separation of concerns (parsing, mapping, formatting)
- **Extensible framework**: Ready for MyoSuite muscle model integration

### Next Steps
- [ ] Define MyoSuite muscle model specifications (number of muscles, names)
- [ ] Implement movement-to-muscle activation lookup table
- [ ] Create anatomically-based muscle activation patterns
- [ ] Test probability-weighted blending for smooth transitions
- [ ] Integrate with MyoSuite simulation environment
- [ ] Validate real-time control loop performance

---

## Update 3.0 - Refined Data Mapping Pipeline (January 15, 2026)

### Achievements
- ✓ Enhanced `Get_Probable_Movements()` to include severity information per movement
- ✓ Refactored `Severity_Converter()` to work with movement tuples instead of raw values
- ✓ Updated `activation_blender()` to combine probability and severity as weights
- ✓ Created coherent data flow pipeline from NN output to activation values

### Function Updates

**`Get_Probable_Movements(data)`** - Enhanced Output Format
- Now returns: `[(movement_name, probability, severity_level), ...]`
- Example: `[('Chuck_Grip', 0.145, 2), ('Hand_Open', 0.828, 2)]`
- Associates the predicted severity level with each probable movement
- Filters movements with probability > 10% threshold

**`Severity_Converter(probable_movements, max_severity=5)`** - Refactored
- Input: List of tuples from `Get_Probable_Movements()`
- Output: `[(movement_name, probability, scaled_severity), ...]`
- Converts severity levels (0-2) to normalized intensity multipliers (0.0-1.0)
- Example: severity=2 → scaled_severity=0.4 (with max_severity=5)
- Preserves movement name and probability for downstream processing

**`activation_blender(converted_movements)`** - Simplified Integration
- Input: List of tuples from `Severity_Converter()`
- Output: `[(movement_name, blended_activation), ...]`
- Blends probability with severity: `blended_activation = probability / scaled_severity`
- Uses inverse relationship: higher severity → lower multiplier needed
- Example: prob=0.828 / severity=0.4 = 2.069 final activation
- Provides weighted activation values for each probable movement

### Data Flow Pipeline
Complete transformation from NN prediction to activation values:

```python
# Step 1: Parse NN output
data = data_parser("Output/NNO.txt")

# Step 2: Extract probable movements with severity
probable_movements = Get_Probable_Movements(data)
# → [('Chuck_Grip', 0.145, 2), ('Hand_Open', 0.828, 2)]

# Step 3: Convert severity levels to intensity multipliers
converted_movements = Severity_Converter(probable_movements)
# → [('Chuck_Grip', 0.145, 0.4), ('Hand_Open', 0.828, 0.4)]

# Step 4: Blend probability and severity for final activation
blended_activations = activation_blender(converted_movements)
# → [('Chuck_Grip', 0.363), ('Hand_Open', 2.069)]
```

### Design Improvements
- **Coherent data structure**: Tuple format maintained through pipeline stages
- **Single-pass processing**: Each function adds value without redundant operations
- **Traceable transformations**: Movement names preserved throughout pipeline
- **Separation of concerns**: Distinct functions for filtering, scaling, and blending
- **Inverse severity scaling**: Probability / Severity = Higher activation for stronger signals

### Next Steps
- [ ] Implement `Muscle_Mapping` class methods for pattern lookup
- [ ] Define muscle activation patterns in `get_MyoSuite_Movement_LUT()`
- [ ] Extend `activation_blender()` for multi-muscle pattern blending
- [ ] Create `MyoSuiteFormatter()` for final output formatting
- [ ] Test with multiple simultaneous movements
- [ ] Validate smooth transitions between movement states
