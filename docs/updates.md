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
