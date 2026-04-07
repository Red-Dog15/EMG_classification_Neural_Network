# EMG Data Time Conversion Calculations

## Data Source
**Dataset**: LibEMG Contraction Intensity  
**GitHub**: https://github.com/LibEMG/ContractionIntensity  
**Reference**: Info.txt

## Sampling Specifications

- **Sampling Rate**: 1000 Hz (1 kHz)
- **Bit Depth**: 16-bit
- **Channels**: 8 EMG channels
- **Filter**: 5th order Butterworth low-pass at 450 Hz

## Time Conversion Formulas

### From Sample Index to Time

```python
# Constants
SAMPLING_RATE_HZ = 1000  # Hz
TIME_PER_SAMPLE_MS = 1000 / SAMPLING_RATE_HZ  # = 1.0 ms
TIME_PER_SAMPLE_S = 1 / SAMPLING_RATE_HZ       # = 0.001 s

# Conversion Functions
Time (milliseconds) = Sample_Index × 1.0 ms
Time (seconds) = Sample_Index × 0.001 s
```

### From Time to Sample Index

```python
Sample_Index = Time (ms) / 1.0 ms
Sample_Index = Time (s) × 1000
```

## Examples

| Sample Index | Time (ms) | Time (s) |
|--------------|-----------|----------|
| 0            | 0         | 0.000    |
| 100          | 100       | 0.100    |
| 200          | 200       | 0.200    |
| 1000         | 1000      | 1.000    |
| 5000         | 5000      | 5.000    |

## Window Calculations

### Typical NN Prediction Window
- **Window Size**: 100 samples
- **Duration**: 100 × 1.0 ms = **100 ms** = **0.1 s**

### Visualization Window (Current Settings)
- **Window Size**: 200 samples
- **Duration**: 200 × 1.0 ms = **200 ms** = **0.2 s**

## Implementation in Code

```python
import torch

def samples_to_time_ms(samples):
    """Convert sample indices to time in milliseconds."""
    return samples * 1.0  # 1 ms per sample at 1 kHz

def samples_to_time_s(samples):
    """Convert sample indices to time in seconds."""
    return samples * 0.001  # 0.001 s per sample at 1 kHz

# Example usage:
time_axis_ms = samples_to_time_ms(torch.arange(200))  # For 200-sample window
# Result: [0.0, 1.0, 2.0, ..., 199.0] ms
```

## Visualization Output

### Folders Structure
```
DATA/Results/
├── Raw_Data/          # Raw EMG signals with time axis
└── Predicted_Data/    # EMG signals with NN predictions overlaid
```

### Plot Features
- **X-axis**: Time in milliseconds (converted from sample indices)
- **Y-axis**: EMG amplitude in mV (millivolts)
- **Color Coding** (Predicted Data only):
  - 🟢 Green: Correct movement and severity prediction
  - 🟠 Orange: Correct movement, incorrect severity
  - 🔴 Red: Incorrect movement prediction
