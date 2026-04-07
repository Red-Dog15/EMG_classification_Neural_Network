# EMG Data Visualization & Analytics Guide

## Quick Start

Run the visualization tool:
```bash
python DATA/Data_visualization.py
```

## Menu Options

When you run the script, you'll see:

```
What would you like to generate?
1. Raw EMG plots (without predictions)
2. Predicted EMG plots (with NN predictions)
3. Analytics report
4. All of the above

Enter your choice (1-4) or press Enter for all:
```

### Option 1: Raw EMG Plots
- Generates clean EMG signal visualizations
- No NN predictions overlaid
- Useful for: Data inspection, signal quality checks
- Output: `DATA/Results/Raw_Data/` (21 plots)
- Time: ~30 seconds

### Option 2: Predicted EMG Plots ⭐ Recommended
- Generates EMG plots with NN predictions
- Color-coded by prediction accuracy:
  - 🟢 **Green**: Both movement and severity correct
  - 🟠 **Orange**: Movement correct, severity wrong
  - 🔴 **Red**: Movement incorrect
- Tracks analytics data automatically
- Output: `DATA/Results/Predicted_Data/` (21 plots)
- Time: ~45 seconds

### Option 3: Analytics Report
- Generates comprehensive analytics from prediction data
- **Requires**: Option 2 or 4 to be run first
- Outputs:
  - `analytics_report.json` - Machine-readable data
  - `analytics_summary.txt` - Human-readable summary
- Includes:
  - Average movement accuracy
  - Average severity accuracy
  - Combined overall accuracy
  - Per-class performance breakdown
  - Critical classes identification (< 70% accuracy)

### Option 4: All of the Above
- Generates everything: Raw plots + Predicted plots + Analytics
- Most comprehensive analysis
- Output: All three folders/files
- Time: ~90 seconds

## Output Structure

```
DATA/Results/
├── Raw_Data/                    # Option 1
│   ├── Light_No_Movement.png
│   ├── Light_Wrist_Flexion.png
│   └── ... (21 total)
│
├── Predicted_Data/              # Option 2
│   ├── Light_No_Movement.png
│   ├── Light_Wrist_Flexion.png
│   └── ... (21 total)
│
├── analytics_report.json        # Option 3
└── analytics_summary.txt        # Option 3
```

## Functions Available

### For Custom Scripts

```python
from DATA.Data_visualization import (
    generate_raw_plots,
    generate_predicted_plots,
    generate_analytics_report
)

# Load data
import Data_Conversion as DC
tensors = DC.load_all_datasets()

# Generate specific outputs
generate_raw_plots(tensors, window_samples=200)
generate_predicted_plots(tensors, window_samples=200)
generate_analytics_report()
```

## Understanding Analytics

### Movement Accuracy
- Percentage of samples where predicted movement matches true movement
- Ignores severity predictions

### Severity Accuracy
- Percentage of samples where predicted severity matches true severity
- Ignores movement predictions

### Combined Accuracy
- Percentage of samples where BOTH movement AND severity are correct
- Most stringent metric

### Critical Classes
- Classes with < 70% accuracy
- Indicates areas needing model improvement
- Listed separately for movements and severities

## Tips

1. **First Run**: Use Option 4 to get complete analysis
2. **Quick Check**: Use Option 2 for fast prediction validation
3. **Data Quality**: Use Option 1 to inspect raw signals
4. **Iterative Testing**: Use Option 2 + 3 when testing model improvements

## Example Workflow

```bash
# Initial analysis
python DATA/Data_visualization.py
# Choose: 4 (All)

# After model retraining
python DATA/Data_visualization.py
# Choose: 2 (Predicted plots with analytics data)

python DATA/Data_visualization.py
# Choose: 3 (Generate new analytics report)
```

## Time Conversion Reference

All plots show time in milliseconds on the x-axis:
- **Sampling Rate**: 1000 Hz (1 kHz)
- **Formula**: Time (ms) = Sample_Index × 1.0 ms
- **Example**: Sample 200 = 200 ms = 0.2 seconds
