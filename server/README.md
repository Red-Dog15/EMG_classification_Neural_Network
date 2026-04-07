# EMG Real-Time Prediction Server

A command-line server for real-time EMG signal classification with dynamic file loading.

## Features

- **Real-time streaming prediction** from CSV files
- **Dynamic file switching** without restarting the server
- **Adjustable prediction rate** (default 10 Hz)
- **Thread-safe** prediction updates
- **Interactive CLI** with intuitive commands

## Quick Start

```bash
# Navigate to Scripts directory
cd Scripts

# Run the server
python server/realtime_server.py
```

## Usage

### Starting the Server

```bash
python server/realtime_server.py
```

You'll see the model load and enter interactive mode:

```
==================================================
EMG Real-Time Prediction Server
==================================================

Loading model from ./models/final_model_full.pth...
Model loaded successfully!
Movement Accuracy: 98.44%
Severity Accuracy: 96.74%

Available files in ./DATA/Example_data:
------------------------------------------------------------
   1. S1_Hard_C1_R1.csv
   2. S1_Hard_C2_R1.csv
   ...
------------------------------------------------------------

server>
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show available CSV files | `list` |
| `load <file>` | Load a CSV file by name | `load S1_Hard_C7_R1.csv` |
| `load <number>` | Load file by number from list | `load 7` |
| `start` | Begin streaming predictions | `start` |
| `stop` | Stop streaming | `stop` |
| `status` | Show current server status | `status` |
| `rate <hz>` | Set prediction rate (Hz) | `rate 20` |
| `help` | Show help message | `help` |
| `quit` / `exit` | Shutdown server | `quit` |

### Example Session

```
server> list
Available files in ./DATA/Example_data:
------------------------------------------------------------
   1. S1_Hard_C1_R1.csv
   2. S1_Hard_C2_R1.csv
   ...
   7. S1_Hard_C7_R1.csv
------------------------------------------------------------

server> load 7

[LOADED] S1_Hard_C7_R1.csv
  Samples: 3007
  Channels: 8
  Duration: ~3.01s (at 1000 Hz)

server> start

[STREAMING] Started real-time prediction from S1_Hard_C7_R1.csv
  Update rate: 10 Hz
  Window size: 100 samples

[  1.7%] Movement: Hand_Open           (99.8%) | Severity: Hard       (100.0%)
[  3.3%] Movement: Hand_Open           (99.9%) | Severity: Hard       (100.0%)
[  5.0%] Movement: Hand_Open           (99.7%) | Severity: Hard       (100.0%)
...

server> stop

[STOPPED] Streaming halted.

server> load S1_Medium_C3_R1.csv

[LOADED] S1_Medium_C3_R1.csv
  Samples: 2987
  Channels: 8
  Duration: ~2.99s (at 1000 Hz)

server> start

[STREAMING] Started real-time prediction from S1_Medium_C3_R1.csv
  Update rate: 10 Hz
  Window size: 100 samples

[  1.7%] Movement: Wrist_Extension     (98.2%) | Severity: Medium     (99.5%)
[  3.3%] Movement: Wrist_Extension     (99.1%) | Severity: Medium     (98.9%)
...

server> quit
Shutting down server...
```

## Configuration

### Prediction Rate

Adjust how frequently predictions are made (default: 10 Hz):

```
server> rate 20
Prediction rate set to 20.0 Hz
```

- Higher rates (20-50 Hz): More responsive, higher CPU usage
- Lower rates (5-10 Hz): Smoother, lower CPU usage

### Window Size

The window size (default: 100 samples) is set when starting the server. To change it, modify the code:

```python
server = EMGRealtimeServer(
    model_path="./models/final_model_full.pth",
    window_size=75  # Smaller window = faster response
)
```

## How It Works

1. **Buffering**: Maintains a rolling buffer of the last `window_size` EMG samples
2. **Streaming**: Reads samples from CSV file at specified rate
3. **Prediction**: Makes prediction every time buffer is full
4. **Display**: Shows real-time predictions with confidence scores

### Data Flow

```
[CSV File] → [Sample Reader] → [Rolling Buffer] → [Model] → [Prediction Display]
             (10 Hz)           (100 samples)     (CNN+GRU)   (Console Output)
```

## Real-Time Deployment

This server simulates real-time operation by streaming from CSV files. For actual hardware integration:

1. Replace `load_file()` with hardware EMG sensor reading
2. Replace file streaming with live sensor polling
3. Use the same rolling buffer and prediction logic

### Hardware Integration Template

```python
def read_from_hardware(self):
    """Read real-time data from EMG sensors."""
    while self.is_streaming:
        # Read new sample from hardware
        new_sample = emg_hardware.read_8_channels()  # Returns [ch1, ..., ch8]
        
        # Add to buffer
        self.buffer.append(new_sample)
        
        # Predict when buffer full
        if len(self.buffer) == self.window_size:
            prediction = self._predict_from_buffer()
            self._send_to_control_system(prediction)
        
        time.sleep(1/1000)  # 1000 Hz sampling rate
```

## Performance

- **Latency**: ~10ms per prediction (CPU)
- **Throughput**: Up to 100 predictions/second
- **Memory**: ~50MB (model + buffer)
- **CPU**: ~5-10% single core at 10 Hz

## Troubleshooting

### File not found
- Ensure you're in the `Scripts/` directory
- Check file paths with `list` command
- Use full path or correct filename

### Model not found
- Train the model first: `python NN/train.py`
- Ensure `models/final_model_full.pth` exists

### Streaming stops immediately
- File may be too short (<100 samples)
- Check file integrity with `status` command

## Next Steps

- **Multi-file playlist**: Queue multiple files for continuous streaming
- **WebSocket API**: Expose predictions to web/mobile clients
- **Recording mode**: Save predictions to file for analysis
- **Visualization**: Real-time plotting of predictions and confidence
