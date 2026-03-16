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

---

## Update 4.0 - Full Simulation Pipeline & Subprocess Inference (March 16, 2026)

### Achievements
- ✓ Implemented subprocess-based NN inference worker isolating PyTorch from myosuite conda env
- ✓ Built full interactive simulation pipeline in `simulation/run_nn.py`
- ✓ Added two-level model picker (architecture + checkpoint variant)
- ✓ Integrated joint tuning system with smoothstep qpos interpolation
- ✓ Created persistent session API for external callers (`create_nn_session`, etc.)
- ✓ Added passive viewer with camera presets and skin toggle via `viewer_utils.py`
- ✓ Implemented severity-based transition timing for realistic movement speed

### Inference Worker Subprocess (`Scripts/NN/inference_worker.py`)

The NN inference now runs in a **separate subprocess** under `Scripts/.venv` (where PyTorch is installed), while the simulation environment runs under the myosuite conda environment. The two processes communicate via **newline-delimited JSON over stdin/stdout**.

**Protocol**:
```
Parent → worker  stdin  : {"window": [[f, f, ...] * 8] * window_size}
                          | {"command": "exit"}
Worker → parent  stdout : {"status": "ready"}              (once, on model load)
                        : {"movement_name": str,
                           "movement_pred": int,
                           "movement_confidence": float,
                           "severity_name": str,
                           "severity_pred": int,
                           "severity_confidence": float}   (per window)
```

Worker stderr is forwarded directly to the parent terminal for debug output — it never pollutes the JSON channel. The worker suppresses stdout during model loading to prevent spurious output before the `ready` signal.

### Interactive Model Picker

`_pick_model()` in `run_nn.py` presents a two-level menu:

**Level 1 — Architecture**:
- NN-A: Full CNN+GRU (`full`)
- NN-B: Standard CNN (`standard_cnn`)
- NN-C: Lightweight CNN (`lightweight`)

**Level 2 — Checkpoint Variant**:
- `best`: lowest validation-loss checkpoint
- `final`: end-of-training checkpoint

Each option shows `✓` or `missing` based on file presence. Navigation supports `b` to go back and `q` to quit.

### Interactive CSV / Movement Picker

`_prompt_csv()` provides two input modes:
- **Input mode**: manually enter a CSV file path
- **Control mode**: select movement class (0–6) and severity (Light/Medium/Hard) to auto-resolve the matching example CSV (`S1_{Severity}_C{idx+1}_R1.csv`)

### Runtime Keyboard Controls

During CSV replay the following single-key commands are handled non-blocking (Windows only via `msvcrt`):
- `b` — abort current replay and return to movement selection
- `q` — quit the simulation entirely

### End-of-CSV Prompt

After each replay finishes the user is shown:
```
s) Same movement again
n) New movement / severity
v) Visual settings
q) Quit
```
The worker and environment are **not restarted** between replays.

### Joint Tuning Integration

When `Output/joint_tuning/<Movement>.json` files exist (exported from the tuning sandbox), `run_nn.py` drives the arm using **joint position interpolation** rather than actuator action-space commands:

1. `_load_exported_joint_targets(movement_name)` — reads `target_joint_qpos` (exact values) or `target_jnt_range` (midpoint fallback) from the JSON file
2. `_resolve_joint_qpos_targets(env, joint_targets)` — maps joint names to qpos indices, clamping to joint limits
3. `_apply_joint_targets_interp(env, start_qpos, target_qpos, phase)` — interpolates from the start pose to the target pose using smoothstep: $f(x) = x^2(3 - 2x)$

If no tuning file is found, the system falls back to LUT-based actuator control via `results_to_action()`.

### Persistent Session API

For external callers (e.g., `gui_controller.py`), three functions manage the simulation lifecycle:
- `create_nn_session(model_path, stride, steps_per_window, print_every)` — spawns the worker, opens the environment and passive viewer, returns a session dict
- `run_csv_once_in_session(session, csv_path, stop_check, log_fn, window_log_fn)` — replays one CSV in an existing session; returns `'completed'`, `'aborted'`, or `'viewer_closed'`
- `close_nn_session(session)` — tears down worker, viewer, and environment

### Passive Viewer (`viewer_utils.py`)

A shared `viewer_utils.py` module provides:
- `open_passive_viewer(env)` / `close_passive_viewer(viewer)` / `sync_passive_viewer(viewer)` — lifecycle management
- `run_viewer_submenu(env, viewer)` — interactive settings menu (camera presets, skin toggle)
- `CAMERA_PRESETS` — named presets: `default`, `front`, `side`, `top`, `close`
- Supports native mujoco bindings, dm_control wrappers, and legacy mujoco_py via `_get_mj_model_data()`

### Severity-Based Transition Timing

`NN_SEVERITY_TRANSITION_DURATION_FACTOR` (in `config.py`) scales how many simulation steps are used for the transition phase:

| Severity | Factor | Effect |
|----------|--------|--------|
| Light    | 4.0    | Slow, gentle transition |
| Medium   | 2.0    | Moderate speed |
| Hard     | 0.45   | Fast, forceful transition |

The transition step count is: `transition_steps = round(steps_per_window × NN_TRANSITION_BASE_FRACTION × duration_factor)`, clipped to `NN_TRANSITION_MIN_STEPS`.

### Config Updates (`simulation/config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `DEFAULT_ENV_ID` | `myoHandReachFixed-v0` | Single env for all movement tasks |
| `NN_WINDOW_SIZE` | 150 | Increased from 100 for smoother predictions |
| `NN_INFERENCE_STRIDE` | 25 | Steps between windows |
| `NN_TRANSITION_BASE_FRACTION` | 0.80 | Fraction of steps used for transition |
| `NN_TRANSITION_MIN_STEPS` | 2 | Floor on transition steps |
| `NN_ACTIVE_STEP_SLEEP_FACTOR` | 1.0 | Multiplier on `env.dt` during active movement |
| `NN_NO_MOVEMENT_HOLD_SLEEP_FACTOR` | 1.0 | Multiplier on `env.dt` during No_Movement hold |
| `VENV_PYTHON_PATH` | `Scripts/.venv/Scripts/python.exe` | Overridable via `NN_VENV_PYTHON` env var |
| `NN_INFERENCE_WORKER_PATH` | `Scripts/NN/inference_worker.py` | Worker script path |

### Next Steps
- [ ] Collect performance metrics for each model architecture under real-time conditions
- [ ] Tune `NN_SEVERITY_TRANSITION_DURATION_FACTOR` values against physical arm response
- [ ] Validate joint tuning JSON files for all 6 movement classes
- [ ] Extend `viewer_utils.py` with recording/export capability
- [ ] Evaluate multi-subject generalisation with new EMG recordings
Saved: DATA/Results/Light_No_Movement.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Light_Wrist_Flexion.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Light_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Light_Wrist_Pronation.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Light_Wrist_Supination.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Light_Chuck_Grip.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Light_Hand_Open.png - ✓ CORRECT
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Pronation.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Wrist_Supination.png - ✓ CORRECT
Saved: DATA/Results/Medium_Chuck_Grip.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Hand_Open.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_No_Movement.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Pronation.png - ✓ CORRECT
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Pronation.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Wrist_Supination.png - ✓ CORRECT
Saved: DATA/Results/Medium_Chuck_Grip.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Hand_Open.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_No_Movement.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Pronation.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Wrist_Supination.png - ✓ CORRECT
Saved: DATA/Results/Medium_Chuck_Grip.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Hand_Open.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_No_Movement.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Medium_Wrist_Pronation.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Wrist_Supination.png - ✓ CORRECT
Saved: DATA/Results/Medium_Chuck_Grip.png - ✗ INCORRECT
Saved: DATA/Results/Medium_Hand_Open.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_No_Movement.png - ⚠ Movement OK, Severity Wrong
Saved: DATA/Results/Hard_Wrist_Flexion.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Extension.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Pronation.png - ✓ CORRECT
Saved: DATA/Results/Hard_Wrist_Supination.png - ✗ INCORRECT
Saved: DATA/Results/Hard_Chuck_Grip.png - ✗ INCORRECT
Saved: DATA/Results/Hard_Hand_Open.png - ✓ CORRECT