"""
Real-time EMG Classification Server
Allows dynamic file selection and continuous prediction streaming.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
import argparse

import threading
import time
from collections import deque
import torch
import pandas as pd
import numpy as np

from NN.predict import load_trained_model
from DATA.Data_Conversion import MOVEMENT_LABELS, SEVERITY_LABELS


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "Scripts" / "NN" / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "final_model_full.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "Scripts" / "DATA" / "Example_data"


class EMGRealtimeServer:
    """
    Server that simulates real-time EMG prediction by streaming data from CSV files.
    Allows dynamic file switching via user commands.
    """
    
    def __init__(self, model_path=None, window_size=100, data_dir=None):
        """
        Initialize the real-time server.
        
        Args:
            model_path: Path or filename of trained model checkpoint
            window_size: Number of samples for prediction window
            data_dir: Path to CSV data directory
        """
        print("="*60)
        print("EMG Real-Time Prediction Server")
        print("="*60)
        
        self.window_size = window_size
        self.data_dir = Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR

        # Model state
        self.model = None
        self.device = None
        self.current_model_path = None

        # Load model
        chosen_model = model_path if model_path else str(DEFAULT_MODEL_PATH)
        self._load_model(chosen_model)
        
        # Streaming state
        self.current_file = None
        self.emg_data = None
        self.current_index = 0
        self.buffer = deque(maxlen=window_size)
        self.is_streaming = False
        self.stream_thread = None
        self.auto_loop = True  # Loop file continuously
        
        # Prediction settings
        self.prediction_rate = 20  # Hz (predictions per second)
        self.samples_per_prediction = 10  # New samples to read before next prediction
        
        # Latest prediction
        self.latest_prediction = None
        self.prediction_lock = threading.Lock()
        self.prediction_history = deque(maxlen=50)  # Keep last 50 predictions for smoothing

    def _resolve_model_path(self, model_path):
        """Resolve checkpoint path from absolute, project-relative, or models-dir-relative inputs."""
        candidate = Path(model_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        search_candidates = [
            PROJECT_ROOT / model_path,
            MODELS_DIR / model_path,
            candidate,
        ]

        for path in search_candidates:
            if path.exists():
                return path.resolve()

        available = ", ".join(self.list_available_models(print_output=False)[:8])
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Looked in project root and {MODELS_DIR}. "
            f"Available examples: {available}"
        )

    def _load_model(self, model_path):
        """Load checkpoint and update active model state."""
        resolved = self._resolve_model_path(model_path)
        print(f"\nLoading model from {resolved}...")
        self.model, checkpoint = load_trained_model(str(resolved))
        self.device = next(self.model.parameters()).device
        self.current_model_path = resolved

        print("Model loaded successfully!")
        if 'test_metrics' in checkpoint:
            print(f"Movement Accuracy: {checkpoint['test_metrics']['movement_acc']*100:.2f}%")
            print(f"Severity Accuracy: {checkpoint['test_metrics']['severity_acc']*100:.2f}%")

    def switch_model(self, model_path):
        """Switch active model checkpoint while server is running."""
        was_streaming = self.is_streaming
        if was_streaming:
            self.stop_streaming()

        self._load_model(model_path)

        if was_streaming and self.current_file is not None:
            self.start_streaming()

    def list_available_models(self, directory=None, print_output=True):
        """List available checkpoint files."""
        models_dir = Path(directory).resolve() if directory else MODELS_DIR
        if not models_dir.exists():
            if print_output:
                print(f"Model directory not found: {models_dir}")
            return []

        model_files = sorted([p.name for p in models_dir.glob("*.pth")])

        if print_output:
            print(f"\nAvailable models in {models_dir}:")
            print("-" * 60)
            for i, name in enumerate(model_files, 1):
                print(f"{i:>2}. {name}")
            print("-" * 60)

        return model_files
        
    def load_file(self, csv_path):
        """
        Load a new CSV file for streaming.
        
        Args:
            csv_path: Path to CSV file
        """
        if not os.path.exists(csv_path):
            print(f"Error: File not found - {csv_path}")
            return False
        
        try:
            # Load data
            df = pd.read_csv(csv_path)
            self.emg_data = df.values  # numpy array [samples, 8]
            self.current_file = csv_path
            self.current_index = 0
            self.buffer.clear()
            
            print(f"\n[LOADED] {os.path.basename(csv_path)}")
            print(f"  Samples: {self.emg_data.shape[0]}")
            print(f"  Channels: {self.emg_data.shape[1]}")
            print(f"  Duration: ~{self.emg_data.shape[0]/1000:.2f}s (at 1000 Hz)")
            
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming predictions from current file."""
        if self.current_file is None:
            print("Error: No file loaded. Use 'load <number>' first.")
            return
        
        if self.is_streaming:
            print("Already streaming!")
            return
        
        self.is_streaming = True
        self.current_index = 0
        self.buffer.clear()
        self.prediction_history.clear()
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print(f"\n[STREAMING] Real-time prediction started")
        print(f"  File: {os.path.basename(self.current_file)}")
        print(f"  Update rate: {self.prediction_rate} Hz")
        print(f"  Mode: Continuous loop")
        print("\nPredictions will update in real-time. Use 'load <#>' to switch files.\n")
    
    def stop_streaming(self):
        """Stop streaming predictions."""
        if not self.is_streaming:
            print("Not currently streaming.")
            return
        
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        print("\n[STOPPED] Streaming halted.")
    
    def _stream_loop(self):
        """Internal streaming loop (runs in separate thread) - continuous real-time predictions."""
        last_print_time = time.time()
        print_interval = 0.5  # Update display every 0.5 seconds
        
        while self.is_streaming:
            # Read next batch of samples
            end_idx = min(self.current_index + self.samples_per_prediction, len(self.emg_data))
            new_samples = self.emg_data[self.current_index:end_idx]
            
            # Add to buffer
            for sample in new_samples:
                self.buffer.append(sample)
            
            self.current_index = end_idx
            
            # Loop back to start when reaching end of file
            if self.current_index >= len(self.emg_data):
                self.current_index = 0
            
            # Make prediction if buffer is full
            if len(self.buffer) == self.window_size:
                prediction = self._make_prediction()
                
                # Store in history for smoothing
                self.prediction_history.append(prediction)
                
                # Get smoothed prediction (majority vote over last few predictions)
                smoothed = self._get_smoothed_prediction()
                
                with self.prediction_lock:
                    self.latest_prediction = smoothed
                
                # Print prediction at regular intervals
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    self._print_realtime_prediction(smoothed)
                    last_print_time = current_time
            
            # Sleep to maintain update rate
            time.sleep(1.0 / self.prediction_rate)
    
    def _make_prediction(self):
        """Make single prediction from current buffer."""
        # Convert buffer to tensor
        buffer_array = np.array(list(self.buffer))
        emg_tensor = torch.tensor(buffer_array, dtype=torch.float32).unsqueeze(0)
        emg_tensor = emg_tensor.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            movement_logits, severity_logits = self.model(emg_tensor)
            
            movement_probs = torch.softmax(movement_logits, dim=1)
            severity_probs = torch.softmax(severity_logits, dim=1)
            
            movement_pred = torch.argmax(movement_probs, dim=1).item()
            severity_pred = torch.argmax(severity_probs, dim=1).item()
            
            movement_conf = movement_probs[0, movement_pred].item()
            severity_conf = severity_probs[0, severity_pred].item()
        
        return {
            'movement': MOVEMENT_LABELS[movement_pred],
            'movement_idx': movement_pred,
            'movement_conf': movement_conf,
            'severity': SEVERITY_LABELS[severity_pred],
            'severity_idx': severity_pred,
            'severity_conf': severity_conf,
            'timestamp': time.time()
        }
    
    def _get_smoothed_prediction(self):
        """Get smoothed prediction using majority vote from recent history."""
        if len(self.prediction_history) == 0:
            return None
        
        # Use last N predictions for smoothing
        recent = list(self.prediction_history)[-10:]
        
        # Count occurrences of each movement
        movement_counts = {}
        severity_counts = {}
        
        for pred in recent:
            mov = pred['movement']
            sev = pred['severity']
            movement_counts[mov] = movement_counts.get(mov, 0) + 1
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        # Get most common
        best_movement = max(movement_counts.items(), key=lambda x: x[1])[0]
        best_severity = max(severity_counts.items(), key=lambda x: x[1])[0]
        
        # Get average confidence for the chosen classes
        mov_confs = [p['movement_conf'] for p in recent if p['movement'] == best_movement]
        sev_confs = [p['severity_conf'] for p in recent if p['severity'] == best_severity]
        
        return {
            'movement': best_movement,
            'movement_conf': np.mean(mov_confs) if mov_confs else 0.0,
            'severity': best_severity,
            'severity_conf': np.mean(sev_confs) if sev_confs else 0.0,
            'stability': len(mov_confs) / len(recent)  # How stable is this prediction
        }
    
    def _print_realtime_prediction(self, pred):
        """Print prediction in real-time format with progress bar."""
        if pred is None:
            return
        
        # Calculate progress through file
        progress = (self.current_index / len(self.emg_data)) * 100
        bar_length = 30
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Stability indicator
        stability = pred.get('stability', 1.0)
        stability_icon = '●' if stability > 0.7 else '◐' if stability > 0.4 else '○'
        
        # Clear line and print
        print(f"\r[{bar}] {progress:5.1f}% | "
              f"{stability_icon} Movement: {pred['movement']:20s} ({pred['movement_conf']*100:5.1f}%) | "
              f"Severity: {pred['severity']:10s} ({pred['severity_conf']*100:5.1f}%)", 
              end='', flush=True)
    
    def get_latest_prediction(self):
        """Get the most recent prediction (thread-safe)."""
        with self.prediction_lock:
            return self.latest_prediction
    
    def list_available_files(self, directory=None):
        """List available CSV files for loading."""
        target_dir = Path(directory).resolve() if directory else self.data_dir
        if not target_dir.exists():
            print(f"Directory not found: {target_dir}")
            return []

        csv_files = [f.name for f in target_dir.glob("*.csv")]
        csv_files.sort()
        
        # Parse filenames to show movement and severity
        movement_map = {
            'C1': 'No Movement',
            'C2': 'Wrist Flexion',
            'C3': 'Wrist Extension',
            'C4': 'Wrist Pronation',
            'C5': 'Wrist Supination',
            'C6': 'Chuck Grip',
            'C7': 'Hand Open'
        }
        
        print(f"\n{'#':<4} {'Movement':<20} {'Severity':<10}")
        print("-" * 60)
        for i, filename in enumerate(csv_files, 1):
            # Parse: S1_Hard_C7_R1.csv -> Hard, C7
            parts = filename.replace('.csv', '').split('_')
            severity = parts[1] if len(parts) > 1 else '?'
            movement_code = parts[2] if len(parts) > 2 else '?'
            movement_name = movement_map.get(movement_code, movement_code)
            
            print(f"{i:<4} {movement_name:<20} {severity:<10}")
        print("-" * 60)
        
        return csv_files
    
    def run_interactive(self):
        """Run interactive command loop."""
        print("\n" + "="*60)
        print("Interactive Mode - Available Commands:")
        print("="*60)
        print("  list                    - List available movements")
        print("  load <number>           - Load file by number")
        print("  movement <number>       - Load file and start streaming")
        print("  models                  - List available model checkpoints")
        print("  model <name or path>    - Switch active model checkpoint")
        print("  start                   - Start prediction")
        print("  stop                    - Stop prediction")
        print("  status                  - Show current status")
        print("  help                    - Show this help")
        print("  quit / exit             - Exit server")
        print("="*60)
        
        available_files = self.list_available_files()
        
        while True:
            try:
                command = input("\nserver> ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                
                if cmd in ['quit', 'exit']:
                    self.stop_streaming()
                    print("Shutting down server...")
                    break
                
                elif cmd == 'help':
                    print("\nAvailable Commands:")
                    print("  list          - List available movements")
                    print("  load <#>      - Load file by number")
                    print("  movement <#>  - Load file and start streaming")
                    print("  models        - List available model checkpoints")
                    print("  model <name>  - Switch active model checkpoint")
                    print("  start         - Start prediction")
                    print("  stop          - Stop prediction")
                    print("  status        - Show current status")
                    print("  quit/exit     - Exit server")
                
                elif cmd == 'list':
                    available_files = self.list_available_files()

                elif cmd == 'models':
                    self.list_available_models()

                elif cmd == 'model':
                    if len(parts) < 2:
                        print("Usage: model <checkpoint_name_or_path>")
                        print("Use 'models' to see available checkpoints")
                        continue

                    model_ref = " ".join(parts[1:])
                    try:
                        self.switch_model(model_ref)
                    except Exception as e:
                        print(f"Error switching model: {e}")
                
                elif cmd in ['load', 'movement']:
                    if len(parts) < 2:
                        print(f"Usage: {cmd} <number>")
                        print("Use 'list' to see available files")
                        continue
                    
                    # Load by number only (don't stop streaming - hot swap!)
                    if parts[1].isdigit():
                        idx = int(parts[1]) - 1
                        if 0 <= idx < len(available_files):
                            filepath = str(self.data_dir / available_files[idx])
                            
                            # Load new file
                            df = pd.read_csv(filepath)
                            self.emg_data = df.values
                            self.current_file = filepath
                            self.current_index = 0
                            
                            # Clear buffers but keep streaming
                            if self.is_streaming:
                                self.buffer.clear()
                                self.prediction_history.clear()
                                print(f"\n[SWITCHED] Now predicting: {os.path.basename(filepath)}\n")
                            else:
                                print(f"\n[LOADED] {os.path.basename(filepath)}")
                                if cmd == 'load':
                                    print("Type 'start' to begin real-time predictions.")
                                elif cmd == 'movement':
                                    self.start_streaming()
                        else:
                            print(f"Invalid number. Choose 1-{len(available_files)}")
                    else:
                        print("Please use a number. Use 'list' to see options.")
                
                elif cmd == 'start':
                    self.start_streaming()
                
                elif cmd == 'stop':
                    self.stop_streaming()
                
                elif cmd == 'status':
                    status_pred = self.get_latest_prediction()
                    print("\n" + "="*60)
                    print("Server Status")
                    print("="*60)
                    print(f"  Current file: {os.path.basename(self.current_file) if self.current_file else 'None'}")
                    print(f"  Current model: {self.current_model_path.name if self.current_model_path else 'None'}")
                    print(f"  Streaming: {'Yes' if self.is_streaming else 'No'}")
                    if self.current_file and self.emg_data is not None:
                        progress = (self.current_index / len(self.emg_data)) * 100
                        print(f"  Progress: {progress:.1f}%")
                    print(f"  Update rate: {self.prediction_rate} Hz")
                    print(f"  Window size: {self.window_size} samples")
                    
                    if status_pred:
                        print(f"\n  Current prediction:")
                        print(f"    Movement: {status_pred['movement']} ({status_pred['movement_conf']*100:.1f}%)")
                        print(f"    Severity: {status_pred['severity']} ({status_pred['severity_conf']*100:.1f}%)")
                        if 'stability' in status_pred:
                            print(f"    Stability: {status_pred['stability']*100:.0f}%")
                    print("="*60)
                
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run real-time EMG prediction server")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Checkpoint path or filename (searched in Scripts/NN/models)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Number of samples for prediction window",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing CSV files for streaming",
    )
    args = parser.parse_args()

    server = EMGRealtimeServer(
        model_path=args.model,
        window_size=args.window_size,
        data_dir=args.data_dir,
    )
    
    server.run_interactive()


if __name__ == "__main__":
    main()
